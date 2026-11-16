#!/usr/bin/env python3
"""Explicitly selected single-slot HTTP checks; excluded from aggregate verification."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--server', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--phase', choices=['baseline', 'lifecycle', 'ordinary-lifecycle', 'lfm-control'], required=True)
p.add_argument('--cases', nargs='+', choices=['stop', 'cancel', 'restore'], default=['stop', 'cancel', 'restore'])
p.add_argument('--port', type=int, default=19159)
p.add_argument('--tag', default='selected')
p.add_argument('--lfm-reference', type=Path)
a = p.parse_args()
work = a.output.resolve()
work.mkdir(parents=True, exist_ok=True)
url = f'http://127.0.0.1:{a.port}'
lfm = a.phase == 'lfm-control'
prompt = 'Finish this sentence: The capital of France is' if lfm else 'The capital of France is'
name = a.phase + '-' + a.tag
result_path = work / (name + '.json')
if result_path.exists():
    p.error('selected result already exists; use a new tag for a justified follow-up')
result = {'phase': a.phase, 'cases': a.cases if a.phase in ['lifecycle', 'ordinary-lifecycle'] else [], 'prompt': prompt}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def equal_file(x, y):
    with Path(x).open('rb') as fx, Path(y).open('rb') as fy:
        while True:
            bx, by = fx.read(1024 * 1024), fy.read(1024 * 1024)
            if bx != by:
                return False
            if not bx:
                return True


def req(route, body=None, timeout=180):
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(url + route, data=data,
        headers={'Content-Type': 'application/json'}), timeout=timeout)


def call(route, body=None):
    with req(route, body) as response:
        return json.load(response)


def completion(text=prompt, **extra):
    body = dict(prompt=text, n_predict=4, temperature=0, seed=42, id_slot=0,
                cache_prompt=False, return_tokens=True)
    body.update(extra)
    return call('/completion', body)


def metrics(label, streaming=None):
    if streaming is None:
        streaming = a.phase not in ['baseline', 'ordinary-lifecycle']
    with req('/metrics') as response:
        text = response.read().decode()
    (work / f'{name}-{label}.metrics').write_text(text)
    if not streaming:
        assert 'llamacpp:engine_memory_' not in text
        return {}
    values = {k: int(v) for k, v in re.findall(r'llamacpp:engine_memory_(\w+) ([0-9]+)\n', text)}
    for domain, cap in [('ram', 4 << 30), ('vram', 2 << 30)]:
        assert values[domain + '_applied_bytes'] == cap, values
        assert values[domain + '_used_bytes'] <= cap and values[domain + '_high_water_bytes'] <= cap, values
    return values


def save(slots, filename):
    info = call('/slots/0?action=save', {'filename': filename})
    path = slots / filename
    assert path.is_file() and path.stat().st_size > 12, info
    result.setdefault('states', {})[filename] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return path


@contextmanager
def server():
    # Never attach to or terminate an unrelated listener.
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(('127.0.0.1', a.port))
    slots = work / name
    slots.mkdir(exist_ok=False)
    streaming = a.phase not in ['baseline', 'ordinary-lifecycle']
    cmd = [str(a.server.resolve()), '-m', str(a.model.resolve()), '--device', 'CUDA0',
           '-ngl', '14' if lfm else '22', '-c', '256' if lfm else '4096', '-b', '128', '-ub', '8',
           '-np', '1', '-t', '4', '-tb', '4', '-fit', 'off', '--no-warmup', '--cache-ram', '0',
           '--ctx-checkpoints', '0', '--metrics', '--slots', '--slot-save-path', str(slots),
           '--host', '127.0.0.1', '--port', str(a.port)]
    if streaming:
        cmd += ['--engine-ram-mib', '4096', '--engine-vram-mib', '2048']
    else:
        cmd += ['--cpu-moe', '--moe-cache-mib', '4096', '--moe-cache-cpu-share', '0']
    result['command'] = cmd
    result['server_sha256'] = sha(a.server)
    env = os.environ.copy()
    env.pop('GGML_CUDA_ENABLE_UNIFIED_MEMORY', None)
    env.pop('LD_PRELOAD', None)
    log_path = work / (name + '.log')
    with log_path.open('w') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited {process.returncode}: ' + log_path.read_text()[-2500:])
                try:
                    if call('/health').get('status') == 'ok':
                        break
                except (OSError, urllib.error.URLError):
                    time.sleep(.2)
            else:
                raise TimeoutError('server startup')
            yield slots
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


try:
    with server() as slots:
        result['initial'] = completion()
        initial = save(slots, 'initial.slot')
        result['initial_metrics'] = metrics('initial', a.phase not in ['baseline', 'ordinary-lifecycle'])
        print('initial', result['initial']['tokens'], flush=True)
        if lfm:
            assert result['initial']['tokens'] == [4741, 20, 309, 278]
            assert a.lfm_reference and equal_file(initial, a.lfm_reference), 'LFM control changed'
        elif a.phase == 'baseline':
            result['continuation'] = completion(prompt + result['initial']['content'], cache_prompt=True)
            save(slots, 'continued.slot')
            (work / 'baseline.json').write_text(json.dumps(result, indent=2) + '\n')
        else:
            baseline = json.loads((work / 'baseline.json').read_text())
            reference = Path(baseline['command'][baseline['command'].index('--slot-save-path') + 1])
            lifecycle_reference = None
            if a.phase == 'lifecycle':
                ordinary = json.loads((work / 'ordinary-lifecycle-selected.json').read_text())
                lifecycle_reference = Path(ordinary['command'][ordinary['command'].index('--slot-save-path') + 1])
            assert result['initial']['tokens'] == baseline['initial']['tokens']
            assert equal_file(initial, reference / 'initial.slot'), 'ordinary/streaming slot differs'
            if 'stop' in a.cases:
                text = result['initial']['content']
                stop = 'Paris' if 'Paris' in text else text.strip()[:3]
                assert stop
                result['stop'] = completion(n_predict=16, stop=[stop])
                assert result['stop']['stop_type'] == 'word' and stop not in result['stop']['content'], result['stop']
                result['stop_metrics'] = metrics('stop')
                print('stop passed', flush=True)
            if 'cancel' in a.cases:
                with req('/completion', dict(prompt=prompt, n_predict=128, temperature=0, id_slot=0,
                         cache_prompt=False, ignore_eos=True, stream=True, return_tokens=True)) as response:
                    for line in response:
                        if line.startswith(b'data:'):
                            event = json.loads(line[5:])
                            if event.get('content') and not event.get('stop'):
                                result['cancel_first'] = event
                                break
                    else:
                        raise AssertionError('no SSE content before cancellation')
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if not call('/slots')[0]['is_processing']:
                        break
                    time.sleep(.1)
                else:
                    raise AssertionError('slot stayed busy after SSE disconnect')
                result['after_cancel'] = completion()
                assert result['after_cancel']['tokens'] == result['initial']['tokens']
                result['after_cancel_matches_cold_state'] = equal_file(save(slots, 'after_cancel.slot'), initial)
                if lifecycle_reference:
                    assert equal_file(slots / 'after_cancel.slot', lifecycle_reference / 'after_cancel.slot')
                    result['after_cancel_matches_ordinary_state'] = True
                result['cancel_metrics'] = metrics('cancel')
                print('cancel and continuation passed', flush=True)
            if 'restore' in a.cases:
                if 'cancel' not in a.cases:
                    result['before_restore'] = completion()
                before = save(slots, 'before_restore.slot')
                follow = prompt + result['initial']['content']
                result['continuation'] = completion(follow, cache_prompt=True)
                continued = save(slots, 'continued.slot')
                assert result['continuation']['tokens'] == baseline['continuation']['tokens']
                result['continuation_matches_cold_state'] = equal_file(continued, reference / 'continued.slot')
                if lifecycle_reference:
                    assert equal_file(continued, lifecycle_reference / 'continued.slot')
                    result['continuation_matches_ordinary_state'] = True
                result['erase'] = call('/slots/0?action=erase', {})
                result['restore'] = call('/slots/0?action=restore', {'filename': before.name})
                assert equal_file(save(slots, 'restored.slot'), before)
                result['restored_continuation'] = completion(follow, cache_prompt=True)
                assert result['restored_continuation']['tokens'] == result['continuation']['tokens']
                assert result['restored_continuation']['timings']['cache_n'] > 0
                assert equal_file(save(slots, 'restored_continued.slot'), continued)
                result['restore_metrics'] = metrics('restore')
                print('restore and exact continuation passed', flush=True)
        result['passed'] = True
finally:
    result_path.write_text(json.dumps(result, indent=2) + '\n')
