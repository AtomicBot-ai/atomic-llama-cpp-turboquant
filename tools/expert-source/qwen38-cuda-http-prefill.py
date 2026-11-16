#!/usr/bin/env python3
"""One selected Qwen CUDA long HTTP prefill control, never an aggregate gate."""
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
p.add_argument('--prompt', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--mode', choices=['native', 'stream'], required=True)
p.add_argument('--port', type=int, default=19163)
p.add_argument('--series', action='store_true', help='Also repeat the prompt and change one token in the same slot')
p.add_argument('--disk-prefix-dir', type=Path, help='Enable the disk prefix option on the streaming server')
p.add_argument('--capture-logits', action='store_true', help='Require the isolated sampling capture build')
p.add_argument('--native-reference-rebuild', action='store_true',
               help='Reuse the captured native reference after a streaming-only rebuild')
a = p.parse_args()
if a.disk_prefix_dir and (not a.series or not a.capture_logits):
    p.error('--disk-prefix-dir requires --series and --capture-logits')
work = a.output.resolve()
work.mkdir(parents=True, exist_ok=True)
slots = work / a.mode
slots.mkdir(exist_ok=False)
url = f'http://127.0.0.1:{a.port}'
prompt = a.prompt.read_text()
result = {'mode': a.mode, 'runtime_reserve_bytes': 2 << 30,
          'prompt_sha256': hashlib.sha256(a.prompt.read_bytes()).hexdigest()}
assert result['prompt_sha256'] == 'ff564330c036acd2575b9c4174df6f1223b09d2e55fdc1f3a7d1ced3a0f7ea2e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def equal_file(x, y):
    with Path(x).open('rb') as f, Path(y).open('rb') as g:
        while True:
            left, right = f.read(1 << 20), g.read(1 << 20)
            if left != right:
                return False
            if not left:
                return True


def call(route, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url + route, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def metrics(label):
    with urllib.request.urlopen(url + '/metrics', timeout=30) as response:
        text = response.read().decode()
    (work / f'{a.mode}-{label}.metrics').write_text(text)
    values = {k: int(v) for k, v in re.findall(r'llamacpp:engine_memory_(\w+) ([0-9]+)\n', text)}
    if a.mode == 'native':
        assert not values, 'ordinary path exposed managed-engine metrics'
    else:
        for domain, cap in [('ram', 4 << 30), ('vram', 2 << 30)]:
            assert values[domain + '_applied_bytes'] == cap
            assert values[domain + '_used_bytes'] <= cap and values[domain + '_high_water_bytes'] <= cap
    return values


def save(label):
    filename = label + '.slot'
    call('/slots/0?action=save', {'filename': filename})
    path = slots / filename
    assert path.is_file() and path.stat().st_size > 12
    result.setdefault('states', {})[label] = dict(bytes=path.stat().st_size, sha256=sha(path))


@contextmanager
def server():
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(('127.0.0.1', a.port))
    binary = a.server.resolve()
    cmd = [str(binary), '-m', str(a.model.resolve()), '--device', 'CUDA0',
           '-ngl', '22', '-c', '4096', '-b', '128', '-ub', '8', '-np', '1',
           '-t', '4', '-tb', '4', '-fit', 'off', '--no-warmup', '--cache-ram', '0',
           '--ctx-checkpoints', '0', '--metrics', '--slots', '--slot-save-path', str(slots),
           '--host', '127.0.0.1', '--port', str(a.port)]
    if a.mode == 'stream':
        cmd += ['--engine-ram-mib', '4096', '--engine-vram-mib', '2048']
        if a.disk_prefix_dir:
            cmd += ['--engine-prefix-checkpoints', str(a.disk_prefix_dir.resolve())]
    else:
        cmd += ['--cpu-moe', '--moe-cache-mib', '4096', '--moe-cache-cpu-share', '0']
    result['command'] = cmd
    result['binaries'] = {binary.name: sha(binary)}
    for name in ['libllama.so', 'libllama-common.so', 'libllama-server-impl.so']:
        result['binaries'][name] = sha(binary.parent / name)
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    env.pop('GGML_CUDA_ENABLE_UNIFIED_MEMORY', None)
    env.pop('PREFIX_PROBE_LOGITS', None)
    if a.capture_logits:
        capture = work / (a.mode + '.f32')
        with capture.open('xb'):
            pass
        env['PREFIX_PROBE_LOGITS'] = str(capture)
    log_path = work / (a.mode + '.log')
    with log_path.open('w') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited {process.returncode}: ' + log_path.read_text()[-2500:])
                try:
                    if call('/health', timeout=5).get('status') == 'ok':
                        break
                except (OSError, urllib.error.URLError):
                    time.sleep(.2)
            else:
                raise TimeoutError('server startup')
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()



def completion(input_prompt=prompt, *, cache_prompt=False, label=None):
    body = dict(prompt=input_prompt, n_predict=4, temperature=0, seed=42,
                cache_prompt=cache_prompt, id_slot=0, return_tokens=True,
                stream=True, return_progress=True, ignore_eos=True)
    run = result if label is None else result.setdefault('series', {}).setdefault(label, {})
    run['request'] = body
    events, tokens, content = [], [], []
    run['events'] = events
    start = time.monotonic()
    ttft = None
    request = urllib.request.Request(url + '/completion', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    final = None
    journal_name = a.mode if label is None else f'{a.mode}-{label}'
    with (work / (journal_name + '.events.jsonl')).open('w') as journal:
        with urllib.request.urlopen(request, timeout=1800) as response:
            for line in response:
                line = line.decode().strip()
                if not line.startswith('data:'):
                    continue
                payload = line[5:].strip()
                if payload == '[DONE]':
                    break
                event = json.loads(payload)
                events.append(event)
                journal.write(json.dumps(event) + '\n')
                journal.flush()
                if 'error' in event:
                    raise RuntimeError(event)
                if 'prompt_progress' in event:
                    print(a.mode, 'prefill', event['prompt_progress'], flush=True)
                if event.get('stop'):
                    final = event
                elif 'prompt_progress' not in event:
                    generated = event.get('tokens', [])
                    if generated and ttft is None:
                        ttft = time.monotonic() - start
                    tokens.extend(generated)
                    content.append(event.get('content', ''))
    run['elapsed_seconds'] = time.monotonic() - start
    run['ttft_seconds'] = ttft
    run['tokens'] = tokens
    run['content'] = ''.join(content)
    run['response'] = final
    print(a.mode, label or 'initial', 'completed', tokens, flush=True)
    return run


def check_completion():
    final, tokens, events = result['response'], result['tokens'], result['events']
    assert final is not None, 'missing final event'
    assert final['tokens_evaluated'] == 3481 and final['tokens_predicted'] == 4
    assert len(tokens) == 4 and final['stop_type'] == 'limit' and not final['truncated']
    progress = [e['prompt_progress'] for e in events if 'prompt_progress' in e]
    assert progress and all(p['total'] == 3481 and p['cache'] == 0 for p in progress)
    processed = [p['processed'] for p in progress]
    result['processed_boundaries'] = processed
    assert processed == [0] + list(range(128, 3481, 128)) + [3481], processed
    print(a.mode, 'validated', tokens, flush=True)


def check_series_completion(run, expected_max_cache):
    final = run['response']
    assert final and final['stop_type'] == 'limit' and not final['truncated'], final
    assert final['tokens_predicted'] == len(run['tokens']) == 4, final
    timings = final['timings']
    cached = timings['cache_n']
    assert 0 <= cached <= expected_max_cache, timings
    assert timings['prompt_n'] + cached == 3481, timings
    progress = [event['prompt_progress'] for event in run['events'] if 'prompt_progress' in event]
    assert progress and all(event['cache'] == cached and event['total'] == 3481 for event in progress)
    boundaries = [event['processed'] for event in progress]
    assert boundaries == [cached] + list(range(cached + 128, 3481, 128)) + [3481], boundaries
    run['processed_boundaries'] = boundaries
    print(a.mode, 'cached', cached, 'evaluated', timings['prompt_n'], flush=True)


def series():
    original = result['prompt_tokens']
    midpoint = len(original) // 2
    modified = original.copy()
    modified[midpoint] = original[midpoint - 1]
    assert modified[midpoint] != original[midpoint]
    result['edit'] = {'index': midpoint, 'before': original[midpoint], 'after': modified[midpoint]}
    for label, tokens, max_cache in [('repeat', original, len(original) - 1), ('edited', modified, midpoint)]:
        run = completion(tokens, cache_prompt=True, label=label)
        run['completion_metrics'] = metrics(label)
        save(label)
        run['save_metrics'] = metrics(label + '-save')
        assert not call('/slots')[0]['is_processing']
        check_series_completion(run, max_cache)
        if a.disk_prefix_dir and a.mode == 'stream':
            expected = 3456 if label == 'repeat' else 1664
            assert run['response']['timings']['cache_n'] == expected, (label, run['response']['timings'])
        if label == 'repeat':
            assert run['tokens'] == result['tokens'] and run['content'] == result['content']
        if a.mode == 'stream':
            native_run = native['series'][label]
            comparable = ['tokens', 'content']
            if not a.disk_prefix_dir:
                comparable.append('processed_boundaries')
            for key in comparable:
                assert run[key] == native_run[key], (label, key)
            assert equal_file(slots / (label + '.slot'), work / 'native' / (label + '.slot')), (label, 'state differs')
            run['ordinary_tokens_and_state_bitwise_equal'] = True


try:
    with server():
        result['prompt_tokens'] = call('/tokenize', dict(content=prompt, add_special=True,
                                                       parse_special=True))['tokens']
        assert len(result['prompt_tokens']) == 3481
        completion(cache_prompt=bool(a.disk_prefix_dir and a.mode == 'stream'))
        result['completion_metrics'] = metrics('completion')
        save('long')
        result['save_metrics'] = metrics('save')
        check_completion()
        assert not call('/slots')[0]['is_processing']
        if a.mode == 'stream':
            native = json.loads((work / 'native.json').read_text())
            assert native['passed']
            comparable = ['prompt_sha256', 'prompt_tokens', 'tokens', 'content', 'processed_boundaries']
            if a.native_reference_rebuild:
                assert a.disk_prefix_dir and a.capture_logits and native.get('logits')
                result['native_reference_binaries'] = native['binaries']
            else:
                comparable.append('binaries')
            for key in comparable:
                assert result[key] == native[key], key
            assert equal_file(slots / 'long.slot', work / 'native' / 'long.slot'), 'state differs'
            result['ordinary_tokens_and_state_bitwise_equal'] = True
        if a.series:
            series()
        if a.capture_logits:
            captured = work / (a.mode + '.f32')
            rows = (3 if a.series else 1) * 4
            assert captured.stat().st_size == rows * 248320 * 4, 'sampling capture incomplete'
            result['logits'] = {'rows': rows, 'bytes': captured.stat().st_size, 'sha256': sha(captured)}
            if a.mode == 'stream':
                assert equal_file(captured, work / 'native.f32'), 'full pre-sampler logits differ'
                result['ordinary_logits_bitwise_equal'] = True
        result['passed'] = True
finally:
    (work / (a.mode + '.json')).write_text(json.dumps(result, indent=2) + '\n')
