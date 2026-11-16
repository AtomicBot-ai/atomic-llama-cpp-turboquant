#!/usr/bin/env python3
"""One selected Qwen CUDA long HTTP slot restoration control, never an aggregate gate."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--server', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--prefill', type=Path, required=True, help='Evidence directory from the paired HTTP prefill runs')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--mode', choices=['native', 'stream'], required=True)
p.add_argument('--port', type=int, default=19164)
a = p.parse_args()
work = a.output.resolve()
work.mkdir(parents=True, exist_ok=True)
slots = work / a.mode
slots.mkdir(exist_ok=False)
url = f'http://127.0.0.1:{a.port}'
prior = json.loads((a.prefill / (a.mode + '.json')).read_text())
assert prior['passed'] and len(prior['prompt_tokens']) == 3481
assert prior['tokens'] == [271, 248068, 271, 248069]
assert prior['response']['tokens_cached'] == 3484
result = {'mode': a.mode, 'runtime_reserve_bytes': 2 << 30,
          'input_state': prior['states']['long'],
          'diagnostic_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


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
    else:
        cmd += ['--cpu-moe', '--moe-cache-mib', '4096', '--moe-cache-cpu-share', '0']
    result['command'] = cmd
    result['binaries'] = {binary.name: sha(binary)}
    for name in ['libllama.so', 'libllama-common.so', 'libllama-server-impl.so']:
        result['binaries'][name] = sha(binary.parent / name)
    assert result['binaries'] == prior['binaries'], 'use the binaries that saved the source slot'
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    env.pop('GGML_CUDA_ENABLE_UNIFIED_MEMORY', None)
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



try:
    original = a.prefill / a.mode / 'long.slot'
    assert original.stat().st_size == result['input_state']['bytes']
    assert sha(original) == result['input_state']['sha256']
    shutil.copyfile(original, slots / 'long.slot')
    with server():
        result['startup_metrics'] = metrics('startup')
        result['restore'] = call('/slots/0?action=restore', {'filename': 'long.slot'})
        result['restore_metrics'] = metrics('restore')
        save('restored')
        result['round_trip_metrics'] = metrics('round-trip')
        result['restored_bitwise_equal'] = equal_file(original, slots / 'restored.slot')
        print(a.mode, 'restore', result['restore'], 'exact', result['restored_bitwise_equal'], flush=True)

        # The last generated token has not been evaluated. Send the exact IDs
        # so retokenizing the visible completion cannot change the prefix.
        body = dict(prompt=prior['prompt_tokens'] + prior['tokens'], n_predict=4,
                    temperature=0, seed=42, id_slot=0, return_tokens=True,
                    cache_prompt=True, ignore_eos=True)
        result['request'] = body
        response = call('/completion', body, timeout=1800)
        result['continuation'] = response
        result['continuation_metrics'] = metrics('continuation')
        save('continued')
        result['save_metrics'] = metrics('save')
        print(a.mode, 'continuation', response['tokens'], response['timings'], flush=True)

        assert result['restored_bitwise_equal'], 'restore round trip changed slot bytes'
        assert response['tokens_evaluated'] == 3485 and not response['truncated']
        assert response['stop_type'] == 'limit' and len(response['tokens']) == response['tokens_predicted'] == 4
        assert response['timings']['cache_n'] == 3484 and response['timings']['prompt_n'] == 1
        assert not call('/slots')[0]['is_processing']
        if a.mode == 'stream':
            native = json.loads((work / 'native.json').read_text())
            assert native['passed'] and native['request'] == body
            for key in ('tokens', 'content', 'tokens_cached', 'stop_type'):
                assert response[key] == native['continuation'][key], key
            assert equal_file(slots / 'continued.slot', work / 'native' / 'continued.slot'), 'continued state differs'
            result['ordinary_continuation_and_state_bitwise_equal'] = True
        result['passed'] = True
finally:
    (work / (a.mode + '.json')).write_text(json.dumps(result, indent=2) + '\n')
