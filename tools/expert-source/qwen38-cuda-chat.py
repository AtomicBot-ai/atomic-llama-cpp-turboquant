#!/usr/bin/env python3
"""Selected Qwen CUDA chat controls, with optional two-turn logit capture."""
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
p.add_argument('--mode', choices=['native', 'stream'], required=True)
p.add_argument('--port', type=int, default=19162)
p.add_argument('--follow-up', action='store_true', help='Run two chat turns with the isolated sampling capture build')
fixture = p.add_mutually_exclusive_group()
fixture.add_argument('--long-prompt', type=Path, help='Use the field-archive fixture and time both chat turns over SSE')
fixture.add_argument('--thinking', action='store_true', help='Use a two-turn arithmetic dialogue with thinking over SSE')
fixture.add_argument('--tool-call', action='store_true', help='Use one SSE function call followed by a tool result')
a = p.parse_args()
if (a.long_prompt or a.thinking or a.tool_call) and not a.follow_up:
    p.error('--long-prompt, --thinking and --tool-call require --follow-up')
work = a.output.resolve()
work.mkdir(parents=True, exist_ok=True)
slots = work / a.mode
slots.mkdir(exist_ok=False)
url = f'http://127.0.0.1:{a.port}'
chat_body = dict(messages=[dict(role='user', content='Say only OK.')],
                 chat_template_kwargs=dict(enable_thinking=False), temperature=0,
                 seed=42, max_tokens=16, id_slot=0, cache_prompt=False,
                 return_tokens=True, verbose=True)
result = {'mode': a.mode, 'chat_request': chat_body, 'runtime_reserve_bytes': 2 << 30}


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
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    env.pop('GGML_CUDA_ENABLE_UNIFIED_MEMORY', None)
    env.pop('PREFIX_PROBE_LOGITS', None)
    if a.follow_up:
        capture = work / (a.mode + '.f32')
        with capture.open('xb'):
            pass
        env['PREFIX_PROBE_LOGITS'] = str(capture)
        result['library_path'] = env.get('LD_LIBRARY_PATH', '')
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


def check_eos(response):
    assert response['stop_type'] == 'eos', response
    assert not response['stopping_word'] and not response['truncated'], response
    assert response['generation_settings']['stop'] == [], response
    assert response['tokens_predicted'] == len(response['tokens']) < 16, response
    assert response['tokens_evaluated'] == len(result['prompt_tokens']), response
    assert response['prompt'] == result['rendered_prompt'], response
    assert response['content'] == 'OK', response


def streamed_chat(request):
    body = dict(request, stream=True, stream_options={'include_usage': True}, return_progress=True)
    http = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                  headers={'Content-Type': 'application/json'})
    summary = {}
    deltas = {'content': [], 'reasoning_content': []}
    tool_call = {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}}
    first_delta = {}
    progress = []
    start = time.monotonic()
    with urllib.request.urlopen(http, timeout=1800) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            event = line[6:].strip()
            if event == b'[DONE]':
                break
            chunk = json.loads(event)
            assert 'error' not in chunk, chunk
            if 'prompt_progress' in chunk:
                progress.append(chunk['prompt_progress'])
            for field in ('__verbose', 'usage'):
                if field in chunk:
                    summary[field] = chunk[field]
            for choice in chunk.get('choices', []):
                if choice['finish_reason']:
                    summary['finish_reason'] = choice['finish_reason']
                for partial in choice['delta'].get('tool_calls', []):
                    assert partial['index'] == 0, 'expected a single tool call'
                    if 'tool_call' not in first_delta:
                        first_delta['tool_call'] = time.monotonic() - start
                    tool_call['id'] += partial.get('id', '')
                    for field, value in partial.get('function', {}).items():
                        tool_call['function'][field] += value
                for field, parts in deltas.items():
                    delta = choice['delta'].get(field)
                    if delta:
                        if not parts:
                            first_delta[field] = time.monotonic() - start
                        parts.append(delta)
        else:
            raise AssertionError('SSE ended without [DONE]')
    summary.update({field: ''.join(parts) for field, parts in deltas.items()})
    summary['ttft_seconds'] = first_delta.get('content')
    summary['reasoning_ttft_seconds'] = first_delta.get('reasoning_content')
    if 'tool_call' in first_delta:
        summary.update(tool_call=tool_call, tool_ttft_seconds=first_delta['tool_call'])
    summary['progress'] = progress
    return summary


def two_turn_chat():
    native = json.loads((work / 'native.json').read_text()) if a.mode == 'stream' else None
    result['turns'] = []
    capture = work / (a.mode + '.f32')
    begin = 0
    use_sse = a.long_prompt or a.thinking or a.tool_call
    follow_up_prompt = 'Repeat your previous answer exactly.'
    request = dict(chat_body, cache_prompt=True)
    if a.tool_call:
        request.update(messages=[dict(role='user', content=(
            'Call lookup_status with key alpha. After the tool result, reply with only its status value.'))],
            tools=[dict(type='function', function=dict(
                name='lookup_status', description='Look up the status value for the given key.',
                parameters=dict(type='object', properties=dict(key=dict(type='string')),
                                required=['key'], additionalProperties=False)))],
            tool_choice='required', parallel_tool_calls=False, max_tokens=64)
    elif a.thinking:
        request.update(messages=[dict(role='user', content='What is 7 + 5? Give only the number in the final answer.')],
                       chat_template_kwargs=dict(enable_thinking=True, reasoning_effort='low', preserve_thinking=True),
                       reasoning_format='deepseek', max_tokens=128)
        follow_up_prompt = 'Add 3 to your previous answer. Give only the number in the final answer.'
    elif a.long_prompt:
        archive, separator, _ = a.long_prompt.read_text().rpartition('\n\n')
        assert separator and archive, 'expected archive and question paragraphs'
        request['messages'] = [dict(role='user', content=archive + '\n\nReply with only the word OK.')]
        result['fixture_sha256'] = sha(a.long_prompt)
    for index in range(2):
        label = f'turn-{index + 1}'
        emits_tool = a.tool_call and index == 0
        rendered = call('/apply-template', request)['prompt']
        prompt_tokens = call('/tokenize', dict(content=rendered, add_special=True, parse_special=True))['tokens']
        if use_sse:
            response = streamed_chat(request)
        else:
            response = call('/v1/chat/completions', request)
            choice = response['choices'][0]
            response.update(content=choice['message']['content'], finish_reason=choice['finish_reason'])
        raw = response['__verbose']
        save(label)
        timings = raw['timings']
        assert raw['prompt'] == rendered and raw['id_slot'] == 0 and not raw['truncated'], label
        finish = 'tool_calls' if emits_tool else 'stop'
        assert raw['stop_type'] == 'eos' and response['finish_reason'] == finish, (label, response)
        assert raw['tokens_evaluated'] == len(prompt_tokens) == response['usage']['prompt_tokens'], label
        count = raw['tokens_predicted']
        assert 0 < count == response['usage']['completion_tokens'] < request['max_tokens'], label
        ttft = response.get('tool_ttft_seconds' if emits_tool else 'ttft_seconds')
        if use_sse:
            assert ttft is not None and ttft > 0, label
            assert timings['cache_n'] + timings['prompt_n'] == len(prompt_tokens), timings
        else:
            assert len(raw['tokens']) == count and response['content'] == raw['content'] == 'OK', label
        assert not call('/slots')[0]['is_processing'], 'slot stayed busy after EOS'
        end = capture.stat().st_size
        assert end - begin == count * 248320 * 4, 'sampling capture incomplete'
        with capture.open('rb') as rows:
            rows.seek(begin)
            logits_sha = hashlib.sha256(rows.read(end - begin)).hexdigest()
        begin = end
        turn = dict(request=request, prompt=rendered, prompt_tokens=prompt_tokens,
                    output_ids=raw['tokens'], stop_type=raw['stop_type'], logits_sha256=logits_sha,
                    content=response['content'], sampled_count=count)
        report = dict(comparison=turn, metrics=metrics(label), timings=timings)
        if use_sse:
            report.update(ttft_seconds=ttft, progress=response['progress'])
        assistant = dict(role='assistant', content=response['content'])
        next_message = dict(role='user', content=follow_up_prompt)
        if a.tool_call:
            # IDs are generated by the HTTP server. Check correlation on the wire and compare rendered inputs.
            report['request'] = turn.pop('request')
            if emits_tool:
                tool_call = response['tool_call']
                function = tool_call['function']
                assert tool_call['id'] and not response['content'].strip(), response
                assert function['name'] == 'lookup_status' and json.loads(function['arguments']) == {'key': 'alpha'}, function
                turn['tool_function'] = function
                report['tool_call_id'] = tool_call['id']
                assistant['tool_calls'] = [tool_call]
                next_message = dict(role='tool', tool_call_id=tool_call['id'], content='{"status":"ZX-418"}')
            else:
                assert 'tool_call' not in response and response['content'].strip() == 'ZX-418', response
        if a.thinking:
            reasoning = response['reasoning_content']
            first_reasoning = response['reasoning_ttft_seconds']
            assert reasoning.strip() and 0 < first_reasoning <= response['ttft_seconds'], label
            assert response['content'].strip() == ('12', '15')[index], label
            turn['reasoning_content'] = reasoning
            report['reasoning_ttft_seconds'] = first_reasoning
            assistant['reasoning_content'] = reasoning
        if a.long_prompt:
            assert 3000 <= len(prompt_tokens) < 4096, label
        result['turns'].append(report)
        if native:
            assert turn == native['turns'][index]['comparison'], label
            assert equal_file(slots / (label + '.slot'), work / 'native' / (label + '.slot')), label
        print(a.mode, label, len(prompt_tokens), timings['cache_n'], repr(response['content']),
              ttft, flush=True)
        request = dict(request, messages=request['messages'] + [assistant, next_message])
        if emits_tool:
            request['tool_choice'] = 'auto'
    if native:
        assert equal_file(capture, work / 'native.f32'), 'full pre-sampler logits differ'
        result['ordinary_logits_and_states_bitwise_equal'] = True
    if a.long_prompt:
        first, follow_up = (turn['timings'] for turn in result['turns'])
        assert first['cache_n'] == 0 and follow_up['cache_n'] >= first['prompt_n'] - 8, (first, follow_up)


def single_turn_chat():
    props = call('/props')
    template = props.get('chat_template')
    assert isinstance(template, str) and template, 'no model chat template'
    result['template_sha256'] = hashlib.sha256(template.encode()).hexdigest()
    result['rendered_prompt'] = call('/apply-template', chat_body)['prompt']
    assert isinstance(result['rendered_prompt'], str)
    result['prompt_tokens'] = call('/tokenize', dict(content=result['rendered_prompt'],
                                                    add_special=True, parse_special=True))['tokens']
    raw = call('/completion', dict(prompt=result['rendered_prompt'], n_predict=16,
               temperature=0, seed=42, cache_prompt=False, id_slot=0, return_tokens=True))
    result['raw'] = raw
    check_eos(raw)
    save('raw')
    result['raw_metrics'] = metrics('raw')
    print(a.mode, 'raw EOS', raw['tokens'], flush=True)
    chat = call('/v1/chat/completions', chat_body)
    result['chat'] = chat
    assert chat['choices'][0]['finish_reason'] == 'stop'
    assert chat['choices'][0]['message']['content'] == 'OK'
    check_eos(chat['__verbose'])
    assert chat['__verbose']['tokens'] == raw['tokens']
    assert chat['usage']['prompt_tokens'] == len(result['prompt_tokens'])
    assert chat['usage']['completion_tokens'] == len(raw['tokens'])
    save('chat')
    result['chat_metrics'] = metrics('chat')
    assert not call('/slots')[0]['is_processing'], 'slot stayed busy after EOS'
    print(a.mode, 'chat EOS', chat['__verbose']['tokens'], flush=True)
    if a.mode == 'stream':
        native = json.loads((work / 'native.json').read_text())
        for key in ['template_sha256', 'rendered_prompt', 'prompt_tokens']:
            assert result[key] == native[key], key
        for key in ['raw', 'chat']:
            response = result[key] if key == 'raw' else result[key]['__verbose']
            reference = native[key] if key == 'raw' else native[key]['__verbose']
            for field in ['content', 'tokens', 'stop_type', 'tokens_predicted', 'tokens_evaluated', 'prompt']:
                assert response[field] == reference[field], (key, field)
            assert equal_file(slots / (key + '.slot'), work / 'native' / (key + '.slot')), key + ' state differs'
        result['ordinary_template_tokens_responses_and_states_equal'] = True

try:
    with server():
        if a.follow_up:
            two_turn_chat()
        else:
            single_turn_chat()
        result['passed'] = True
finally:
    (work / (a.mode + '.json')).write_text(json.dumps(result, indent=2) + '\n')
