import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request


MAX_STEPS = 10000
BASE_PATH = Path(__file__).parent
PROMPT = (BASE_PATH / 'whenwords.txt').read_text()
SYSTEM_PROMPT = 'You are an expert Python programmer that uses the POSIX shell to write Python.'
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'sh',
            'description': 'Runs an arbitrary POSIX shell command. Returns its stdout, stderr and exit code.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'cmd': {'type': 'string'},
                },
                'required': ['cmd'],
                'additionalProperties': False,
            }
        }
    }
]


def query(base_url, payload, token):
    headers = {
        'Content-Type': 'application/json',
    }
    if token is not None:
        headers['Authorization'] = f'Bearer {token}'

    req = urllib.request.Request(
        f'{base_url}/v1/chat/completions',
        data=json.dumps(payload).encode(),
        headers=headers,
        method='POST',
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode('utf-8')
            return json.loads(body)
    except urllib.error.HTTPError as e:
        err_msg = e.fp.read().decode()
        try:
            err_msg = json.loads(err_msg)['error']['message']
        except json.decoder.JSONDecodeError:
            pass
        print(err_msg, file=sys.stderr)
        raise e


def get_tool_result_message(tool):
    assert all(key in tool for key in ('type', 'id', 'function')), tool
    assert tool['type'] == 'function', tool
    tool_id = tool['id']
    tool_func = tool['function']

    assert tool_func.get('name') == 'sh', tool_func
    assert 'arguments' in tool_func, tool_func
    tool_args = json.loads(tool_func['arguments'])
    assert 'cmd' in tool_args, tool_args
    sh_cmd = tool_args['cmd']
    assert isinstance(sh_cmd, str)

    sh_res = subprocess.run(
        sh_cmd,
        shell=True,
        capture_output=True,
        encoding='utf-8',
        env={'LD_PRELOAD': '/usr/lib/faketime/libfaketime.so.1'},
    )

    print(f'Launched command\n{sh_res.args}')
    print(f'Exit code: {sh_res.returncode}')
    print(f'Stdout\n{sh_res.stdout}')
    print(f'Stderr\n{sh_res.stderr}')

    return {
        'role': 'tool',
        'tool_call_id': tool_id,
        'content': json.dumps({
            'exitcode': sh_res.returncode,
            'stdout': sh_res.stdout,
            'stderr': sh_res.stderr,
        }),
    }


def messages_to_payload(messages, args):
    return {
        'messages': messages,
        'model': args.model,
        'tool_choice': 'auto',
        'tools': TOOLS,
        'temperature': 0.0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_url')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--trajectory-dir', type=str, default=None)
    parser.add_argument('--replay', type=str, default=None)
    parser.add_argument('--model', type=str, default='Qwen/Qwen3.6-35B-A3B')
    parser.add_argument('--openrouter-token', type=str, default=None)
    args = parser.parse_args()

    if args.replay is not None:
        replayed_messages = json.loads((Path(args.trajectory_dir) / args.replay).read_text())
        payload = messages_to_payload(replayed_messages, args)
        response = query(args.base_url, payload, args.openrouter_token)
        response_msg = response['choices'][0]['message']

        GREEN = "\033[0;32m"
        END = "\033[0m"

        print(f'{GREEN}Content{END}\n{response_msg["content"]}')
        print(f'{GREEN}Reasoning{END}\n{response_msg["reasoning"]}')
        for ii, tc in enumerate(response_msg["tool_calls"]):
            cmd = json.loads(tc["function"]["arguments"])["cmd"]

            print(f'{GREEN}Tool output {ii}{END}')
            print(cmd)

        exit(0)

    messages = [
        {
            'role': 'system',
            'content': SYSTEM_PROMPT,
        },
        {
            'role': 'user',
            'content': PROMPT,
        },
    ]

    for step in range(MAX_STEPS):
        payload = messages_to_payload(messages, args)

        payload_hash = hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()
        print(f'Sending request with hash {payload_hash}')

        if args.verbose:
            print(json.dumps(payload, indent=2))

        response = query(args.base_url, payload, args.openrouter_token)

        if args.verbose:
            print(f'Got response\n{json.dumps(response, indent=2)}')

        assert len(response.get('choices', [])) >= 1, response
        first_choice = response['choices'][0]
        assert 'message' in first_choice, first_choice
        model_message = first_choice['message']
        messages.append(model_message)

        if not model_message.get('tool_calls', []):
            print('ALL DONE')
            break

        for tool in model_message['tool_calls']:
            messages.append(get_tool_result_message(tool))

        if args.trajectory_dir is not None:
            (Path(args.trajectory_dir) / f'{step:05d}.json').write_text(
                json.dumps(messages, indent=2)
            )
