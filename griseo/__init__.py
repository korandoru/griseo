# Copyright 2023 Korandoru Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import textwrap
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, REMAINDER
from importlib import metadata

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)  # for exponential backoff

__version__ = metadata.version(__package__)

del metadata  # avoids polluting the results of dir(__package__)


def spin(response, print_role) -> (str, str):
    role, content = '', ''
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if delta.get('role') and print_role:
            role = delta['role']
            print(f"{delta['role']} >> ", end='', flush=True)
        if delta.get('content'):
            content += delta['content']
            print(delta['content'], end='', flush=True)
    print()
    return role, content


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(openai.error.RateLimitError),
    reraise=True)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


class Context:

    def __init__(self, file):
        self._messages = []
        self._prompts = []
        self.reset(file)

    def tell(self, msg, print_role):
        self._messages.append({"role": "user", "content": msg})
        response = completions_with_backoff(
            model="gpt-3.5-turbo",
            messages=self._messages,
            stream=True)
        role, content = spin(response, print_role)
        self._messages.append({"role": role, "content": content})

    def reset(self, file=None):
        if file is not None:
            from griseo import prompts
            self._prompts = prompts.load(file)
        self._messages = self._prompts


def main():
    import dotenv
    import os

    dotenv.load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise Exception(
            "No API key provided. You can configure your API key by `export OPENAI_API_KEY=<API-KEY>`"
            " or `echo OPENAI_API_KEY=<API-KEY> >> .env`.")
    openai.api_key = api_key

    griseo = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    griseo.add_argument('words', nargs=REMAINDER)
    griseo.add_argument('-v', '--version', action='version', version=__version__, help='show version')
    griseo.add_argument('-p', '--prompt', metavar='FILE', default='default.csv', help='prompts file')

    args = griseo.parse_args()
    ctx = Context(args.prompt)

    # oneshot
    if len(args.words) > 0:
        ctx.tell(' '.join(args.words), print_role=False)
        return

    # interactive
    commands = {
            'r': lambda: ctx.reset(),
            'reset': lambda: ctx.reset(),
            'q': lambda: sys.exit(0),
            'quit': lambda: sys.exit(0),
            'h': lambda: print(usage),
            'help': lambda: print(usage),
    }

    usage = textwrap.dedent("""
    :c, :clear            reset chat context
    :h, :help             show this help message
    :q, :quit             exit the conversation
    """)

    print(f"Welcome to chat with Griseo!\n{usage}")

    logging.basicConfig(stream=sys.stderr, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger()

    while True:
        try:
            req = input('user << ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if req.startswith(':'):
            if req[1:] in commands:
                commands[req[1:]]()
            else:
                logger.warning(f"unknown command {req}\n{usage}")
            continue

        if len(req) > 0:  # skip empty input
            try:
                ctx.tell(req, print_role=True)
            except openai.error.RateLimitError as e:
                print(e.user_message)
            except openai.error.InvalidRequestError as e:
                print(e.user_message)
