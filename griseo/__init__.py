# Copyright 2023 tison <wander4096@gmail.com>
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

import sys

import click
import openai


def spin(response) -> (str, str):
    role, content = '', ''
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if delta.get('role'):
            role = delta['role']
            print(f"{delta['role']} >> ", end='', flush=True)
        if delta.get('content'):
            content += delta['content']
            print(delta['content'], end='', flush=True)
    print()
    return role, content


@click.group()
def griseo():
    pass


@click.command()
@click.argument('words', nargs=-1)
def tell(words):
    content = ' '.join(words)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": content}],
        stream=True)
    spin(response)


@click.command()
def chat():
    class Context:
        def __init__(self):
            self._messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

        def tell(self, msg):
            self._messages.append({"role": "user", "content": msg})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self._messages,
                stream=True)
            role, content = spin(response)
            self._messages.append({"role": role, "content": content})

        def clear(self):
            self._messages = []

    ctx = Context()

    commands = {
        'c': lambda: ctx.clear(),
        'clear': lambda: ctx.clear(),
        'q': lambda: sys.exit(0),
        'quit': lambda: sys.exit(0),
    }

    while True:
        req = input('user << ').strip()
        if req.startswith(':'):
            commands[req[1:]]()
        ctx.tell(req)


griseo.add_command(tell)
griseo.add_command(chat)


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
    griseo()
