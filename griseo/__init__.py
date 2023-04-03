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

import code
import sys
import textwrap

import click
import openai


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
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if delta.get('content'):
            print(delta['content'], end='', flush=True)
    print()


@click.command()
def chat():
    class Context:
        def __init__(self):
            self._messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

        def tell(self, msg):
            self._messages.append({"role": "user", "content": msg})
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self._messages)
            resp = response["choices"][0]["message"]
            self._messages.append({"role": resp["role"], "content": resp["content"]})
            print(resp["content"])

        def clear(self):
            self._messages = []

    ctx = Context()
    banner = textwrap.dedent(f"""\
    Python {sys.version} on {sys.platform}
    'Type "help", "copyright", "credits" or "license" for more information.'
    
    Welcome to chat with Griseo! You can use:
    * ctx.tell("...") to interact
    * ctx.clear()     to reset the conversation.
    """)
    code.interact(banner=banner, local=locals())


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
