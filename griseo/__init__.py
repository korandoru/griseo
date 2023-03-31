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
        messages=[{"role": "user", "content": content}])
    resp = response["choices"][0]["message"]
    print(resp["content"])


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
    code.interact(local=locals())


griseo.add_command(tell)
griseo.add_command(chat)


def main():
    import dotenv
    import os

    dotenv.load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    griseo()
