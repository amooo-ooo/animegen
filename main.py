from abc import ABC
from typing import Iterable

import discord
from discord import app_commands
from discord.ext import commands
from gradio_client import Client, handle_file, exceptions as gradio_exc

from dotenv import load_dotenv
from pathlib import Path
import asyncio
import functools
import random
import os
import toml
import re


class DFAWordBlacklist:
    DEFAULT = 'default'

    def __init__(self) -> None:
        self._directory = {}

    def _get_or_create_path(self, path: str, value):
        current_dir = self._directory
        path_to_go = path
        while len(path_to_go) > 1:
            if path_to_go[0] not in current_dir:
                current_dir[path_to_go[0]] = {}
            elif not isinstance(current_dir[path_to_go[0]], dict):
                return current_dir[path_to_go[0]]
            current_dir = current_dir[path_to_go[0]]
            path_to_go = path_to_go[1:]
        current_dir[path_to_go] = value
        return current_dir

    def _get_path(self, path: str):
        current_dir = self._directory
        while path:
            if path[0] not in current_dir:
                return None
            current_dir = current_dir[path[0]]
            if not isinstance(current_dir, dict):
                return current_dir
            path = path[1:]
        return current_dir

    def add_word(self, word: str):
        self._get_or_create_path(word, True)

    def add_all(self, items: Iterable[str]):
        for item in items:
            self.add_word(item)

    def compile(self):
        todo: list[tuple[str, dict[str, dict | bool]]] = [
            ('', self._directory)]
        while todo:
            dir_name, val = todo.pop()
            for k, v in val.items():
                if isinstance(v, dict):
                    todo.append((dir_name + k, v))
            offset = 0
            if not dir_name:
                continue
            while dir_name := dir_name[1:]:
                offset += 1
                path = self._get_path(dir_name)
                if isinstance(path, dict):
                    val[DFAWordBlacklist.DEFAULT] = (path, offset)

    def exec(self, value: str) -> list[tuple[int, int]]:
        result = []
        current_dir = self._directory
        start_index = 0
        i = -1
        while (i := (i + 1)) < len(value):
            char = value[i]
            if char in current_dir:
                if isinstance(current_dir[char], dict):
                    current_dir = current_dir[char]
                else:
                    result.append((start_index, i + 1))
                    start_index = i + 1
                    current_dir = self._directory
            elif DFAWordBlacklist.DEFAULT in current_dir:
                current_dir, offset = current_dir[DFAWordBlacklist.DEFAULT]
                start_index += offset
                i -= 1
            elif current_dir is self._directory:
                start_index = i + 1
            else:
                current_dir = self._directory
                start_index = i
                i -= 1
        return result

    def replace(self, value: str, replace: str):
        result = ''
        matches = self.exec(value)
        i = 0
        while matches:
            start, end = matches.pop(0)
            result += value[i:start]
            result += replace if len(replace) != 1 else (replace *
                                                         (end - start))
            i = end
        result += value[i:]
        return result



class Animegen(commands.Bot, ABC):
    def __init__(self, config_path="config.toml",
                 *args, **options):
        super().__init__(*args, **options)
        self.img_client = Client("Boboiazumi/animagine-xl-3.1")
        self.chat_client = Client("Be-Bo/llama-3-chatbot_70b")
        self.config = toml.load(config_path)

        self.general = self.config["general"]
        self.params = self.config['command_params']
        self.defaults = self.config['defaults']

        self.chat_participants = []
        self.chat_history = []

        path = Path(Path(__file__).parent, "prompts")
        with open(Path(path, "system.txt"), "r") as f:
            self.system = f.read().strip() + "\n"

        with open(Path(path, "reminder.txt"), "r") as f:
            self.reminder = f.read().strip() + "\n"

    async def generate(
            self,
            prompt: str | None = "frieren",
            negative_prompt: str | None = "",
            kwargs: dict = {}):

        image, details = await asyncio.threads.to_thread(functools.partial(
            self.img_client.predict,
            prompt=prompt + ", " + self.params['additional_prompt'],
            negative_prompt=negative_prompt + ", " + self.params["additional_negative_prompt"],
            seed=kwargs.get(
                "seed", random.randint(0, 2147483647)),
            guidance_scale=7,
            num_inference_steps=kwargs.get(
                "steps", self.defaults["steps"]),
            sampler=kwargs.get(
                "sampler", self.defaults["sampler"]),
            aspect_ratio_selector=kwargs.get(
                "aspect_ratio", self.defaults["aspect_ratio"]),
            style_selector=kwargs.get(
                "style_selector", self.defaults["style_selector"]),
            quality_selector=kwargs.get(
                "quality_selector", self.defaults["quality_selector"]),
            img_path=handle_file(
                'https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
            img2img_strength=0.65,
            api_name="/run"
        ))

        path = image[0]["image"]
        return path

    async def chat(self, message):
        if not self.chat_history:
            message = (self.system + message)
        elif not (len(self.chat_history) % self.general["context_window"]):
            message = (self.reminder + message)

        result = await asyncio.threads.to_thread(functools.partial(
            self.chat_client.predict,
            message=message,
            api_name="/chat"))

        if len(self.chat_history) >= self.general["context_window"]:
            self.chat_history = self.chat_history[1:] + [result]
        else:
            self.chat_history.append(result)

        return result

    async def send_with_image(self, channel, message, prompt):
        try:
            path = await self.generate(prompt)
            with open(path, 'rb') as f:
                await channel.send(message, file=discord.File(f))
            os.remove(path)
        except gradio_exc.AppError as e:
            pass
            # TODO: handle image generation exception
            # await channel.send(
            #     message + f' [gradio: image gen failed: {str(e)}]')

    async def on_message(self, message):
        if message.author.id in self.chat_participants:
            # Defer the response
            async with message.channel.typing():
                # Generate the response
                response = await self.chat(f"{message.author.display_name}: {message.content}")

            if match := re.search(r"\[image: ([^\]]*)\]", response, re.IGNORECASE):
                await self.send_with_image(
                    message.channel, re.
                    sub(r"\[image: [^\]]*\]", '',
                        response, 0, re.IGNORECASE),
                    match.group(1))
            else:
                await message.channel.send(response)


class Bot:
    def __init__(self, token, **options):
        super().__init__(**options)
        instance = Animegen(command_prefix="!",
                            intents=discord.Intents.all())

        @instance.event
        async def on_ready():
            print(f'{instance.user} is cooking!!')
            try:
                synced = await instance.tree.sync()
                print(f"Synced {len(synced)} command(s)")
            except Exception as e:
                print(e)

        blacklist = DFAWordBlacklist()
        if 'words' in instance.config['blacklist']:
            blacklist.add_all(instance.config['blacklist']['words'])
        if 'files' in instance.config['blacklist']:
            for file in instance.config['blacklist']['files']:
                try:
                    with open(file, 'rt', encoding='utf8') as f:
                        blacklist.add_all(
                            filter(lambda x: bool(x.strip()),
                                   map(lambda x: x.removesuffix('\n'),
                                       f.readlines())))
                except IOError:
                    pass  # Ignore failing files
        blacklist.compile()
        verbose = instance.general['verbose']

        PROMPT = instance.params["additional_prompt"].replace(", ", "`, `") + "`."
        NEGATIVE_PROMPT = instance.params["additional_negative_prompt"].replace(", ", "`, `") + "`."

        def split(prompt: str,
                  negative_prompt: str):

            if prompt:
                prompt = ("`" + prompt.replace(", ",
                          ",").replace(",", "`, `") + "`, `")
            else:
                prompt = (
                    "`" + instance.defaults["prompt"].replace(",", "`, `") + "`, ")

            if negative_prompt:
                negative_prompt = (
                    "`" + negative_prompt.replace(", ", ",").replace(",", "`, `") + "`, `")
            elif instance.defaults["negative_prompt"]:
                negative_prompt = (
                    "`" + instance.defaults["negative_prompt"].replace(",", "`, `") + "`, `")

            return (prompt + PROMPT, negative_prompt + NEGATIVE_PROMPT)

        def cleanse(prompt):
            t = prompt.lower()
            t = blacklist.replace(t, '*')
            return t

        def embed(user, prompt: str, negative_prompt: str, kwargs):

            embedVar = discord.Embed(title=f"Generating for `@{user}`!",
                                     description="Powered by `Boboiazumi/animagine-xl-3.1` on Huggingface Spaces with Gradio.",
                                     color=0xf4e1cc)

            prompt, negative_prompt = split(prompt, negative_prompt)

            embedVar.add_field(name="Prompt", value=prompt, inline=False)
            embedVar.add_field(name="Negative Prompt",
                               value=negative_prompt, inline=False)
        
            for title, value in kwargs.items():
                embedVar.add_field(name=title.title(),
                                   value=value, inline=True)

            return embedVar

        @instance.tree.command(name="chat", description="Join, create or leave a conversation with Animegen")
        async def chat(interaction: discord.Interaction):
            if not (interaction.user.id in instance.chat_participants):
                instance.chat_participants.append(interaction.user.id)
                await interaction.response.send_message(f"`@{interaction.user.display_name}` has joined the conversation!")
            else:
                instance.chat_participants.remove(interaction.user.id)
                await interaction.response.send_message(f"`@{interaction.user.display_name}` has left the conversation!")

            if instance.chat_participants == []:  # refresh
                instance.client = Client("Be-Bo/llama-3-chatbot_70b")
                instance.chat_history = []

        @instance.tree.command(name="imagine", description="Generate an image")
        @app_commands.describe(
            prompt="Tags for generating the image",
            negative_prompt="Penalty tags for generating the image",
            sampler=str(instance.params['samplers']),
            seed="Seed for generating the image",
            steps="Inference steps for generating the image",
            width="Custom width",
            height='Custom height',
            quality_selector=str(instance.params['quality_selectors']),
            style_selector=str(instance.params['style_selectors'])
        )
        async def imagine(
                interaction: discord.Interaction,
                prompt: str = instance.defaults['prompt'],
                negative_prompt: str = instance.defaults['negative_prompt'],
                sampler: str = instance.defaults['sampler'],
                steps: int = instance.defaults['steps'],
                width: int = int(instance.defaults['aspect_ratio'].split(" x ")[0]),
                height: int = int(instance.defaults['aspect_ratio'].split(" x ")[1]),
                quality_selector: str = instance.defaults['quality_selector'],
                style_selector: str = instance.defaults['style_selector'],
                seed: int = -1):

            prompt = cleanse(prompt)

            if not (sampler in instance.params['samplers']):
                sampler = instance.defaults['sampler']

            if not (quality_selector in instance.params['quality_selectors']):
                quality_selector = instance.defaults['quality_selector']

            if not (style_selector in instance.params['style_selectors']):
                style_selector = instance.defaults['style_selector']

            if 0 > seed or seed > 2147483647:
                seed = random.randint(0, 2147483647)

            if verbose:
                log = f"{interaction.user.global_name} - {prompt}"
                print(log)

                with open("log.txt", "a") as f:
                    f.write(log + "\n")

            await interaction.response.defer()

            kwargs = {'seed': seed,
                      'sampler': sampler,
                      'aspect_ratio': f"{width} x {height}",
                      'steps': steps,
                      'style_selector': style_selector,
                      'quality_selector': quality_selector
                      }

            try:
                embed_log = embed(interaction.user.display_name,
                                  prompt,
                                  negative_prompt,
                                  kwargs)
                
                path = await instance.generate(prompt, negative_prompt,  kwargs)

                with open(path, 'rb') as f:
                    await interaction.followup.send(embed=embed_log, file=discord.File(f))

                os.remove(path)
            except Exception as e:
                time = ':'.join(str(e).split(':')[2:])[:5]
                await interaction.followup.send(f"> Quota was met for generating images. Go touch some grass for {time} minute(s) and come back!")
                print(e)

        instance.run(token)


if __name__ == "__main__":
    load_dotenv()
    bot = Bot(token=os.getenv('TOKEN'))
