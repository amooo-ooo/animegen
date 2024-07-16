from abc import ABC
import contextlib
import datetime
from typing import Iterable

import discord
from discord import app_commands
from discord.ext import commands
from gradio_client import Client, handle_file, exceptions as gradio_exc
from datetime import timedelta

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
            char = value[i].lower()
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
        single_chr = len(replace) == 1 or (
            len(replace) == 2 and replace.startswith('\\'))
        result = ''
        matches = self.exec(value)
        i = 0
        while matches:
            start, end = matches.pop(0)
            result += value[i:start]
            result += (replace * (end - start)) if single_chr else replace
            i = end
        result += value[i:]
        return result


class AnimegenMemory:
    def __init__(self, prompts_path: Path, memories_path: Path):
        self.client = Client("Be-Bo/llama-3-chatbot_70b")
        self.memories_path = memories_path
        if not self.memories_path.exists():
            os.makedirs(self.memories_path)
        self.query_prompt = (prompts_path.joinpath("memory_query.txt")
                             .read_text().strip() + '\n')
        self.user_select_prompt = (prompts_path
                                   .joinpath("memory_select_user.txt").read_text().strip() + '\n')
        self.write_prompt = (prompts_path.joinpath("memory_write.txt")
                             .read_text().strip() + '\n')

    async def _message_client(self, message: str) -> str:
        return await asyncio.threads.to_thread(functools.partial(
            self.client.predict,
            message=message,
            api_name="/chat"
        ))

    async def query(self, query: str):
        query_prompt = self.query_prompt.format_map(
            {"files": ', '.join(map(lambda x: x.stem, self.memories_path.iterdir()))})
        response = await self._message_client(
            f'{query_prompt}query: {query}')
        while match := re.search(r"\[\s*read\s*:\s*([^\]]*)\s*\]", response, re.I):
            file_name = match.group(1).lower().strip()
            file = self.memories_path.joinpath(file_name + '.txt')
            if file.exists():
                file_content = f'--- FILE {file_name} CONTENTS ---\n'
                file_content += file.read_text() + '\n'
                file_content += f'--- END FILE {file_name} CONTENTS ---\n'
                response = await self._message_client(
                    f'{file_content}{query_prompt}query: {query}')
            else:
                response = await self._message_client(
                    f'--- FILE {file_name} DOES NOT EXIST ---\n'
                    f'{query_prompt}query: {query}')
        return response

    async def save(self, info: str):
        user_prompt = self.user_select_prompt.format_map(
            {"files": ', '.join(map(lambda x: x.stem, self.memories_path.iterdir()))})
        response = await self._message_client(
            f'{user_prompt}info: {info}')
        for file in response.split(','):
            file = file.strip().lower()
            file_contents = ''
            if match := re.fullmatch(r"\[\s*new\s*:\s*(\w+)\s*\]", file):
                file = match.group(1)
                path = self.memories_path.joinpath(file + '.txt')
                file_contents = f'--- CREATED FILE FOR USER {file} ---\n'
            else:
                path = self.memories_path.joinpath(file + '.txt')
                if not path.exists():
                    continue
                file_contents = (
                    f'--- FILE CONTENTS FOR {file} ---\n'
                    + path.read_text() + '\n'
                    + f'--- END FILE CONTENTS FOR {file} ---\n')
            write_prompt = self.write_prompt.format_map({"user": file})
            response = await self._message_client(
                f'{file_contents}{write_prompt}info: {info}')
            with path.open('wt') as f:
                f.write(response)


class Animegen(commands.Bot, ABC):
    IMAGE_EMBED_REGEX = r"\[\s*[image]{3,5}\s*:\s*([^\]]*)\s*\]"
    QUERY_REGEX = r'\[\s*query\s*\]'
    SAVE_MEM_REGEX = r"\[\s*save\s*:\s*(.*)\s*:\s*save\s*\]"
    CONFIG_READ_CHANNEL_HISTORY = 'read_channel_history'

    def __init__(self, config_path="config.toml",
                 *args, **options):
        super().__init__(*args, **options)
        self.img_client = Client("Boboiazumi/animagine-xl-3.1")
        self.chat_client: Client | None = Client("Be-Bo/llama-3-chatbot_70b")
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            cfg_path = Path('config.default.toml')
        self.config = toml.load(cfg_path)

        self._background_tasks = []

        self.blacklist = DFAWordBlacklist()
        if 'words' in self.config['blacklist']:
            self.blacklist.add_all(self.config['blacklist']['words'])
        if 'files' in self.config['blacklist']:
            for file in self.config['blacklist']['files']:
                try:
                    with open(file, 'rt', encoding='utf8') as f:
                        self.blacklist.add_all(
                            filter(lambda x: bool(x.strip()),
                                   map(lambda x: x.removesuffix('\n'),
                                       f.readlines())))
                except IOError:
                    pass  # Ignore failing files
        self.blacklist.compile()

        self.general = self.config["general"]
        self.params = self.config['command_params']
        self.defaults = self.config['defaults']

        self.chat_participants = []
        self.chat_context = []
        self.chat_history = []
        self.memory_path = Path(Path(__file__).parent, "memory")
        self.user_memories = set(os.listdir(self.memory_path))

        prompts_path = Path(Path(__file__).parent,
                            self.general["prompts"] if 'prompts' in self.general else 'prompts')
        self.memory_handler = AnimegenMemory(prompts_path, self.memory_path)
        with open(Path(prompts_path, "system.txt"), "r") as f:
            self.system = f.read().strip() + "\n"

        with open(Path(prompts_path, "reminder.txt"), "r") as f:
            self.reminder = f.read().strip() + "\n"
            
        with open(Path(prompts_path, "memory.txt"), "r") as f:
            self.memory = f.read().strip() + "\n"

        with open(Path(prompts_path, "memory.txt"), "r") as f:
            self.memory = f.read().strip() + "\n"

    def add_task(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        if (len(self._background_tasks) > 5):
            self._background_tasks[random.randint(1, 3)].cancel()
        task.add_done_callback(self._background_tasks.remove)

    async def generate(
            self,
            prompt: str | None = "frieren",
            negative_prompt: str | None = "",
            kwargs: dict = {}):

        image, details = await asyncio.threads.to_thread(functools.partial(
            self.img_client.predict,
            prompt=prompt + ", " + self.params['additional_prompt'],
            negative_prompt=negative_prompt + ", " +
            self.params["additional_negative_prompt"],
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
    
    # def load_memory(self, username):
    #     path = Path(self.memory_path, str(username) + ".txt")
    #     if os.path.isfile(path):
    #         with open(path, "r") as f:
    #             return (f.read().strip())
    #     return ""

    # async def save_memory(self, username):
    #     path = Path(self.memory_path, str(username) + ".txt")
    #     context = await asyncio.threads.to_thread(functools.partial(
    #         self.chat_client.predict,
    #         message=self.memory.replace("{username}", username),
    #         api_name="/chat"))

    #     with open(path, "a") as f:
    #         f.write(context + "\n")

    async def message_as_str(self, msg: discord.Message):
        replies = ''
        if msg.reference is not None:
            reply_msg = await self.message_as_str(
                await msg.channel.fetch_message(msg.reference.message_id))
            replies = f'[in response to: `{reply_msg}`]'
        return f"{replies}[{msg.created_at}] {msg.author.display_name}: {msg.clean_content}"

    async def chat(self, channel: discord.abc.Messageable, message: str):
        if not self.chat_history:
            history = ''
            if (self.CONFIG_READ_CHANNEL_HISTORY in self.general
                    and int(self.general[self.CONFIG_READ_CHANNEL_HISTORY])):
                length = int(self.general[self.CONFIG_READ_CHANNEL_HISTORY])
                async for old_msg in channel.history(limit=length):
                    history = f'{await self.message_as_str(old_msg)}\n{history}'
                history = (f'\n--- CHANNEL HISTORY ---\n'
                           f'{history}--- END CHANNEL HISTORY ---\n')
                print(history)
            message = (self.system + history + message)
        elif not (len(self.chat_history) % self.general["context_window"]):
            message = (self.reminder + message)

        try:
            result = await asyncio.threads.to_thread(functools.partial(
                self.chat_client.predict,
                message=message,
                api_name="/chat"))
        except Exception as e:
            if 'debug' in self.general and self.general['debug']:
                return f"```{e}```"

        if len(self.chat_history) >= self.general["context_window"]:
            self.chat_history = self.chat_history[1:] + [result]
        else:
            self.chat_history.append(result)

        return result

    @contextlib.asynccontextmanager
    async def gen_image(self, prompt):
        try:
            path = await self.generate(prompt)
            try:
                with open(path, 'rb') as f:
                    yield discord.File(f)
            finally:
                os.remove(path)
        except gradio_exc.AppError as e:
            yield e

    async def handle_message(self, message: discord.Message):
        # Defer the response
        async with message.channel.typing():
            message_str = await self.message_as_str(message)
            # Generate the response
            response = await self.chat(message.channel, message_str)
            def save_mem(match: re.Match[str]) -> str:
                self.memory_handler.save(match.group(1))
                return ''
            response = re.sub(self.SAVE_MEM_REGEX, save_mem, response, 0, re.IGNORECASE)
            while re.search(self.QUERY_REGEX, response, re.I):
                query_res = self.memory_handler.query(
                    re.sub(self.QUERY_REGEX, '', response, 0, re.I))
                response = re.sub(self.SAVE_MEM_REGEX, save_mem,
                    await self.chat(
                        message.channel,
                        f'--- QUERY ---\n'
                        f'query: {response}\n'
                        f'response: {query_res}\n'
                        f'--- END QUERY ---\n'
                        f'ORIGINAL MESSAGE: {message_str}'),
                    0, re.IGNORECASE)

            response = self.blacklist.replace(response, '\\*')
            if 'debug' in self.general and self.general['debug']:
                print(response)

            if match := re.search(self.IMAGE_EMBED_REGEX, response, re.IGNORECASE):
                async with self.gen_image(match.group(1)) as img:
                    text_msg = re.sub(self.IMAGE_EMBED_REGEX, '',
                                      response, 0, re.IGNORECASE)
                    if isinstance(img, Exception):
                        response = self.blacklist.replace(
                            await self.chat(
                                message.channel,
                                f'dev: [image generation failed with error: '
                                f'"{img}". ur original msg was "{text_msg}". '
                                f'please send a followup message (please note '
                                f'images will not work in your followup)]'),
                            '\\*')
                        await message.channel.send(response)
                    else:
                        await message.channel.send(text_msg, file=img)
            else:
                await message.channel.send(response)

    async def on_message(self, message: discord.Message):
        if message.author.id in self.chat_participants and self.chat_client is not None:
            self.add_task(self.handle_message(message))

    async def on_user_leave(self, channel: discord.abc.Messageable, user: str):
        await self.memory_handler.save(
            await self.chat(
                channel,
                f'dev: [the user {user} is leaving the chat! this is your last'
                f' chance to save any information about them! respond with '
                f'*any* important memories to remember about them! your '
                f'response here is sent directly to memory, as if using [save]]')
        )


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

        verbose = instance.general['verbose']
        watchlist = {}

        PROMPT = instance.params["additional_prompt"].replace(
            ", ", "`, `") + "`."
        NEGATIVE_PROMPT = instance.params["additional_negative_prompt"].replace(
            ", ", "`, `") + "`."

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
            elif not (instance.defaults["negative_prompt"] == ''):
                negative_prompt = (
                    "`" + instance.defaults["negative_prompt"].replace(",", "`, `") + "`, `")

            return (prompt + PROMPT, negative_prompt + NEGATIVE_PROMPT)

        def embed(user, prompt: str, negative_prompt: str, kwargs):

            embedVar = discord.Embed(title=f"Generating for `@{user}`!",
                                     description="Powered by `Boboiazumi/animagine-xl-3.1` on Huggingface Spaces with Gradio.",
                                     color=0xf4e1cc)
            
            embedVar.set_author(name=instance.user.display_name, icon_url=instance.user.display_avatar)

            embedVar.set_author(name=instance.user.display_name,
                                icon_url=instance.user.display_avatar)

            prompt, negative_prompt = split(prompt, negative_prompt)

            embedVar.add_field(name="Prompt", value=prompt, inline=False)
            embedVar.add_field(name="Negative Prompt",
                               value=negative_prompt, inline=False)

            for title, value in kwargs.items():
                embedVar.add_field(name=title.title(),
                                   value=value, inline=True)

            return embedVar
        
        def embed_logs(title: str, 
                       description: str,
                       thumbnail: str | None = None):
            
            embedVar = discord.Embed(title=title,
                                     description=description,
                                     color=0xf4e1cc)
            
            if thumbnail:
                embedVar.set_thumbnail(url=thumbnail)
                
            embedVar.set_author(name=instance.user.display_name, icon_url=instance.user.display_avatar)
            return embedVar

        def embed_logs(title: str,
                       description: str,
                       thumbnail: str | None = None):

            embedVar = discord.Embed(title=title,
                                     description=description,
                                     color=0xf4e1cc)

            if thumbnail:
                embedVar.set_thumbnail(url=thumbnail)

            embedVar.set_author(name=instance.user.display_name,
                                icon_url=instance.user.display_avatar)
            return embedVar

        @instance.tree.command(name="chat", description="Join, create or leave a conversation with Animegen")
        async def chat(interaction: discord.Interaction):
            if not (interaction.user.id in instance.chat_participants):
                instance.chat_participants.append(interaction.user.id)
                instance.chat_context.append(interaction.user.id)
                await interaction.response.send_message(f"`@{interaction.user.display_name}` has joined the conversation!")
            else:
                instance.chat_participants.remove(interaction.user.id)
                await interaction.response.send_message(f"`@{interaction.user.display_name}` has left the conversation!")
                await instance.on_user_leave(interaction.channel, interaction.user.display_name)
                # await instance.save_memory(interaction.user.display_name)

            if not instance.chat_participants:  # refresh
                instance.chat_client = None
                instance.chat_history = []
            if instance.chat_participants and instance.chat_client is None:
                async with interaction.channel.typing():
                    instance.chat_client = await asyncio.to_thread(functools.partial(
                        Client, "Be-Bo/llama-3-chatbot_70b"))

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
                width: int = int(
                    instance.defaults['aspect_ratio'].split(" x ")[0]),
                height: int = int(
                    instance.defaults['aspect_ratio'].split(" x ")[1]),
                quality_selector: str = instance.defaults['quality_selector'],
                style_selector: str = instance.defaults['style_selector'],
                seed: int = -1):

            prompt = instance.blacklist.replace(prompt, '')

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

        # moderation commands start here
        @app_commands.guild_only()
        @app_commands.default_permissions(kick_members=True)
        @instance.tree.command(name="kick", description="Kick a user from the Discord server.", )
        @app_commands.describe(member="Discord user", reason="Reason why")
        async def kick(interaction, member: discord.Member, *, reason: str = "None"):
            await member.kick(reason=reason)
            await interaction.response.send_message(f'> User `@{member.display_name}` has been kicked!')

        @app_commands.guild_only()
        @app_commands.default_permissions(administrator=True, ban_members=True)
        @instance.tree.command(name="ban", description="Ban a user from the Discord server.")
        @app_commands.describe(member="Discord user", reason="Reason why")
        async def ban(interaction, member: discord.Member, *, reason: str = "None"):
            await member.ban(reason=reason)
            await interaction.response.send_message(f'> User `@{member.display_name}` has been banned!')

        @app_commands.guild_only()
        @app_commands.default_permissions(administrator=True, ban_members=True)
        @instance.tree.command(name="unban", description="Unban a user from the Discord server.")
        @app_commands.describe(user_id="ID of the user to unban")
        async def unban(interaction, user_id: int):
            user = await bot.fetch_user(user_id)
            await interaction.guild.unban(user)
            await interaction.response.send_message(f'> User `@{user.display_name}` has been unbanned!')

        @app_commands.guild_only()
        @app_commands.default_permissions(administrator=True, mute_members=True)
        @instance.tree.command(name="mute", description="Mute a user's messages.")
        @app_commands.describe(member="Discord user", reason="Reason why", time="Length of mute in minutes")
        async def mute(interaction, member: discord.Member, time: int, *, reason: str = "None"):
            end_time = discord.utils.utcnow() + timedelta(minutes=time)

            await member.edit(timed_out_until=end_time, reason=reason)
            await interaction.response.send_message(f'> User `@{member.display_name}` has been muted for {time} minutes!')
            await member.send(f'You have been muted in {interaction.guild.name} for {time} minutes' +
                              (f' for the following reason: {reason}!' if reason else '!'))

        @app_commands.guild_only()
        @app_commands.default_permissions(administrator=True, mute_members=True)
        @instance.tree.command(name="unmute", description="Unmute a user's messages.")
        @app_commands.describe(member="Discord user")
        async def unmute(interaction, member: discord.Member):

            await member.edit(timed_out_until=None)

            await interaction.response.send_message(f'> User `@{member.display_name}` has been unmuted!')
            await member.send(f'You have been unmuted in {interaction.guild.name}.')

        @app_commands.guild_only()
        @app_commands.default_permissions(administrator=True, moderate_members=True)
        @instance.tree.command(name="warn", description="Warn a user and add them to the watchlist.")
        @app_commands.describe(member="Discord user", reason="Reason why")
        async def warn(interaction, member: discord.Member, *, reason: str = "None"):
            dm_channel = await member.create_dm()
            await dm_channel.send(f'You have been warned for the following reason: {reason}')
            watchlist[member.id] = reason
            await interaction.response.send_message(f'> User `@{member.display_name}` has been warned and added to the watchlist!')

        instance.run(token)


if __name__ == "__main__":
    load_dotenv()
    bot = Bot(token=os.getenv('TOKEN'))
