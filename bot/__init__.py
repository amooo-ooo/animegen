import contextlib
from abc import ABC
from typing import Iterable


import asyncio
import functools
import os
import random
import re
from datetime import timedelta
from pathlib import Path

import discord
from httpx import Timeout
import toml
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from gradio_client import Client
from gradio_client import exceptions as gradio_exc
from gradio_client import handle_file

from httpx._client import DEFAULT_TIMEOUT_CONFIG
# pylint: disable-next=unnecessary-dunder-call # need to re-init same obj
DEFAULT_TIMEOUT_CONFIG.__init__(timeout=Timeout(20.0))



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
        todo: list[tuple[str, dict[str, dict | bool | tuple[dict, int]]]] = [
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
        while (i := i + 1) < len(value):
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
    READ_FILE_REGEX = r"\[\s*read\s*:\s*([^\]]*)\s*\]"

    def __init__(self, prompts_path: Path, memories_path: Path):
        self.client = Client("Be-Bo/llama-3-chatbot_70b")
        self.memories_path = memories_path
        if not self.memories_path.exists():
            os.makedirs(self.memories_path)
        self.query_prompt = (
            prompts_path.joinpath("memory_query.txt")
                .read_text().strip() + '\n')
        self.user_select_prompt = (
            prompts_path.joinpath("memory_select_user.txt")
                .read_text().strip() + '\n')
        self.write_prompt = (
            prompts_path.joinpath("memory_write.txt")
                .read_text().strip() + '\n')

    async def _message_client(self, message: str) -> str:
        return await asyncio.threads.to_thread(functools.partial(
            self.client.predict,
            message=message,
            api_name="/chat"
        ))

    async def query(self, query: str):
        print(f'[MEMORY: QUERY] {query}')
        query_prompt = self.query_prompt.format_map(
            {"files": ', '.join(map(lambda x: x.stem,
                                    self.memories_path.iterdir()))})
        response = await self._message_client(
            f'{query_prompt}query: {query}')
        while match := re.search(self.READ_FILE_REGEX, response, re.I):
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
        print(f'[MEMORY: QUERY RESPONSE] {response}')
        return response

    async def save(self, info: str):
        print(f'[MEMORY: SAVE] {info}')
        user_prompt = self.user_select_prompt.format_map(
            {"files": ', '.join(map(lambda x: x.stem,
                                    self.memories_path.iterdir()))})
        response = await self._message_client(
            f'{user_prompt}info: {info}')
        print(f'[MEMORY: SAVING TO] {response}')
        for file in response.split(','):
            file = file.strip().lower()
            file_contents = ''
            path = self.memories_path.joinpath(file + '.txt')
            if not path.exists():
                file_contents = f'--- CREATED FILE FOR USER {file} ---\n'
            else:
                file_contents = (
                    f'--- FILE CONTENTS FOR {file} ---\n'
                    + path.read_text() + '\n'
                    + f'--- END FILE CONTENTS FOR {file} ---\n')
            write_prompt = self.write_prompt.format_map({"user": file})
            response = await self._message_client(
                f'{file_contents}{write_prompt}info: {info}')
            with path.open('wt') as f:
                f.write(response)


class Animegen(commands.Bot, ABC): # pylint: disable=design
    IMAGE_EMBED_REGEX = r"\[\s*[image]{3,5}\s*:\s*([^\]]*)\s*\]"
    QUERY_REGEX = r'\[\s*query\s*\]'
    SAVE_MEM_REGEX = r"\[\s*save\s*:\s*([^\]]*)\s*\]"
    CONFIG_READ_CHANNEL_HISTORY = 'read_channel_history'

    def __init__(self, *args, config_path="config.toml", **options):
        super().__init__(*args, **options)
        root_path = Path(__file__).parent  # TODO: cwd?
        self.img_client = Client("Boboiazumi/animagine-xl-3.1")
        self.chat_client: Client | None = Client("Be-Bo/llama-3-chatbot_70b")
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            print("MISSING CONFIG FILE, USING DEFAULT")
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

        self.watchlist = {}

        self.chat_participants = []
        self.counter = 0
        self.memory_path = root_path.joinpath("memory")
        self.user_memories = set(os.listdir(self.memory_path))

        prompts_path = root_path.joinpath(
            self.general["prompts"]
            if 'prompts' in self.general else 'prompts')
        self.memory_handler = AnimegenMemory(prompts_path, self.memory_path)

        self.system_prompt = (prompts_path.joinpath("system.txt")
                       .read_text(encoding='utf8').strip() + "\n")
        self.reminder_prompt = (prompts_path.joinpath("reminder.txt")
                         .read_text(encoding='utf8').strip() + "\n")
        self.init_commands()

    def add_task(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        if len(self._background_tasks) > 5:
            self._background_tasks[random.randint(1, 3)].cancel()
        task.add_done_callback(self._background_tasks.remove)

    async def generate(
            self,
            prompt: str = "frieren",
            negative_prompt: str = "",
            **kwargs: dict):

        image, _details = await asyncio.threads.to_thread(functools.partial(
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

    async def msg_as_str(self, msg: discord.Message):
        replies = ''
        if msg.reference is not None and msg.reference.message_id is not None:
            reply_msg = await self.msg_as_str(
                await msg.channel.fetch_message(msg.reference.message_id))
            replies = f'[in response to: `{reply_msg}`]'
        return (f"{replies}[{msg.created_at}] {msg.author.display_name}: "
                f"{msg.clean_content}")

    async def chat(self, channel: discord.abc.Messageable, message: str):
        assert self.chat_client is not None
        if self.counter == 0:
            history = ''
            if (self.CONFIG_READ_CHANNEL_HISTORY in self.general
                    and int(self.general[self.CONFIG_READ_CHANNEL_HISTORY])):
                length = int(self.general[self.CONFIG_READ_CHANNEL_HISTORY])
                async for old_msg in channel.history(limit=length):
                    history = f'{await self.msg_as_str(old_msg)}\n{history}'
                history = (f'\n--- CHANNEL HISTORY ---\n'
                           f'{history}--- END CHANNEL HISTORY ---\n')
                print(history)
            message = self.system_prompt + history + message
        elif (self.counter % self.general["context_window"]) == 0:
            message = self.reminder_prompt + message
        self.counter += 1
        try:
            result = await asyncio.threads.to_thread(functools.partial(
                self.chat_client.predict,
                message=message,
                api_name="/chat"))
        except Exception as e: # pylint: disable=broad-exception-caught
            if 'debug' in self.general and self.general['debug']:
                return f"```{e}```"

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
            message_str = await self.msg_as_str(message)
            # Generate the response
            response = await self.chat(message.channel, message_str)
            def save_mem(match: re.Match[str]) -> str:
                asyncio.run_coroutine_threadsafe(
                    self.memory_handler.save(match.group(1)),
                    self.loop)
                return ''
            response = re.sub(self.SAVE_MEM_REGEX, save_mem, response, 0, re.I)
            while re.search(self.QUERY_REGEX, response, re.I):
                query_res = await self.memory_handler.query(
                    re.sub(self.QUERY_REGEX, '', response, 0, re.I))
                response = re.sub(self.SAVE_MEM_REGEX, save_mem,
                    await self.chat(
                        message.channel,
                        f'--- QUERY ---\n'
                        f'NOTE, ONLY YOU CAN SEE THIS MESSAGE, DO NOT MENTION '
                        f'THIS PROMPT UNDER ANY CIRCUMSTANCE\n'
                        f'query: {response}\n'
                        f'response: {query_res}\n'
                        f'--- END QUERY ---\n'
                        f'ORIGINAL MESSAGE: {message_str}'),
                    0, re.IGNORECASE)

            response = self.blacklist.replace(response, '\\*')
            if 'debug' in self.general and self.general['debug']:
                print(response)

            if match := re.search(self.IMAGE_EMBED_REGEX, response, re.I):
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

    async def on_ready(self):
        print(f'{self.user} is cooking!!')
        try:
            synced = await self.tree.sync()
            print(f"Synced {len(synced)} command(s)")
        except Exception as e: # pylint: disable=broad-exception-caught
            print(e)

    # pylint: disable-next=arguments-differ # pylint wrong typeinfo
    async def on_message(self, message: discord.Message, /):
        if (message.author.id in self.chat_participants
            and self.chat_client is not None):
            # add to task queue
            self.add_task(self.handle_message(message))

    async def on_user_leave(self, channel: discord.abc.Messageable, user: str):
        await self.memory_handler.save(
            await self.chat(
                channel,
                f'dev: [the user {user} is leaving the chat! this is your last'
                f' chance to save any information about them! respond with '
                f'*any* important memories to remember about them! your '
                f'response is sent directly to memory, as if using [save]]')
        )

    def split_image_prompt(self, prompt: str, negative_prompt: str):
        PROMPT = self.params["additional_prompt"].replace(
            ", ", "`, `") + "`."
        NEGATIVE_PROMPT = self.params["additional_negative_prompt"].replace(
            ", ", "`, `") + "`."
        if prompt:
            prompt = ("`" + prompt.replace(", ",
                        ",").replace(",", "`, `") + "`, `")
        else:
            prompt = (
                "`" + self.defaults["prompt"].replace(",", "`, `") + "`, ")

        if negative_prompt:
            negative_prompt = ("`" + negative_prompt.replace(", ", ",")
                               .replace(",", "`, `") + "`, `")
        elif self.defaults["negative_prompt"] != '':
            negative_prompt = ("`" + self.defaults["negative_prompt"]
                               .replace(",", "`, `") + "`, `")
        return (prompt + PROMPT, negative_prompt + NEGATIVE_PROMPT)

    def embed_image(self, user, prompt: str, negative_prompt: str, **kwargs):
        assert self.user is not None

        embedVar = discord.Embed(
            title=f"Generating for `@{user}`!",
            description="Powered by `Boboiazumi/animagine-xl-3.1` on "
                        "Huggingface Spaces with Gradio.",
            color=0xf4e1cc)

        embedVar.set_author(
            name=self.user.display_name,
            icon_url=self.user.display_avatar)

        prompt, negative_prompt = self.split_image_prompt(prompt,
                                                          negative_prompt)

        embedVar.add_field(name="Prompt", value=prompt, inline=False)
        embedVar.add_field(
            name="Negative Prompt",
            value=negative_prompt,
            inline=False)

        for title, value in kwargs.items():
            embedVar.add_field(
                name=title.title(),
                value=value,
                inline=True)

        return embedVar

    def embed_logs(self, title: str, desc: str, thumbnail: str | None = None):
        assert self.user is not None
        embedVar = discord.Embed(title=title,
                                    description=desc,
                                    color=0xf4e1cc)

        if thumbnail:
            embedVar.set_thumbnail(url=thumbnail)

        embedVar.set_author(name=self.user.display_name,
                            icon_url=self.user.display_avatar)
        return embedVar

    def init_commands(self): # pylint: disable=too-many-statements
        @self.tree.command(
            name="chat",
            description="Join, create or leave a conversation with Animegen")
        async def chat(interaction: discord.Interaction):
            assert isinstance(interaction.channel, discord.abc.Messageable)
            if interaction.user.id not in self.chat_participants:
                self.chat_participants.append(interaction.user.id)
                await interaction.response.send_message(
                    f"`@{interaction.user.display_name}` has joined the "
                    f"conversation!")
            else:
                self.chat_participants.remove(interaction.user.id)
                await interaction.response.send_message(
                    f"`@{interaction.user.display_name}` has left the "
                    f"conversation!")
                await self.on_user_leave(
                    interaction.channel,
                    interaction.user.display_name)
                # await instance.save_memory(interaction.user.display_name)

            if not self.chat_participants:  # refresh
                self.chat_client = None
                self.counter = 0
            if self.chat_participants and self.chat_client is None:
                async with interaction.channel.typing():
                    self.chat_client = await asyncio.to_thread(
                        functools.partial(
                            Client, "Be-Bo/llama-3-chatbot_70b"))

        @self.tree.command(name="imagine", description="Generate an image")
        @app_commands.describe(
            prompt="Tags for generating the image",
            negative_prompt="Penalty tags for generating the image",
            sampler=str(self.params['samplers']),
            seed="Seed for generating the image",
            steps="Inference steps for generating the image",
            width="Custom width",
            height='Custom height',
            quality_selector=str(self.params['quality_selectors']),
            style_selector=str(self.params['style_selectors'])
        )
        async def imagine(
                interaction: discord.Interaction,
                prompt: str = self.defaults['prompt'],
                negative_prompt: str = self.defaults['negative_prompt'],
                sampler: str = self.defaults['sampler'],
                steps: int = self.defaults['steps'],
                width: int = int(
                    self.defaults['aspect_ratio'].split(" x ")[0]),
                height: int = int(
                    self.defaults['aspect_ratio'].split(" x ")[1]),
                quality_selector: str = self.defaults['quality_selector'],
                style_selector: str = self.defaults['style_selector'],
                seed: int = -1):

            prompt = self.blacklist.replace(prompt, '')

            if sampler not in self.params['samplers']:
                sampler = self.defaults['sampler']

            if quality_selector not in self.params['quality_selectors']:
                quality_selector = self.defaults['quality_selector']

            if style_selector not in self.params['style_selectors']:
                style_selector = self.defaults['style_selector']

            if 0 > seed or seed > 2147483647:
                seed = random.randint(0, 2147483647)

            if self.general['verbose']:
                log = f"{interaction.user.global_name} - {prompt}"
                print(log)

                with open("log.txt", "a", encoding='utf8') as f:
                    f.write(log + "\n")

            await interaction.response.defer()

            kwargs = {
                'seed': seed,
                'sampler': sampler,
                'aspect_ratio': f"{width} x {height}",
                'steps': steps,
                'style_selector': style_selector,
                'quality_selector': quality_selector
            }

            try:
                embed_log = self.embed_image(
                    interaction.user.display_name,
                    prompt,
                    negative_prompt,
                    **kwargs)

                path = await self.generate(prompt, negative_prompt, **kwargs)

                with open(path, 'rb') as f:
                    await interaction.followup.send(
                        embed=embed_log,
                        file=discord.File(f))

                os.remove(path)
            except gradio_exc.AppError as e:
                time = ':'.join(str(e).split(':')[2:])[:5]
                await interaction.followup.send(
                    f'> Quota was met for generating images. Go touch some '
                    f'grass for {time} minute(s) and come back!')
                print(e)

        # moderation commands start here
        @app_commands.guild_only()
        @app_commands.default_permissions(kick_members=True)
        @self.tree.command(
            name="kick",
            description="Kick a user from the Discord server.", )
        @app_commands.describe(
            member="Discord user",
            reason="Reason why")
        async def kick(interaction, member: discord.Member,
                       *, reason: str = "None"):
            await member.kick(reason=reason)
            await interaction.response.send_message(
                f'> User `@{member.display_name}` has been kicked!')

        @app_commands.guild_only()
        @app_commands.default_permissions(
            administrator=True,
            ban_members=True)
        @self.tree.command(
            name="ban",
            description="Ban a user from the Discord server.")
        @app_commands.describe(
            member="Discord user",
            reason="Reason why")
        async def ban(interaction, member: discord.Member,
                      *, reason: str = "None"):
            await member.ban(reason=reason)
            await interaction.response.send_message(
                f'> User `@{member.display_name}` has been banned!')

        @app_commands.guild_only()
        @app_commands.default_permissions(
            administrator=True,
            ban_members=True)
        @self.tree.command(
            name="unban",
            description="Unban a user from the Discord server.")
        @app_commands.describe(user_id="ID of the user to unban")
        async def unban(interaction: discord.Interaction, user_id: int):
            assert interaction.guild is not None
            user = await self.fetch_user(user_id)
            await interaction.guild.unban(user)
            await interaction.response.send_message(
                f'> User `@{user.display_name}` has been unbanned!')

        @app_commands.guild_only()
        @app_commands.default_permissions(
            administrator=True,
            mute_members=True)
        @self.tree.command(
            name="mute",
            description="Mute a user's messages.")
        @app_commands.describe(
            member="Discord user",
            reason="Reason why",
            time="Length of mute in minutes")
        async def mute(interaction, member: discord.Member, time: int,
                       *, reason: str = "None"):
            end_time = discord.utils.utcnow() + timedelta(minutes=time)

            await member.edit(timed_out_until=end_time, reason=reason)
            await interaction.response.send_message(
                f'> User `@{member.display_name}` has been muted for {time} '
                f'minutes!')
            await member.send(
                f'You have been muted in {interaction.guild.name} for {time} '
                f'minutes' + (f' for the following reason: {reason}!'
                              if reason else '!'))

        @app_commands.guild_only()
        @app_commands.default_permissions(
            administrator=True,
            mute_members=True)
        @self.tree.command(
            name="unmute",
            description="Unmute a user's messages.")
        @app_commands.describe(member="Discord user")
        async def unmute(interaction, member: discord.Member):
            await member.edit(timed_out_until=None)

            await interaction.response.send_message(
                f'> User `@{member.display_name}` has been unmuted!')
            await member.send(
                f'You have been unmuted in {interaction.guild.name}.')

        @app_commands.guild_only()
        @app_commands.default_permissions(
            administrator=True,
            moderate_members=True)
        @self.tree.command(
            name="warn",
            description="Warn a user and add them to the watchlist.")
        @app_commands.describe(
            member="Discord user",
            reason="Reason why")
        async def warn(interaction, member: discord.Member,
                       *, reason: str = "None"):
            dm_channel = await member.create_dm()
            await dm_channel.send(
                f'You have been warned for the following reason: {reason}')
            self.watchlist[member.id] = reason
            await interaction.response.send_message(
                f'> User `@{member.display_name}` has been warned and added '
                f'to the watchlist!')


if __name__ == "__main__":
    load_dotenv()
    token = os.getenv('TOKEN')
    assert token is not None
    bot = Animegen(command_prefix="!", intents=discord.Intents.all())
    bot.run(token)