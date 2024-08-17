# pylint: disable=missing-docstring
import contextlib
from abc import ABC, abstractmethod
import datetime
import itertools
from typing import Iterable, Literal, Optional


import asyncio
import os
import random
import re
from datetime import timedelta
from pathlib import Path

import discord
from httpx import Timeout
from httpx._config import DEFAULT_TIMEOUT_CONFIG
import toml
from discord import app_commands
from discord.ext import commands
from gradio_client import Client
from gradio_client import exceptions as gradio_exc
from gradio_client import handle_file
import youtube_dl

from bot.xsexyreload import SexyReloader

# pylint: disable-next=unnecessary-dunder-call # need to re-init same obj
DEFAULT_TIMEOUT_CONFIG.__init__(timeout=Timeout(200.0))


# Models:
# 'Be-Bo/llama-3-chatbot_70b',
# 'vilarin/Llama-3.1-8B-Instruct',
# 'orionai/llama-3.1-70b-demo'


class ChatModel(ABC):
    _system_prompt: str
    _reprompt_frequency: int
    _chat_client: Optional[Client]
    _chat_client_model: str
    _counter: int

    def __init__(
            self,
            system_prompt: str, *,
            reprompt_frequency: int = 1,
            model: str = 'Be-Bo/llama-3-chatbot_70b') -> None:
        self._system_prompt = system_prompt
        self._reprompt_frequency = reprompt_frequency
        self._chat_client_model = model
        self._chat_client = None
        self._counter = 0

    async def reset_client(self):
        self._chat_client = await asyncio.to_thread(
            Client, self._chat_client_model)
        self._counter = 0

    @abstractmethod
    async def _predict(self, msg: str, additional_info: str) -> str:
        pass

    def use_prompt(self, system_prompt: str):
        self._system_prompt = system_prompt

    def if_prompt(self) -> str | Literal['']:
        if (self._counter % self._reprompt_frequency) == 0:
            return self._system_prompt
        return ''

    async def send(self, msg: str, **extra_info: str) -> str:
        extra_info_str = '\n'.join(
            f'--- {section} ---\n{content}\n--- END {section} ---'
            for section, content in extra_info.items())
        result = None
        while result is None:
            try:
                result = await self._predict(msg, extra_info_str)
                self._counter += 1
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(e)
                if isinstance(e, gradio_exc.AppError):
                    await self.reset_client()
        return result

    def close(self):
        if self._chat_client is not None:
            self._chat_client.close()
        self._chat_client = None
        self._counter = 0

    def is_open(self) -> bool:
        return self._chat_client is not None


class BeepBoop(ChatModel):
    def __init__(
            self,
            system_prompt: str, *,
            reprompt_frequency: int = 1) -> None:
        super().__init__(
            system_prompt,
            reprompt_frequency=reprompt_frequency,
            model='Be-Bo/llama-3-chatbot_70b')

    def append_history_message(self, _: str):
        pass

    async def _predict(self, msg: str, additional_info: str) -> str:
        if self._chat_client is None:
            await self.reset_client()
        assert self._chat_client is not None
        final_msg = (
            f'{additional_info}\n{self.if_prompt()}\n'
            f'{msg}')
        print(f'<<<{final_msg}>>>')
        return await asyncio.threads.to_thread(
            self._chat_client.predict,
            message=final_msg,
            api_name="/chat")


class OrionAiModel(ChatModel):
    _history_capacity: int
    _history: list[str]

    def __init__(
            self,
            system_prompt: str, *,
            reprompt_frequency: int = 1,
            history_capacity: int = 64) -> None:
        super().__init__(
            system_prompt,
            reprompt_frequency=reprompt_frequency,
            model='orionai/llama-3.1-70b-demo')
        self._history_capacity = history_capacity
        self._history = []

    def append_history_message(self, msg: str):
        self._history.append(msg)
        if len(self._history) > self._history_capacity:
            self._history.pop(0)

    async def _predict(self, msg: str, additional_info: str) -> str:
        if self._chat_client is None:
            await self.reset_client()
        assert self._chat_client is not None
        self.append_history_message(msg)
        final_msg = (
            f'{additional_info}\n{self._system_prompt}\n'
            f'{'\n'.join(self._history)}')
        print(f'<<<{final_msg}>>>')
        return await asyncio.threads.to_thread(
            self._chat_client.predict,
            user_message=final_msg,
            api_name="/predict")


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


async def msg_as_str(msg: discord.Message):
    replies = ''
    if msg.reference is not None and msg.reference.message_id is not None:
        reply_msg = await msg_as_str(
            await msg.channel.fetch_message(msg.reference.message_id))
        replies = f'[in response to: `{reply_msg}`]'
    chnl = (
        f'@{msg.channel.recipient}' if isinstance(msg.channel, discord.DMChannel)
        else f'<#{msg.channel.id}>' if isinstance(msg.channel, discord.PartialMessageable)
        else f'#{msg.channel.name}')
    return (f"{replies}[{msg.created_at}:{chnl}] {msg.author.name}: "
            f"{msg.clean_content}")


class AnimegenMemory:
    READ_FILE_REGEX = re.compile(r"\[\s*read\s*:\s*([^\]]*)\s*\]", re.I)
    MISSING_REGEX = re.compile(r'\[\s*none\s*\]', re.I)

    def __init__(
            self,
            prompts_path: Path,
            memories_path: Path,
            debug: bool = False):
        # should be initialized before use
        self._event_loop: asyncio.AbstractEventLoop = None  # type: ignore
        self.client = BeepBoop('')  # , history_capacity=1
        self.background_tasks: set[asyncio.Task] = set()
        self.memories_path = memories_path
        self.debug = debug
        self._queued_to_save: list[tuple[datetime.datetime, str]] = []
        if not self.memories_path.exists():
            os.makedirs(self.memories_path)
        self.query_prompt = (
            prompts_path.joinpath("memory_query.txt")
            .read_text(encoding='utf8').strip() + '\n')
        self.user_select_prompt = (
            prompts_path.joinpath("memory_select_user.txt")
            .read_text(encoding='utf8').strip() + '\n')
        self.write_prompt = (
            prompts_path.joinpath("memory_write.txt")
            .read_text(encoding='utf8').strip() + '\n')

    async def init_client(self):
        await self.client.reset_client()

    async def handle_message(self, message: discord.Message) -> str | None:
        visited = set()
        query_prompt = self.query_prompt.format_map(
            {"files": ', '.join(map(lambda x: x.stem,
                                    self.memories_path.iterdir()))})
        message_str = await msg_as_str(message)
        read_queue = [message.author.name.lower().strip()]
        final_response = ''
        while read_queue:
            file_name = read_queue.pop()
            if file_name in visited:
                continue
            visited.add(file_name)
            file = self.memories_path.joinpath(file_name + '.txt')
            if file.exists():
                file_content = file.read_text(encoding='utf8')
            else:
                file_content = 'NO CURRENT MEMORIES FOR THIS USER'
            self.client.use_prompt(query_prompt)
            response = await self.client.send(
                message_str, **{f'MEMORIES ABOUT {file_name}': file_content})
            if self.debug:
                print(f'[MEMORY: QUERY RESPONSE from {file_name}] {response}>')
            read_queue.extend(itertools.chain(*map(
                lambda match: map(str.strip,
                                  match.group(1).lower().split(',')),
                re.finditer(self.READ_FILE_REGEX, response))))
            if not re.search(self.MISSING_REGEX, response):
                final_response += (
                    f'[memories about {file_name}] '
                    f'{re.sub(self.READ_FILE_REGEX, "", response)}\n')
        await self.queue_save(message_str)
        return None if not final_response else final_response

    async def queue_save(self, message: str):
        time = datetime.datetime.now(datetime.UTC)
        self._queued_to_save.append((time, message))
        if len(self._queued_to_save) > 10:
            self._event_loop.create_task(self.save())

    async def save(self):
        messages = ''
        if not self._queued_to_save:
            return
        for time, msg in self._queued_to_save:
            messages += f'[{time}] {msg}\n'
        self._queued_to_save.clear()
        user_prompt = self.user_select_prompt.format_map(
            {"files": ', '.join(map(lambda x: x.stem,
                                    self.memories_path.iterdir()))})
        self.client.use_prompt(user_prompt)
        response = await self.client.send(messages)
        if self.debug:
            print(f'[MEMORY: SAVING TO] {response}')
        for file in response.split(','):
            file = file.strip().lower()
            file_contents = ''
            path = self.memories_path.joinpath(file + '.txt')
            if not path.exists():
                file_contents = 'NO CURRENT MEMORIES - CREATING NEW FILE!'
            else:
                file_contents = path.read_text(encoding='utf8')
            write_prompt = self.write_prompt.format_map(
                {"user": file})
            self.client.use_prompt(write_prompt)
            response = await self.client.send(
                messages, **{f"MEMORIES ABOUT {file}": file_contents})
            with path.open('wt', encoding='utf8') as f:
                f.write(response)


class Animegen(commands.Bot, ABC):  # pylint: disable=design
    IMAGE_EMBED_REGEX = re.compile(
        r"\[\s*ima?ge?\s*:\s*([^\]]*)\s*\]", re.I)
    QUERY_REGEX = re.compile(r'\[\s*query\s*\]', re.I)
    SAVE_MEM_REGEX = re.compile(r"\[\s*save\s*:\s*([^\]]*)\s*\]", re.I)
    MISSING_REGEX = re.compile(r'\[\s*none\s*\]', re.I)
    CONFIG_READ_CHANNEL_HISTORY = 'read_channel_history'
    NAME = 'astolfo'

    def __init__(self, root_path: Path = Path(__file__).parent,
                 *args, **options):
        super().__init__(*args, **options)
        self.img_client: Client = None  # type: ignore
        cfg_path = root_path.joinpath('config.toml')
        if not cfg_path.exists():
            print("MISSING CONFIG FILE, USING DEFAULT")
            cfg_path = root_path.joinpath('config.default.toml')
        self.config = toml.load(cfg_path)

        self._background_tasks: list[asyncio.Task] = []

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
        self.debug = bool('debug' in self.general and self.general['debug'])
        self.blacklist_channel_ids = (
            set(self.general['blacklist_channel_ids'])
            if 'blacklist_channel_ids' in self.general else ())

        self.watchlist = {}

        self.chat_participants = []
        self.memory_path = root_path.joinpath("memory")
        self.user_memories = set(os.listdir(self.memory_path))

        self.vc: discord.VoiceClient | None = None
        self.song_q: list[Path] = []
        self.now_playing: Path | None = None

        prompts_path = root_path.joinpath(
            self.general['prompts_dir']
            if 'prompts_dir' in self.general else 'prompts')
        self.memory_handler = AnimegenMemory(
            prompts_path,
            self.memory_path,
            debug=self.debug)

        self.system_prompt = (prompts_path.joinpath("system.txt")
                              .read_text(encoding='utf8').strip() + "\n")
        self.reminder_prompt = (prompts_path.joinpath("reminder.txt")
                                .read_text(encoding='utf8').strip() + "\n")

        self.chat_client = BeepBoop(
            self.system_prompt,
            reprompt_frequency=self.general['context_window'])

        youtube_dl.std_headers["User-Agent"] = (
            "Mozilla/5.0 (Linux; Android 10; K) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.6533.103 Mobile Safari/537.36")
        self.ytdl = youtube_dl.YoutubeDL({
            "format": "140",
            "outtmpl": str(Path(self.general['music_dir']).absolute()
                           / youtube_dl.DEFAULT_OUTTMPL)
        })

        self.hotreload: SexyReloader | None = None

        if 'debug_guilds' in self.general:
            self.init_commands(debug_guilds=self.general['debug_guilds'])
        else:
            self.init_commands()

    async def setup_hook(self) -> None:
        # pylint: disable-next=protected-access
        self.memory_handler._event_loop = self.loop
        return await super().setup_hook()

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
            base_image: str | discord.Attachment
        = 'https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png',
            use_base_image: bool = False,
            **kwargs: dict):

        image, _details = await asyncio.threads.to_thread(
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
            isImg2Img=use_base_image,
            img_path=handle_file(
                base_image.url
                if isinstance(base_image, discord.Attachment) else
                base_image),
            img2img_strength=kwargs.get(
                "base_image_strength", 0.65),
            api_name="/run"
        )

        path = image[0]["image"]
        return path

    async def chat(
            self,
            channel: discord.abc.Messageable,
            message: str,
            **extra_info: str) -> str:
        kwargs = extra_info
        if (self.chat_client._counter == 0
            and self.CONFIG_READ_CHANNEL_HISTORY in self.general
                and int(self.general[self.CONFIG_READ_CHANNEL_HISTORY])):
            # Get channel history
            length = int(self.general[self.CONFIG_READ_CHANNEL_HISTORY])
            history = ''
            async for old_msg in channel.history(limit=length):
                history = f'{await msg_as_str(old_msg)}\n{history}'
            kwargs['CHANNEL HISTORY'] = history

        return await self.chat_client.send(message, **kwargs)

    @contextlib.asynccontextmanager
    async def gen_image(
            self, prompt,
            typing_channel: discord.abc.Messageable | None = None):
        try:
            if typing_channel is not None:
                async with typing_channel.typing():
                    path = Path(await self.generate(prompt))
            else:
                path = Path(await self.generate(prompt))
            try:
                with path.open('rb') as f:
                    yield discord.File(f, filename=path.name)
            finally:
                # os.remove(path)
                pass
        except Exception as e:  # pylint: disable=broad-exception-caught
            yield e

    async def handle_message(self, message: discord.Message):
        # Defer the response
        async with message.channel.typing():
            memory_ctx = await self.memory_handler.handle_message(message)
            kwargs = {}
            if memory_ctx is not None:
                kwargs['LONG TERM MEMORY'] = (
                    f'NOTE ONLY YOU CAN SEE THIS\n{memory_ctx}')
            message_str = await msg_as_str(message)
            # Generate the response
            response = await self.chat(message.channel, message_str, **kwargs)
        response = self.blacklist.replace(response, '\\*')
        if self.debug:
            print(response)

        if re.search(self.MISSING_REGEX, response):
            return  # No response

        matches: list[str] = re.split(self.IMAGE_EMBED_REGEX, response, re.I)
        image_gen_failed = False
        last_message = None
        for i, msg in enumerate(matches):
            if not msg.strip():
                continue
            if (i % 2) == 1:  # Image prompt
                if image_gen_failed:
                    continue
                async with self.gen_image(
                        msg, typing_channel=message.channel) as img:
                    if isinstance(img, Exception):
                        async with message.channel.typing():
                            response = await self.chat(
                                message.channel,
                                f'dev: [image generation failed with error: '
                                f'"{img}". '
                                f'please send a followup message including '
                                f'quota time till reset if available (please '
                                f'note images will not work in followup)]')
                            response = self.blacklist.replace(response,
                                                              '\\*')
                        await message.channel.send(response)
                        await self.memory_handler.queue_save(
                            f'{self.NAME}: {response}')
                        image_gen_failed = True
                    else:
                        if last_message is not None:
                            last_message = await last_message.add_files(img)
                        else:
                            last_message = await message.channel.send(file=img)
                        await self.memory_handler.queue_save(
                            f'{self.NAME}: [image: {msg}]')
            else:
                while len(msg) > 2000:
                    split_idx = msg.rfind('\n', None, 2000)
                    if split_idx < 0:
                        msg.rfind(' ', None, 2000)
                    if split_idx < 0:
                        pass  # uh oh
                    await message.channel.send(msg[:split_idx])
                    msg = msg[split_idx + 1:]
                last_message = await message.channel.send(msg)
                await self.memory_handler.queue_save(
                    f'{self.NAME}: {response}')

    async def prep_img_client(self):
        self.img_client = await asyncio.to_thread(
            Client, "Boboiazumi/animagine-xl-3.1")

    def filesystem_update(self, mod: set[str]):
        if mod:
            print(f'Polled filesystem, updates in: {mod}')
        if Animegen.init_commands.__qualname__ in mod:
            # Resync commands
            if 'debug_guilds' in self.general:
                self.tree.clear_commands(guild=None)
                for i in self.general['debug_guilds']:
                    guild = discord.Object(i, type=discord.Guild)
                    self.tree.clear_commands(guild=guild)
            else:
                self.tree.clear_commands(guild=None)
            if 'debug_guilds' in self.general:
                self.init_commands(
                    debug_guilds=self.general['debug_guilds'])
            else:
                self.init_commands()
            self.loop.create_task(self.sync_commands())

    # async def poll_filesystem(self):
    #     assert self.hotreload is not None
    #     try:
    #         mod = self.hotreload.poll()
    #     except Exception as e:
    #         print(f'Exception while hotreloading: {e}')
    #     finally:
    #         self.loop.create_task(self.poll_filesystem())

    async def sync_commands(self):
        try:
            if 'debug_guilds' in self.general:
                await self.tree.sync()
                for i in self.general['debug_guilds']:
                    guild = discord.Object(i, type=discord.Guild)
                    synced = await self.tree.sync(guild=guild)
                    print(f"Synced {len(synced)} command(s) to {guild}")
            else:
                synced = await self.tree.sync()
                print(f"Synced {len(synced)} command(s)")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(e)

    async def on_ready(self):
        print(f'{self.user} is cooking!!')
        self.loop.create_task(self.prep_img_client())
        if self.debug:
            self.hotreload = SexyReloader(lambda: globals(), self.loop)
            # pylint: disable-next=unnecessary-lambda
            self.hotreload.watch(lambda m: self.filesystem_update(m))
            # self.loop.create_task(self.poll_filesystem())
            self.loop.create_task(self.chat_client.reset_client())
            self.loop.create_task(self.memory_handler.init_client())
        await self.sync_commands()

    # pylint: disable-next=arguments-differ # pylint wrong typeinfo
    async def on_message(self, message: discord.Message, /):
        if self.user is not None and message.author.id == self.user.id:
            self.chat_client.append_history_message(await msg_as_str(message))
        if (message.channel.id not in self.blacklist_channel_ids
                and message.author.id in self.chat_participants):
            # add to task queue
            self.add_task(self.handle_message(message))

    async def on_user_leave(self, _channel: discord.abc.Messageable,
                            _user: discord.User | discord.Member):
        pass

    COMMA_SEPERATOR_REGEX = re.compile(r',\s*')

    def split_image_prompt(self, prompt: str, negative_prompt: str):
        if not prompt:
            prompt = self.defaults["prompt"]
        if not negative_prompt:
            negative_prompt = self.defaults["negative_prompt"]
        prompt += ', ' + self.params["additional_prompt"]
        negative_prompt += ', ' + self.params["additional_negative_prompt"]
        prompt = ', '.join(map(
            lambda tag: f'`{tag}`',
            re.split(self.COMMA_SEPERATOR_REGEX, prompt.strip(', '))))
        negative_prompt = ', '.join(map(
            lambda tag: f'`{tag}`',
            re.split(self.COMMA_SEPERATOR_REGEX, negative_prompt.strip(', '))))
        return (prompt + '.', negative_prompt + '.')

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

        if 'use_base_image' in kwargs and kwargs['use_base_image']:
            assert isinstance(kwargs['base_image'], discord.Attachment)
            embedVar.set_image(url=kwargs['base_image'].url)

        for title, value in kwargs.items():
            if title in ('use_base_image', 'base_image'):
                continue
            embedVar.add_field(
                name=title.title().replace('_', ' '),
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

    async def add_user(self, ctx: discord.Interaction,
                       user: discord.Member | discord.User):
        assert isinstance(ctx.channel, discord.abc.Messageable)
        msg = ''
        if user.id not in self.chat_participants:
            self.chat_participants.append(user.id)
            msg = f"{user.mention} has joined the conversation!"
        else:
            self.chat_participants.remove(user.id)
            msg = f"{user.mention} has left the conversation!"
            await self.on_user_leave(ctx.channel, user)

        deferred = False
        if not self.chat_participants:  # refresh
            self.chat_client.close()
            for task in self._background_tasks:
                task.cancel()
            deferred = True
            await ctx.response.defer(thinking=True)
            await self.memory_handler.save()
        if self.chat_participants and not self.chat_client.is_open():
            deferred = True
            await ctx.response.defer(thinking=True)
            await self.chat_client.reset_client()
        if deferred:
            await ctx.followup.send(
                msg,
                allowed_mentions=discord.AllowedMentions(users=False))
        else:
            await ctx.response.send_message(
                msg,
                allowed_mentions=discord.AllowedMentions(users=False))

    async def _music_loop(self):
        rt = Path(self.general['music_dir'])
        assert self.vc is not None
        if not self.song_q:
            self.song_q = list(rt.rglob("*.m4a"))
            random.shuffle(self.song_q)
        # print(list(map(lambda f: f.name, self.q)))
        self.now_playing = self.song_q.pop()
        print(f'now playing {self.now_playing.name}')
        f = discord.FFmpegPCMAudio(str(self.now_playing.absolute()))
        play_fut = self.vc.loop.create_future()
        self.vc.play(
            f,
            after=play_fut.set_result,
            application='lowdelay',
            bandwidth='full',
            bitrate=128,
            expected_packet_loss=0.25,
            signal_type='music')
        await play_fut
        play_fut = None
        self.now_playing = None

    async def play_music(self):
        while True:
            await self._music_loop()

    def init_commands(self, debug_guilds: list[int] | None = None):
        kwargs = {}
        if debug_guilds is not None:
            kwargs['guilds'] = list(map(
                # Cannot use self.get_guild, not ready yet!
                lambda id: discord.Object(id, type=discord.Guild),
                debug_guilds))

        @self.tree.command(
            name="music",
            description="paly music",
            **kwargs)
        async def music(ctx: discord.Interaction):
            assert isinstance(ctx.user, discord.Member)
            if ctx.user.voice and ctx.user.voice.channel:
                self.vc = await ctx.user.voice.channel.connect(self_deaf=True)
                self.vc.loop.create_task(self.play_music())
                await ctx.response.send_message("sure, iwll sing for u mastwerr :D!! >.<")
            else:
                await ctx.response.send_message("mastwerr, i cant sing for u if ur not in vc!! >.<")

        @self.tree.command(
            name="skip",
            description="goo'bie songg",
            **kwargs)
        async def skip(ctx: discord.Interaction):
            if self.vc is None:
                await ctx.response.send_message(
                    "no song cuwwently pwayying!! >.<", ephemeral=True)
                return
            self.vc.stop()
            await ctx.response.send_message(
                "yeah, i didnt rweally lik that song either >.<")

        @self.tree.command(
            name="jumpscare",
            description="shhh",
            **kwargs)
        async def jumpscare(ctx: discord.Interaction):
            if self.vc is None:
                await ctx.response.send_message(
                    "no song cuwwently pwayying!! >.<", ephemeral=True)
                return
            self.song_q.append(Path(self.general['music_dir'])
                               / "H Anime ASMR-ZgTck7vGwIk.m4a")
            self.vc.stop()
            await ctx.response.send_message(
                "got u bwoskwii! ^-^ >.<", ephemeral=True)

        @self.tree.command(
            name="queue",
            description="make song get queues to play",
            **kwargs)
        @app_commands.describe(
            song="Song to play or Url to download (using ytdl)")
        async def queue(ctx: discord.Interaction, song: Optional[str]):
            if song is None:
                songs = ''.join(f"\n{i}. `{name}`" for i, (f, name) in enumerate(
                    (path, re.sub(
                        r"(.*?)-[\-_a-zA-Z0-9]*\.\w{2,4}", r"\1", path.name))
                    for path in reversed(self.song_q))
                    if i < 10)
                if self.now_playing is not None:
                    np = re.sub(r"(.*?)-[\-_a-zA-Z0-9]*\.\w{2,4}", r"\1",
                                self.now_playing.name)
                    np = f'currently pwayingw `{np}` >.<'
                else:
                    np = ('not cuwwentwy playingw anythiwing.. i can change '
                          'that for u tho mastwerr!! >.<')
                await ctx.response.send_message(
                    f"{np}\ncomignw upp:{songs}")
                return
            root = Path(self.general['music_dir'])
            sub = root / song
            if sub.is_relative_to(root) and sub.exists():
                self.song_q.append(sub)
                await ctx.response.send_message(
                    "okaii mastwerr iwll sing thaat next!! >.<")
                return
            await ctx.response.defer(thinking=True)
            try:
                info = await asyncio.to_thread(self.ytdl.extract_info, song)
                if not isinstance(info, list):
                    info = [info] if info is not None else []
                for i in info:
                    self.song_q.append(
                        Path(self.general['music_dir']).absolute()
                        / self.ytdl.prepare_filename(i))
                await ctx.followup.send("downlaoded song i think >.<")
            except Exception as e:
                await ctx.followup.send(
                    f"somthing went wrong o.o sowwwy mastwer!! ```{e}```")

        @queue.autocomplete('song')
        async def get_url(ctx: discord.Interaction, song: str):
            result = []
            i = 0
            for f, song in ((path, re.sub(
                    r"(.*?)-[\-_a-zA-Z0-9]*\.\w{2,4}", r"\1", path.name))
                    for path in Path(self.general['music_dir']).glob('*.m4a')):
                if i >= 25:
                    break
                if song.lower() in f.name.lower():
                    result.append(app_commands.Choice(
                        name=song,
                        value=f.name))
                    i += 1
            return result

        @self.tree.command(
            name="chat",
            description="Join, create or leave a conversation with Animegen",
            **kwargs)
        async def chat(ctx: discord.Interaction):
            await self.add_user(ctx, ctx.user)

        @self.tree.command(
            name="achat",
            description="Join, create or leave a conversation with Animegen",
            **kwargs)
        @app_commands.default_permissions(manage_messages=True)
        async def chat_other(ctx: discord.Interaction, user: Optional[discord.Member]):
            await self.add_user(ctx, user if user is not None else ctx.user)

        @self.tree.command(
            name="whoschatting",
            description="List users in the current conversation with Animegen",
            **kwargs)
        async def whoschatting(interaction: discord.Interaction):
            users = ', '.join(map(
                lambda id: f'<@{id}>',
                self.chat_participants))
            await interaction.response.send_message(
                f'these users are currently chatting with me! :D {users}'
                if users else 'noone wants to talk with me at the moment :(',
                allowed_mentions=discord.AllowedMentions(users=False))

        @self.tree.command(
            name="imagine",
            description="Generate an image",
            **kwargs)
        @app_commands.describe(
            prompt="Tags for generating the image",
            negative_prompt="Penalty tags for generating the image",
            sampler="The sampler to use for generation",
            seed="Seed for generating the image",
            steps="Inference steps for generating the image",
            width="Custom width",
            height='Custom height',
            quality="The quality of image to generate",
            style="The style of image to generate",
            base_image="Image to base generation off of",
            base_image_strength="Strength of base image"
        )
        @app_commands.choices(
            sampler=[app_commands.Choice(name=s, value=s)
                     for s in self.params['samplers']],
            quality=[app_commands.Choice(name=s, value=s)
                     for s in self.params['quality_selectors']],
            style=[app_commands.Choice(name=s, value=s)
                   for s in self.params['style_selectors']]
        )
        async def imagine(
                interaction: discord.Interaction,
                prompt: str = self.defaults['prompt'],
                negative_prompt: str = self.defaults['negative_prompt'],
                sampler: app_commands.Choice[str] = self.defaults['sampler'],
                steps: int = self.defaults['steps'],
                width: int = int(
                    self.defaults['aspect_ratio'].split(" x ")[0]),
                height: int = int(
                    self.defaults['aspect_ratio'].split(" x ")[1]),
                quality: app_commands.Choice[str] = self.defaults['quality_selector'],
                style: app_commands.Choice[str] = self.defaults['style_selector'],
                seed: int = -1,
                base_image: Optional[discord.Attachment] = None,
                base_image_strength: float = 0.65):

            prompt = self.blacklist.replace(prompt, '')

            if sampler not in self.params['samplers']:
                sampler = self.defaults['sampler']

            if quality not in self.params['quality_selectors']:
                quality = self.defaults['quality_selector']

            if style not in self.params['style_selectors']:
                style = self.defaults['style_selector']

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
                'style_selector': style,
                'quality_selector': quality
            }

            if base_image is not None:
                kwargs['use_base_image'] = True
                kwargs['base_image'] = base_image
                kwargs['base_image_strength'] = base_image_strength

            try:
                embed_log = self.embed_image(
                    interaction.user.display_name,
                    prompt,
                    negative_prompt,
                    **kwargs)

                path = Path(await self.generate(
                    prompt,
                    negative_prompt,
                    **kwargs))

                with path.open('rb') as f:
                    await interaction.followup.send(
                        embed=embed_log,
                        file=discord.File(f, filename=path.name))

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
            description="Kick a user from the Discord server.",
            **kwargs)
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
            description="Ban a user from the Discord server.",
            **kwargs)
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
            description="Unban a user from the Discord server.",
            **kwargs)
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
            description="Mute a user's messages.",
            **kwargs)
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
            description="Unmute a user's messages.",
            **kwargs)
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
            description="Warn a user and add them to the watchlist.",
            **kwargs)
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
