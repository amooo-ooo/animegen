# py -m bot
import os
from pathlib import Path

import discord
from dotenv import load_dotenv
from . import Animegen


load_dotenv()
token = os.getenv('TOKEN')
assert token is not None
bot = Animegen(
    Path(__file__).parent,
    command_prefix="!",
    intents=discord.Intents.all())
bot.run(token)
