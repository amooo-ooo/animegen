# py animegen.py
import os
from pathlib import Path
import threading

import discord
from dotenv import load_dotenv

from activity.app import run_flask
from bot import Animegen

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    load_dotenv()
    token = os.getenv('TOKEN')
    assert token is not None
    bot = Animegen(
        Path('./bot'),
        command_prefix="!",
        intents=discord.Intents.all())
    bot.run(token)
