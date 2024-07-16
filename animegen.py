import discord
from bot import Animegen
from activity.app import run_flask
from dotenv import load_dotenv
import threading
import os

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    load_dotenv()
    token = os.getenv('TOKEN')
    assert token is not None
    bot = Animegen(command_prefix="!", intents=discord.Intents.all())
    bot.run(token)