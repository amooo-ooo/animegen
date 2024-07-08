import discord
from discord import app_commands
from discord.ext import commands, tasks
from gradio_client import Client, handle_file
from config import *
import random
import os

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
client = Client("Boboiazumi/animagine-xl-3.1")

async def generate(prompt: str = "freiren"):
    image, details = client.predict(
            prompt=prompt + ", " + PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            seed=random.randint(0, 2147483647),
            img_path=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
            api_name="/run"
    )

    path = image[0]["image"]
    return path

@bot.event
async def on_ready():
    print("Bot is cooking!!!")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(e)

@bot.tree.command(name="hello")
async def hello(interaction: discord.Interaction):
    await interaction.response.send_message(f"Hey, {interaction.user.mention}!")

@bot.tree.command(name="imagine")
@app_commands.describe(prompt = "Tags for generating the image")
async def imagine(interaction: discord.Interaction, 
                  prompt: str = "frieren"):

    log = f"{interaction.user.global_name} - {prompt}"
    print(log)

    with open("log.txt", "a") as f:
        f.write(log + "\n")
    await interaction.response.defer()

    try:
        path = await generate(prompt=prompt)
        with open(path, 'rb') as f:
            await interaction.followup.send(prompt, file=discord.File(f))
        os.remove(path)
    except Exception as e:
        await interaction.followup.send(f"> Quota was met for generating images. Go touch some grass for {':'.join(str(e).split(':')[2:])[:5]} minute(s) and come back!")
        print(e)

bot.run(TOKEN)