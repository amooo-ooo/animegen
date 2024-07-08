import discord
from discord import app_commands
from discord.ext import commands, tasks
from gradio_client import Client, handle_file
from config import *
import random
import os

_NEGATIVE_PROMPT = ", ".join([f"`{m}`" for m in NEGATIVE_PROMPT.split(", ")])
_PROMPT = ", ".join([f"`{m}`" for m in PROMPT.split(", ")])

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
client = Client("Boboiazumi/animagine-xl-3.1")

async def generate(prompt: str = "freiren",
            negative_prompt: str = "",
            seed: str = None,
            custom_width=1024,
            custom_height=1024,
            guidance_scale=7,
            num_inference_steps=28,
            sampler="Euler a",
            style_selector="(None)",
            quality_selector="Standard v3.1"):

    image, details = client.predict(
            prompt=prompt + PROMPT,
            negative_prompt=negative_prompt + NEGATIVE_PROMPT,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            sampler=sampler,
            aspect_ratio_selector=f"{custom_width} x {custom_height}",
            style_selector=style_selector,
            quality_selector=quality_selector,
            img_path=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
            img2img_strength=0.65,
            api_name="/run"
    )

    path = image[0]["image"]
    return path


def split(prompt: str,
          negative_prompt: str):

    if prompt:
        prompt = ", ".join([f"`{i.strip()}`" for i in prompt.split(",")]) + " "
    else:
        prompt = _PROMPT

    if negative_prompt:
        negative_prompt = ", ".join([f"`{i.strip()}`" for i in negative_prompt.split(",")]) + " "

    return (prompt + ".", negative_prompt + ", " + _NEGATIVE_PROMPT + ".")

def embed(user, 
          prompt: str, 
          negative_prompt: str,
          sampler: str = SAMPLER,
          seed: str = None,
          width: int = 896,
          height: int = 1152,
          steps: str = STEPS,
          style_selector: str = STLYE_SELECTOR,
          quality_selector: str = QUALITY_SELECTOR):

    if seed is None:
        seed = random.randint(0, 2147483647)

    embedVar = discord.Embed(title=f"Generating for `@{user}`!", 
                             description="Powered by `Boboiazumi/animagine-xl-3.1` on Huggingface Spaces with Gradio.", 
                             color=0xf4e1cc)
    
    prompt, negative_prompt = split(prompt, negative_prompt)

    embedVar.add_field(name="Prompt", value=prompt, inline=False)
    embedVar.add_field(name="Negative Prompt", value=negative_prompt, inline=False)

    embedVar.add_field(name="Sampler", value=sampler, inline=True)
    embedVar.add_field(name="Seed", value=seed, inline=True)
    embedVar.add_field(name="Inference Steps", value=steps, inline=True)

    embedVar.add_field(name="Resolution", value=f"{width} x {height} ", inline=True)
    embedVar.add_field(name="Quality Selector", value=quality_selector, inline=True)
    embedVar.add_field(name="Style Selector", value=style_selector, inline=True)

    #embedVar.set_footer(text=f"You're currenty position {len(queue)} in the queue!")
    return embedVar

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
@app_commands.describe(prompt = "Tags for generating the image",
                       negative_prompt = "Penalty tags for generating the image",
                       sampler = str(SAMPLERS),
                       seed = "Seed for generating the image",
                       steps = "Inference steps for generating the image",
                       width= "Custom width",
                       height='Custom height')
async def imagine(interaction: discord.Interaction, 
                  prompt: str = "frieren",
                  negative_prompt: str = "",
                  sampler: str = SAMPLER,
                  seed: int = None,
                  steps: int = 28,
                  width: int = 896,
                  height: int = 1152):

    if not (sampler in SAMPLERS):
        sampler = SAMPLER
    
    if seed is None:
        seed = random.randint(0, 2147483647)

    log = f"{interaction.user.global_name} - {prompt}"
    print(log)

    with open("log.txt", "a") as f:
        f.write(log + "\n")

    await interaction.response.defer()

    # must be a cleaner way to do this
    try:
        path = await generate(prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    sampler=sampler,
                                    seed=seed,
                                    num_inference_steps=steps,
                                    custom_width=width,
                                    custom_height=height)
        
        with open(path, 'rb') as f:
            await interaction.followup.send(embed=embed(interaction.user.display_name,
                                                                prompt,
                                                                negative_prompt=negative_prompt,
                                                                seed=seed,
                                                                sampler=sampler, width=width, height=height),
                                                                file=discord.File(f))
            
        os.remove(path)
    except Exception as e:
        await interaction.followup.send(f"> Quota was met for generating images. Go touch some grass for {':'.join(str(e).split(':')[2:])[:5]} minute(s) and come back!")
        print(e)


bot.run(TOKEN)