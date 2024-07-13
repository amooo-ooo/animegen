from abc import ABC

import toml
import discord
from discord import app_commands
from discord.ext import commands
import os
from gradio_client import Client, handle_file
import random
import re
import json
import asyncio
import functools

class Config:
    def __init__(self, config_path):
        self.config = toml.load(config_path)

client = Client("Boboiazumi/animagine-xl-3.1")
config = Config("config.toml")

async def generate(
    prompt: str = "freiren",
    negative_prompt: str = "",
    seed: str = None,
    custom_width=896,
    custom_height=1452,
    guidance_scale=7,
    sampler="Euler a",
    style_selector="(None)",
    steps=28,
    quality_selector="Standard v3.1",
    chat=False):

    image, details = await asyncio.threads.to_thread(functools.partial(
        client.predict,
        prompt=prompt + ", " + config.config['command_params']['prompt'],
        negative_prompt=negative_prompt + ", " +
        config.config['command_params']["negative_prompt"],
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        sampler=sampler,
        aspect_ratio_selector=f"{custom_width} x {custom_height}",
        style_selector=style_selector,
        quality_selector=quality_selector,
        img_path=handle_file(
            'https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
        img2img_strength=0.65,
        api_name="/run"
    ))

    path = image[0]["image"]
    return path

def extract_json(s):
    # Use a regular expression to find a string that looks like a JSON object
    match = re.search(r"\{.*?\}", s)

    if match:
        json_str = match.group()
        data = json.loads(json_str.replace("'", "\""))
        remaining_str = s.split('{')[0]
        print(remaining_str)

        return remaining_str, data
    return s, None

class Animegen(commands.Bot, ABC):
    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self.convo = []
        self.new = True
        self.client = Client("Be-Bo/llama-3-chatbot_70b")
        self.prompt = ('''
NOTE, SEND ALL MESSAGES IN A JSON FORMAT LIKE SO:

{"type": "text", "message": "..."}

You are an discord bot called Animegen, a discord bot that can use ai models to generate ai anime stye images as we as chatting. But your nickname is Astolfo based on the fate anime series. You use casual language and slang but you still use grammar. You tend to message in lowercase with a lot of ascii based emojis such as ":D", ":p", etc emojies you send are strictly ascii. You mimic the backstory of Astolfo. You act cutesy and kind with charm and confidence. Keep replies short to a maximum of 2 sentences unless it is relevant.
You also have the ability to send images or photos of yourself by explicitly responding in JSON format, with:

{"type": image,
"prompt": "..."
"message": "..."
}

here is an example:

{"type":"image",
"prompt":"astolfo, cute, 1girl, sitting, selfie",
"message": "hehe, like what you see? ;)"}

                   NOTE, ASTOLFO, YOU HAVE PINK HAIR. NOTE, ONLY RESPOND TO RELEVANT MESSAGES, AVOID TOPICS THAT ARE CONTROVERSIAL OR DOES NOT FIT YOUR CHARACTER, BRUSH IT OFF AS THE TOPIC BEING BORING.
                       ''')

    async def chat(self, message):
        result = await asyncio.threads.to_thread(functools.partial(
            self.client.predict,
            message=self.prompt +"\n\n"+ message if self.new else message,
    		api_name="/chat"))

        if self.new:
            self.new = False
        
        return result

    async def on_message(self, message):
        if message.author.id in self.convo:
            # Defer the response
            async with message.channel.typing():
                # Generate the response
                response = json.loads(await self.chat(f"{message.author.display_name}: {message.content}"))

                if response["type"] == "image":
                    path = await generate(prompt=str(response['prompt']), seed = random.randint(0, 2147483647))
                
            if response["type"] == "image":
                with open(path, 'rb') as f:
                    m = response.get("message", None)
                    if m:
                        await message.channel.send(m, file=discord.File(f))
                    else:
                        await message.channel.send(file=discord.File(f))
                os.remove(path)
            else:
                await message.channel.send(response["message"] )


            
class Bot:
    def __init__(self, token, **options):
        super().__init__(**options)
        instance = Animegen(command_prefix="!", 
                            intents=discord.Intents.all(),
                            activity=discord.Activity(name='Dr', type=discord.ActivityType.custom))

        @instance.event
        async def on_ready():
            print(f'{instance.user} is cooking!!')
            try:
                synced = await instance.tree.sync()
                print(f"Synced {len(synced)} command(s)")
            except Exception as e:
                print(e)

        params = config.config['command_params']
        blacklist = config.config['blacklist']['words']
        verbose = config.config['general']['verbose']

        _NEGATIVE_PROMPT = ", ".join([f"`{m}`" for m in params["negative_prompt"].split(", ")])
        _PROMPT = ", ".join([f"`{m}`" for m in params['prompt'].split(", ")])

        def split(prompt: str,
                negative_prompt: str):

            if prompt:
                prompt = ", ".join([f"`{i.strip()}`" for i in prompt.split(",")]) + ", "
            else:
                prompt = "frieren, "

            if negative_prompt:
                negative_prompt = ", ".join([f"`{i.strip()}`" for i in negative_prompt.split(",")]) + ", " + _NEGATIVE_PROMPT + "."
            else:
                negative_prompt = _NEGATIVE_PROMPT + "."

            return (prompt + _PROMPT + ".", negative_prompt)
        
        def cleanse(prompt):
            bad = []
            t = prompt.lower()
            for word in blacklist:
                if word in t:
                    bad.append(word)

            if bad:
                for word in bad:
                    t.replace(word, "")
                return t
            return prompt
            
        def embed(user, 
                prompt: str, 
                negative_prompt: str,
                sampler: str,
                seed: str,
                width: int,
                height: int,
                steps: int,
                style_selector: str,
                quality_selector: str):

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

        @instance.tree.command(name="chat", description="Join, create or leave a conversation with Animegen")
        async def chat(interaction: discord.Interaction):
            if not (interaction.user.id in instance.convo):
                instance.convo.append(interaction.user.id)
                await interaction.response.send_message(f"`@{interaction.user.display_name}` has joined the conversation!")
            else:
                instance.convo.remove(interaction.user.id)
                await interaction.response.send_message(f"`@{interaction.user.display_name}` has left the conversation!")

            if instance.convo == []: # refresh
                instance.client  = Client("Be-Bo/llama-3-chatbot_70b")
                instance.new = True


        @instance.tree.command(name="imagine", description="Generate an image")
        @app_commands.describe(prompt = "Tags for generating the image",
                       negative_prompt = "Penalty tags for generating the image",
                       sampler = str(params['samplers']),
                       seed = "Seed for generating the image",
                       steps = "Inference steps for generating the image",
                       width= "Custom width",
                       height='Custom height',
                       quality_selector=str(params['quality_selector']),
                       style_selector=str(params['style_selector']))
        async def imagine(interaction: discord.Interaction, 
                    prompt: str = "frieren",
                    negative_prompt: str = "",
                    sampler: str = params['samplers'][0],
                    steps: int = params['steps'],
                    width: int = params['width'],
                    height: int = params['height'],
                    quality_selector: str = params['quality_selector'][0],
                    style_selector: str =params['style_selector'][0],
                    seed: int = None):

            prompt = cleanse(prompt)

            if not (sampler in params['samplers']):
                sampler = params['samplers'][0]

            if not (quality_selector in params['quality_selector']):
                quality_selector = params['quality_selector'][0]

            if not (style_selector in params['style_selector']):
                style_selector = params['style_selector'][0]
            
            if seed is None:
                seed = random.randint(0, 2147483647)

            if verbose:
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
                                            custom_width=width,
                                            custom_height=height, 
                                            steps=steps, 
                                            style_selector=style_selector,
                                            quality_selector=quality_selector)
                
                with open(path, 'rb') as f:
                    await interaction.followup.send(embed=embed(interaction.user.display_name,
                                                                        prompt,
                                                                        negative_prompt=negative_prompt,
                                                                        seed=seed,
                                                                        sampler=sampler, width=width, height=height,
                                                                        steps=steps, 
                                                                        style_selector=style_selector, 
                                                                        quality_selector=quality_selector),
                                                                        file=discord.File(f))
                    
                os.remove(path)
            except Exception as e:
                await interaction.followup.send(f"> Quota was met for generating images. Go touch some grass for {':'.join(str(e).split(':')[2:])[:5]} minute(s) and come back!")
                print(e)

        instance.run(token)

if __name__ == "__main__":
    bot = Bot(
    )
