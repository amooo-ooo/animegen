from gradio_client import Client, handle_file
import random
client = Client("Boboiazumi/animagine-xl-3.1")
result = client.predict(
		prompt="frieren, masterpiece, best quality, very aesthetic, absurdres",
		negative_prompt="nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
		seed=random.randint(0, 2147483647),
		custom_width=1024,
		custom_height=1024,
		guidance_scale=7,
		num_inference_steps=28,
		sampler="Euler a",
		aspect_ratio_selector="896 x 1152",
		style_selector="(None)",
		quality_selector="Standard v3.1",
		use_upscaler=False,
		upscaler_strength=0.55,
		upscale_by=1.5,
		add_quality_tags=True,
		isImg2Img=False,
		img_path=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
		img2img_strength=0.65,
		api_name="/run"
)
print(result)