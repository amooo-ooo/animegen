from gradio_client import Client

client = Client("cagliostrolab/animagine-xl-3.1")
result = client.predict(
        "frieren, masterpiece, best quality, very aesthetic, absurdres",	# str  in 'Prompt' Textbox component
        "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",	# str  in 'Negative Prompt' Textbox component
        0,	# float (numeric value between 0 and 2147483647) in 'Seed' Slider component
        512,	# float (numeric value between 512 and 2048) in 'Width' Slider component
        512,	# float (numeric value between 512 and 2048) in 'Height' Slider component
        1,	# float (numeric value between 1 and 12) in 'Guidance scale' Slider component
        28,	# float (numeric value between 1 and 50) in 'Number of inference steps' Slider component
        "Euler a",	# Literal['DPM++ 2M Karras', 'DPM++ SDE Karras', 'DPM++ 2M SDE Karras', 'Euler', 'Euler a', 'DDIM']  in 'Sampler' Dropdown component
        "1024 x 1024",	# Literal['1024 x 1024', '1152 x 896', '896 x 1152', '1216 x 832', '832 x 1216', '1344 x 768', '768 x 1344', '1536 x 640', '640 x 1536', 'Custom']  in 'Aspect Ratio' Radio component
        "(None)",	# Literal['(None)', 'Cinematic', 'Photographic', 'Anime', 'Manga', 'Digital Art', 'Pixel art', 'Fantasy art', 'Neonpunk', '3D Model']  in 'Style Preset' Radio component
        "(None)",	# Literal['(None)', 'Standard v3.0', 'Standard v3.1', 'Light v3.1', 'Heavy v3.1']  in 'Quality Tags Presets' Dropdown component
        False,	# bool  in 'Use Upscaler' Checkbox component
        0.5,	# float (numeric value between 0 and 1) in 'Strength' Slider component
        1,	# float (numeric value between 1 and 1.5) in 'Upscale by' Slider component
        True,	# bool  in 'Add Quality Tags' Checkbox component
        api_name="/run"
)
print(result)