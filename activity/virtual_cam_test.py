# will be hosted online 

from pathlib import Path
import cv2
import pyvirtualcam

image_path = Path(Path(__file__).parent, 'assets', "peak.png")
image = cv2.imread(str(image_path))

if image is None:
    print(f'Failed to load image at {image_path}')
else:
    print(f'Successfully loaded image at {image_path}')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (1280, 720))

# Start the virtual camera
with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
    print(f'Using virtual camera: {cam.device}')

    while True:
        cam.send(image)
        cam.sleep_until_next_frame()
