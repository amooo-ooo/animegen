from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
from aiortc import RTCPeerConnection, RTCSessionDescription
import cv2
import uuid
import asyncio
import logging
import time
import numpy as np
from mss import mss

app = Flask(__name__, static_url_path='/static')

# Set to keep track of RTCPeerConnection instances
pcs = set()

# Function to generate video frames from the camera
def generate_camera_frames():
    camera = cv2.VideoCapture(0)
    while True:
        start_time = time.time()
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Concatenate frame and yield for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
            elapsed_time = time.time() - start_time
            logging.debug(f"Frame generation time: {elapsed_time} seconds")

# Function to generate video frames from the screen
def generate_frames():
    sct = mss()
    while True:
        start_time = time.time()
        # Capture the screen
        screenshot = sct.grab(sct.monitors[0])
        # Convert the screenshot to a numpy array and then to color
        img = np.array(screenshot)
        frame = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # Concatenate frame and yield for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
        elapsed_time = time.time() - start_time
        logging.debug(f"Frame generation time: {elapsed_time} seconds")

# Route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Asynchronous function to handle offer exchange
async def offer_async():
    params = await request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create an RTCPeerConnection instance
    pc = RTCPeerConnection()

    # Generate a unique ID for the RTCPeerConnection
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pc_id = pc_id[:8]

    # Create a data channel named "chat"
    # pc.createDataChannel("chat")

    # Create and set the local description
    await pc.createOffer(offer)
    await pc.setLocalDescription(offer)

    # Prepare the response data with local SDP and type
    response_data = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    return jsonify(response_data)

# Wrapper function for running the asynchronous offer function
def offer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    future = asyncio.run_coroutine_threadsafe(offer_async(), loop)
    return future.result()

# Route to handle the offer request
@app.route('/offer', methods=['POST'])
def offer_route():
    return offer()

# Route to stream video frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(debug=True, host='localhost', ssl_context='adhoc', use_reloader=False)

# Run the Flask app
if __name__ == "__main__":
    run_flask()