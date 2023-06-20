import os
import cv2
import gradio as gr
from modules.SingleShotCamera import SingleShotCamera
import time
from flask import Flask, render_template
import threading
import datetime


def trigger_camera():

    start = time.time()

    # Capture an image
    img = camera.capture_image()

    # Top left corner of the crop box
    start_y, start_x = 800, 950

    # Bottom right corner of the crop box
    end_y, end_x = 1300, 1450

    img = img[start_y:end_y, start_x:end_x]

    # print elapsed time in ms
    print("Elapsed time: {:.2f} ms".format((time.time() - start)*1000))

    # Create the directory if it does not exist
    if not os.path.exists('images'):
        os.makedirs('images')

    # creates date time
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the image with the date and time in the filename
    cv2.imwrite(os.path.join('images', f"image_{date_time}.png"), img)

    return img


# Create a camera object
camera = SingleShotCamera()

css = "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"

with gr.Blocks(css=css) as demo:

    with gr.Box():
        # Crear la interfaz de Gradio
        # specify the shape, height and width as needed
        image_viewer = gr.Image(shape=(100, 100), width=100, height=100)
        trigger_button = gr.Button(label="Trigger Camera")
        trigger_button.click(trigger_camera, outputs=image_viewer)

# Run Gradio interface in a separate thread
threading.Thread(target=demo.launch).start()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=5000)
