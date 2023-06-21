import os
import cv2
import gradio as gr
from modules.SingleShotCamera import SingleShotCamera
import time
from flask import Flask, render_template
import threading
import datetime
import numpy as np
from keras.models import load_model
import tensorflow as tf


def trigger_camera():

    # memorized time
    start = time.time()

    # Capture an image
    img = camera.capture_image()

    # gett elapsed time in ms
    acquistion_time = (time.time() - start)*1000

    # memorized time
    start = time.time()

    # Top left corner of the crop box
    start_y, start_x = 800, 950

    # Bottom right corner of the crop box
    end_y, end_x = 1300, 1450

    # crop image
    img = img[start_y:end_y, start_x:end_x]

    # resize
    img = cv2.resize(img, (224, 224))

    # Normalize the image
    normalized_image_array = (img.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)

    # get elapsed time in ms
    process_time = (time.time() - start)*1000

    total_time = acquistion_time + process_time

    # Create the directory if it does not exist
    if not os.path.exists('images'):
        os.makedirs('images')

    # creates date time
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # filename
    filename = os.path.join('images', f"image_{date_time}.png")

    # Save the image with the date and time in the filename
    cv2.imwrite(filename, img)

    # returns values
    return img, {"Class A": float(prediction[0][0]), "Class B": float(prediction[0][1])}, "{:.2f} ms".format(acquistion_time), "{:.2f} ms".format(process_time), "{:.2f} ms".format(total_time)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("models/classify/keras_model.h5", compile=False)

# Load the labels
class_names = open("models/classify/labels.txt", "r").readlines()


# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Create a camera object
camera = SingleShotCamera()

css = "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:

    with gr.Row():
        with gr.Column():
            with gr.Row():
                image_viewer = gr.Image().style(height=500)
            with gr.Row():
                trigger_button = gr.Button(label="Trigger Camera")
        with gr.Column():
            label_viewer = gr.Label(num_top_classes=2, label='Classification')
            acquistion_time_texbox = gr.Textbox(label="Acquistion time")
            process_time_texbox = gr.Textbox(label="Process time")
            total_time_texbox = gr.Textbox(label="Total time")

        trigger_button.click(trigger_camera, outputs=[
                             image_viewer, label_viewer, acquistion_time_texbox, process_time_texbox, total_time_texbox])


# Run Gradio interface in a separate thread
threading.Thread(target=demo.launch).start()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=5000)
