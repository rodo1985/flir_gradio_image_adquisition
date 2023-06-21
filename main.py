import os
import cv2
import gradio as gr
from modules.SingleShotCamera import SingleShotCamera
import time
from flask import Flask, render_template
import threading
import datetime
import base64

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict


def predict_image_classification_sample(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5, max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))


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

    # filename
    filename = os.path.join('images', f"image_{date_time}.png")

    # Save the image with the date and time in the filename
    cv2.imwrite(filename, img)

    # image prediction
    predict_image_classification_sample(
    project="795715818331",
    endpoint_id="1477985520289054720",
    location="us-central1",
    filename=filename
    )

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
