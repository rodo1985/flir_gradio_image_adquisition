import os, base64
from io import BytesIO
import cv2
import warnings

import numpy as np
warnings.filterwarnings("ignore")
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials/credentials.json"

class GCP_Inference:
    def __init__(self, project: str, endpoint_id: str, location: str = "us-central1",
                 api_endpoint: str = "us-central1-aiplatform.googleapis.com"):
        self.client_options = {"api_endpoint": api_endpoint}
        self.client = aiplatform.gapic.PredictionServiceClient(client_options=self.client_options)
        self.endpoint = self.client.endpoint_path(
            project=project, location=location, endpoint=endpoint_id
        )
        self.parameters = predict.params.ImageObjectDetectionPredictionParams(
            confidence_threshold=0.5, max_predictions=5,
        ).to_value()
    
    def preproces(self, img):
    
        # Save the image to a BytesIO object
        _, im_buf_arr = cv2.imencode(".png", img)
        im_file = BytesIO(im_buf_arr)

        # The format of each instance should conform to the deployed model's prediction input schema.
        encoded_content = base64.b64encode(im_file.getvalue()).decode("utf-8")

        return encoded_content

    def run(self, encoded_content):

        
        instance = predict.instance.ImageObjectDetectionPredictionInstance(
            content=encoded_content,
        ).to_value()
        instances = [instance]

        response = self.client.predict(
            endpoint=self.endpoint, instances=instances, parameters=self.parameters
        )

        predictions = response.predictions
        formatted_preds = [dict(pred) for pred in predictions]

        # empty results array
        results = np.zeros(2)

        if formatted_preds[0]['displayNames'][0] == 'A':
            results[0] = float(formatted_preds[0]['confidences'][0])
            results[1] = 1 - float(formatted_preds[0]['confidences'][0])
        else:
            results[1] = float(formatted_preds[0]['confidences'][0])
            results[0] = 1 - float(formatted_preds[0]['confidences'][0])

        # Return formatted results
        return np.array([results])