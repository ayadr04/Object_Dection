from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import uvicorn

import numpy as np
from io import BytesIO
import cv2

# You can choose what model you want to use

#from CNN import Detector
from Faster_R_Cnn import Detector
#from YOLO_v3 import Detector


app = FastAPI()
detector = Detector()


def read_image(file) -> np.ndarray:
    """Reads and decodes an image from an uploaded file."""
    image_stream = BytesIO(file)
    image_stream.seek(0)
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    return image


@app.post("/detection/")
async def detect_on_img(file: UploadFile = File(...)):
    # Read image
    img = read_image(await file.read())

    # Run detection
    results_list = detector.run_detection_on_img(img)

    # Draw results on image
    img_with_boxes = detector.draw_detections(img, results_list)

    # Encode image as JPEG
    _, img_encoded = cv2.imencode('.jpg', img_with_boxes)
    img_bytes = BytesIO(img_encoded.tobytes())

    # Return image with bounding boxes
    return StreamingResponse(img_bytes, media_type="image/jpeg")


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
