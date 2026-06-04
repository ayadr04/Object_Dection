import cv2
import numpy as np
import tensorflow as tf

class Detector:
    def __init__(self, model_path="model_V2.keras", class_names_path='voc.names', input_size=(224, 224)):
        """
        Initializes the detector with your custom Keras model.
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.input_size = input_size
        self.class_names = self.load_class_names(class_names_path)
        print(f'* Model loaded from {model_path}')
        print(f'* Class names loaded from {class_names_path}')

    def load_class_names(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def preprocess_image(self, img):
        img_resized = cv2.resize(img, self.input_size)
        img_normalized = img_resized / 255.0
        return np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    def run_detection_on_img(self, img):
        """
        Runs the model on the image and returns the prediction.
        Assumes 1 object per image.
        """
        input_tensor = self.preprocess_image(img)
        class_preds, bbox_preds = self.model.predict(input_tensor, verbose=0)

        class_id = np.argmax(class_preds[0])
        confidence = class_preds[0][class_id]
        bbox = bbox_preds[0]  # [xmin, ymin, xmax, ymax] (normalized)

        # Convert normalized bbox to original image scale
        h, w, _ = img.shape
        xmin = int(bbox[0] * w)
        ymin = int(bbox[1] * h)
        xmax = int(bbox[2] * w)
        ymax = int(bbox[3] * h)

        result = {
            'label': self.class_names[class_id],
            'confidence': f'{confidence*100:.2f}%',
            'bbox_xywh': [xmin, ymin, xmax - xmin, ymax - ymin]
        }
        return [result]

    def draw_detections(self, img, detections):
        for det in detections:
            x, y, w, h = det['bbox_xywh']
            label = det['label']
            conf = det['confidence']

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {conf}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        return img
