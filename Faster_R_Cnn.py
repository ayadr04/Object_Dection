import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

class Detector:
    """
    Object Detection using TensorFlow Hub Faster R-CNN ResNet50.
    """

# Initialisation
    def __init__(self, model_url='https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1',# depuis TensorFlow Hub
                 coco_labels_path='.\coco.names', score_threshold=0.5):
        self.load_model(model_url)
        self.load_coco_labels(coco_labels_path)
        self.set_detection_params(score_threshold) # pour filtrer les prédictions faibles
    

# Chargement du modèle 
    def load_model(self, model_url):
        print(f'--> Loading model from TensorFlow Hub: {model_url}')
        self.detector = hub.load(model_url)
        print('--> Model loaded successfully')

# Chargement des étiquettes (classes)
    def load_coco_labels(self, coco_labels_path):
        with open(coco_labels_path, 'rt') as f:
            self.labels = f.read().strip().split('\n')
        print(f'--> Loaded COCO labels from {coco_labels_path}')
#  stocker le seuil de confiance minimal
    def set_detection_params(self, score_threshold):
        self.score_threshold = score_threshold
# Détection sur image
    def run_detection_on_img(self, img):
        input_tensor = tf.convert_to_tensor(img)[tf.newaxis, ...]
        detections = self.detector(input_tensor)

        boxes = detections['detection_boxes'][0].numpy()
        class_ids = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()

        h, w, _ = img.shape
        results = []

        for i in range(len(scores)):
            if scores[i] < self.score_threshold:
                continue

            y1, x1, y2, x2 = boxes[i]
            x, y, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            box_w, box_h = x2 - x, y2 - y

            label = self.labels[class_ids[i] - 1] if 0 < class_ids[i] <= len(self.labels) else "N/A"
            confidence = f"{int(scores[i] * 100)}%"

            results.append({
                'label': label,
                'confidence': confidence,
                'bbox_xywh': [x, y, box_w, box_h]
            })

        return results

# Affichage des résultats
    def draw_detections(self, img, detections):
        for det in detections:
            x, y, w, h = det['bbox_xywh']
            label = det['label']
            conf = det['confidence']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, f'{label} {conf}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return img
