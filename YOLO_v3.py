import cv2
import numpy as np

class Detector:
    """
    Class for performing object detection using YOLOv3 model.
    """

    def __init__(self, config_path='./yolov3.cfg', weights_path='./yolov3.weights',
                 coco_names_path='./coco.names', score_threshold=0.5, NMS_threshold=0.5):
        self.load_model(config_path, weights_path)
        self.load_coco_labels(coco_names_path)
        self.set_detection_params(score_threshold, NMS_threshold)

    def load_model(self, config_path, weights_path):
        net = cv2.dnn.readNet(config_path, weights_path)
        self.net = net
        print(f'* Model config loaded from {config_path}\n* Model weights loaded from {weights_path}')

    def load_coco_labels(self, coco_names_path):
        with open(coco_names_path, 'rt') as coco_file:
            self.labels = coco_file.read().rstrip('\n').rsplit('\n')
        print(f'* COCO labels loaded from {coco_names_path}')

    def preprocess_img(self, img, size=(320, 320)):
        self.resize = size
        blob = cv2.dnn.blobFromImage(img, 1 / 255, size, [0, 0, 0], 1, crop=False)
        return blob

    def set_detection_params(self, score_threshold, NMS_threshold):
        self.score_threshold = score_threshold
        self.NMS_threshold = NMS_threshold

    def run_detection_on_img(self, img):
        blob = self.preprocess_img(img)
        self.net.setInput(blob)
        layers_names = self.net.getLayerNames()
        output_names = [layers_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        outputs = self.net.forward(output_names)

        hT, wT, _ = img.shape
        bbox = []
        class_ids = []
        confs = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.score_threshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    class_ids.append(class_id)
                    confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox, confs, self.score_threshold, self.NMS_threshold)

        results_list = []
        for i in indices.flatten():
            result_dict = {}
            box = bbox[i]
            class_id = class_ids[i]
            conf = int(confs[i] * 100)
            result_dict['label'] = self.labels[class_id]
            result_dict['confidence'] = str(conf) + '%'
            result_dict['bbox_xywh'] = box
            results_list.append(result_dict)

        return results_list

    def draw_detections(self, img, detections):
        for det in detections:
            x, y, w, h = det['bbox_xywh']
            label = det['label']
            conf = det['confidence']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, f'{label} {conf}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 2)
        return img
