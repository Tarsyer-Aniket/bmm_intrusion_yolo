import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
from functools import partial

atm_class_names = [
            "person",
            "helmet",
            "face",
            "head"
]

pretrained_class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic_light",
            "fire_hydrant",
            "stop_sign",
            "parking_meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports_ball",
            "kite",
            "baseball_bat",
            "baseball_glove",
            "skateboard",
            "surfboard",
            "tennis_racket",
            "bottle",
            "wine_glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot_dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted_plant",
            "bed",
            "dining_table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell_phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy_bear",
            "hair_drier",
            "toothbrush",
        ]

class SSD_TfLite_Detection():
    def __init__(self, CONFIDENCE_THRESH=0.5, TFLITE_MODEL_PATH='models/pretrained.tflite', CLASSES=pretrained_class_names):

        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img = np.expand_dims(img, 0)
        print(TFLITE_MODEL_PATH)

        # tflite model init
        self._interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=2)
        self._interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        # inference helper
        self._set_input_tensor = partial(self._interpreter.set_tensor, input_details[0]['index'])
        self._get_boxes_tensor = partial(self._interpreter.get_tensor, output_details[0]['index'])
        self._get_classes_tensor = partial(self._interpreter.get_tensor, output_details[1]['index'])
        self._get_scores_tensor = partial(self._interpreter.get_tensor, output_details[2]['index'])

        self._confidence_thresh = CONFIDENCE_THRESH
        self.class_names = CLASSES
        if len(self.class_names) == 80:
            self._save_all_class = False
        else:
            self._save_all_class = True

        

    def _pre_processing(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        return img

    def inference(self, img):

        input_tensor = self._pre_processing(img)

        self._set_input_tensor(input_tensor)

        self._interpreter.invoke()

        # get results
        bboxes = self._get_boxes_tensor()[0]
        classes = self._get_classes_tensor()[0]
        scores = self._get_scores_tensor()[0]

        bbox_lists, class_lists, score_lists = self._post_processing(bboxes, classes, scores)

        if len(bbox_lists) > 0:
            bbox_lists *= np.tile(img.shape[1::-1], 2)
            bbox_lists = bbox_lists.astype(int)

        return bbox_lists, class_lists, score_lists


    def _post_processing(self, bboxes, classes, scores):

        bbox_lists = []
        score_lists = []
        class_lists = []
        save_bbox_score = False
        for i in range(len(scores)):                # for each detection
            if scores[i] > self._confidence_thresh and scores[i] <= 1 :      # check if that detection score passes some threshold


                if self._save_all_class:
                    class_lists.append(int(classes[i]))
                    save_bbox_score = True
                else:
                    if int(classes[i]) == 0: # person - 0
                        class_lists.append(int(classes[i]))
                        save_bbox_score = True
                        

                if save_bbox_score:

                    ymin = float(bboxes[i][0])
                    xmin = float(bboxes[i][1])
                    ymax = float(bboxes[i][2])
                    xmax = float(bboxes[i][3])
                    
                    bbox_lists.append([xmin, ymin, xmax, ymax])
                    
                    score_lists.append(scores[i])
                    save_bbox_score = False
            elif scores[i] > 1:
                print('check')
                print(scores[i])

                


        return bbox_lists, class_lists, score_lists
