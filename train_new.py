# This train_new.py is used for retrain the YOLO model for your own dataset.
# and refreed to KUASWoodyLIN's implementation, you can run this tool when you change  your own
# DATA_PATH
#
# 1、Generate your own annotation file and class names file，
#    python voc_annotation.py
#    python coco_annotation.py
#
# 2、load .h5 file to initalize your model
#
# 3、start to train your own model, and compute mAP for your model


import os
import argparse
import numpy as np
import math
import colorsys
import json
import shutil
from collections import defaultdict
from glob import glob
from matplotlib.pyplot import cm
from PIL import Image
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss, yolo_eval_v2, yolo_eval
from yolo3.utils import get_random_data, letterbox_image

#you should write your own train_url
DATA_PATH = 'tmp/coco/'

parser = argparse.ArgumentParser(description='Yolo v3 Keras base on TensorFlow implementation.')
parser.add_argument('--classes_file', type=str, help='The .txt file include dataset <classes>',
                    default='model_data/coco_classes.txt')
parser.add_argument('--anchors_file', type=str, help='The .txt file include yolo anchors type',
                    default='model_data/yolo_anchors.txt')
# Train File
parser.add_argument('--rodnet_train_file', type=str, help='The .txt file include <img path>, <bbox>, <class>',
                    # default=DATA_PATH + '/dataset/train.txt'
                    default=os.path.join(DATA_PATH, 'coco_train_2014.txt'))
# Evaluate file
parser.add_argument('--rodnet_val_file', type=str, help='The .txt file include <img path>, <bbox>, <class>',
                    # default=DATA_PATH + '/dataset/val.txt'
                    default=os.path.join(DATA_PATH, 'coco_val_2014.txt'))
# The images you what to save in tensorboard
parser.add_argument('--rodnet_eval_attention_images', type=str, help='Path with .jpg file, ',
                    # default=DATA_PATH + '/dataset/val/sample_100_images'
                    default='tmp/coco/train2014')

args = parser.parse_args()

LOGS_PATH = 'rodnet_logs/'
MODELS_PATH = os.path.join(LOGS_PATH, 'models')
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)


class Yolo(object):
    def __init__(self):

        self.model_path = 'model_data/yolo_weights.h5'
        # Training set path
        self.train_annotation_path = args.rodnet_train_file

        # Validation set path
        self.val_annotation_path = args.rodnet_val_file

        # Highlight images
        self.eval_save_images_id = [img_name.split('.')[0] for img_name in
                                    os.listdir(args.rodnet_eval_attention_images)]

        # Detecter setting
        self.classes_path = args.classes_file
        self.anchors_path = args.anchors_file
        self.class_names = self.get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        self.anchors = self.get_anchors(self.anchors_path)
        self.input_shape = (416, 416)  # multiple of 32, hw
        self.shape = (416, 416, 3)
        self.gpu_num = 2

        # training batch size
        self.step1_batch_size = 32
        self.step2_batch_size = 8  # note that more GPU memory is required after unfreezing the body

        # PIL setting
        self.image_size = (1280, 720, 3)
        self.colors = np.array(cm.hsv(np.linspace(0, 1, self.num_classes)).tolist()) * 255

        # mAP setting
        self.min_overlap = 0.5
        self.gt_counter_per_class = defaultdict(int)  # dictionary with counter per class

        # Temp file path
        self.tmp_gt_files_path = "tmp_gt_files"
        self.tmp_pred_files_path = "tmp_pred_files"
        if not os.path.exists(self.tmp_gt_files_path):
            os.mkdir(self.tmp_gt_files_path)
            self.train_data, self.val_data, self.val_images = self.read_txt_file()
            shutil.copytree(self.tmp_gt_files_path, self.tmp_gt_files_path + '_org')
        if not os.path.exists(self.tmp_pred_files_path):
            os.mkdir(self.tmp_pred_files_path)

        self.sess = K.get_session()
        # Evaluate setting parameter
        self.score = 0.3
        self.iou = 0.45
        self.input_image_shape = K.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes, self.eval_inputs = self.generate()
        # self.boxes, self.scores, self.classes, self.eval_inputs = yolo_eval(self.yolo_model.output_shape, self.anchors,
        #                                                                        len(self.class_names), self.input_image_shape,
        #                                                                        score_threshold=self.score, iou_threshold=self.iou)
        self.model = self.create_model(yolo_weights_path=self.model_path)

        # Create tensorboard logger
        self.callback = TensorBoard(LOGS_PATH)
        self.callback.set_model(self.yolo_model)

    def generate(self):
        # import pdb
        # pdb.set_trace()
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes, self.yolo_model.output

    def create_model(self, load_pretrained=True, freeze_body=2,
                     yolo_weights_path='model_data/yolo_weights.h5'):
        # K.clear_session()  # get a new session
        image_input = Input(shape=self.shape)
        h, w = self.input_shape
        num_anchors = len(self.anchors)

        model_body = yolo_body(image_input, num_anchors // 3, self.num_classes)
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, self.num_classes))

        if load_pretrained:
            model_body.load_weights(yolo_weights_path)
            print('Load Yolo weights {}.'.format(yolo_weights_path))
            if freeze_body in [1, 2]:
                # Freeze darknet53 body or freeze all but 3 output layers.
                num = (185, len(model_body.layers) - 3)[freeze_body - 1]
                for i in range(num): model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        # -------------------------------
        #          Yolo Detector
        # -------------------------------
        y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                               num_anchors // 3, self.num_classes + 5)) for l in range(3)]
        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': self.anchors, 'num_classes': self.num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])
        yolo_model = Model(inputs=[image_input, *y_true], outputs=model_loss)
        print('==========================================   Yolo Body   =========================================')
        yolo_model.summary()
        return yolo_model

    def train(self):

        # Adversarial ground truths
        dummy_r = np.zeros(self.step1_batch_size)  # Dummy gt for gradient rodnet

        # Data generator
        yolo_train_batch = self.data_generator_wrapper(self.train_data, self.step1_batch_size,
                                                       self.input_shape, self.anchors, self.num_classes)

        print('Start')
        print('Evaluate mAP')
        mAP, recall, precision = self.eval(-1, self.val_images, self.tmp_gt_files_path, 'mAP')
        print('###########################################')
        print('mAP is: %s' % mAP)
        print('recall is %s' % recall)
        print('precision is %s' % precision)
        print('###########################################')
        epoch = len(self.train_data) // self.step1_batch_size

        # Step1
        start = 0
        end = 50 * epoch
        # Yolo Compile
        self.model.compile(loss={'yolo_loss': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr=1e-3))
        for step in range(start, end):
            # ---------------------
            #      Train Yolo
            # ---------------------
            img_input, y_true = next(yolo_train_batch)
            # y_loss = self.yolo_model.train_on_batch([img_input, *y_true], dummy_r)
            y_loss = self.model.train_on_batch([img_input, *y_true], dummy_r)
            print("%d [Yolo loss: %f]" % (step, y_loss))

            # Evaluate
            if step % epoch == 0:
                print('Evaluate mAP')
                mAP, recall, precision = self.eval(-1, self.val_images, self.tmp_gt_files_path, 'mAP')
                print('###########################################')
                print('mAP is: %s' % mAP)
                print('recall is %s' % recall)
                print('precision is %s' % precision)
                print('###########################################')

            # Save model every 5 epoch
            if step % epoch == 5:
                # self.yolo_model.save_weights(os.path.join(MODELS_PATH, 'Step1_yolo_weight_{}.h5'.format(step // epoch)))
                self.model.save_weights(os.path.join(MODELS_PATH, 'Step1_yolo_weight_{}.h5'.format(step // epoch)))

        # Step2
        start = end
        end = end + 50 * epoch
        self.set_trainability(self.yolo_model, trainable=True)
        self.yolo_model.compile(loss={'yolo_loss': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr=1e-3))
        for step in range(start, end):
            # ---------------------
            #      Train Yolo
            # ---------------------
            img_input, y_true = next(yolo_train_batch)
            y_loss = self.yolo_model.train_on_batch([img_input, *y_true], dummy_r)
            print("%d [Yolo loss: %f]" % (step, y_loss))


            # Evaluate
            if step % epoch == 0:
                print('Evaluate mAP')
                mAP, recall, precision = self.eval(-1, self.val_images, self.tmp_gt_files_path, 'mAP')
                print('###########################################')
                print('mAP is: %s' % mAP)
                print('recall is %s' % recall)
                print('precision is %s' % precision)
                print('###########################################')

            # Save model every 5 epoch
            if step % epoch == 5:
                self.yolo_model.save_weights(os.path.join(MODELS_PATH, 'Step1_yolo_weight_{}.h5'.format(step // epoch)))

    def eval(self, step, eval_images_path, ground_truth_path, tag='image'):
        # Add the class predict temp dict
        class_pred_tmp = {}
        for class_name in self.class_names:
            class_pred_tmp[class_name] = []

        # Predict!!!
        for start in range(0, len(eval_images_path), self.step2_batch_size):
            end = start + self.step2_batch_size
            images_path = eval_images_path[start:end]
            images = []
            images_org = []
            images_shape = []
            files_id = []
            for image_path in images_path:
                image = Image.open(image_path)
                file_id = os.path.split(image_path)[-1].split('.')[0]
                boxed_image = letterbox_image(image, tuple(reversed(self.input_shape)))
                image_data = np.array(boxed_image, dtype='float32')
                image_data /= 255.
                images_shape.append([image.size[1], image.size[0]])
                images.append(image_data)
                images_org.append(image)
                files_id.append(file_id)
            images = np.array(images)

            out_bboxes_1, out_bboxes_2, out_bboxes_3 = self.yolo_model.predict(images)
            # TODO: fix here
            for i, out in enumerate(zip(out_bboxes_1, out_bboxes_2, out_bboxes_3)):
                # Predict
                out_boxes, out_scores, out_classes = self.sess.run(
                    [self.boxes, self.scores, self.classes],
                    feed_dict={
                        # self.eval_inputs: out
                        self.eval_inputs[0]: np.expand_dims(out[0], 0),
                        self.eval_inputs[1]: np.expand_dims(out[1], 0),
                        self.eval_inputs[2]: np.expand_dims(out[2], 0),
                        self.input_image_shape: images_shape[i]
                    })

                image = np.array(images_org[i])
                ord_h = image.shape[0]
                ord_w = image.shape[1]
                new_h = int(image.shape[0] * 3 / 4)
                new_w = int(image.shape[1] * 3 / 4)
                for o, c in enumerate(out_classes):
                    predicted_class = self.class_names[c]
                    box = out_boxes[o]
                    score = out_scores[o]

                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(ord_h, np.floor(bottom + 0.5).astype('int32'))
                    right = min(ord_w, np.floor(right + 0.5).astype('int32'))

                    bbox = "{} {} {} {}".format(left, top, right, bottom)
                    class_pred_tmp[predicted_class].append(
                        {"confidence": str(score), "file_id": files_id[i], "bbox": bbox})

        # Create predict temp
        for class_name in self.class_names:
            with open(self.tmp_pred_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
                json.dump(class_pred_tmp[class_name], outfile)

        # calculate the AP for each class
        sum_AP = 0.0
        count_true_positives = {}
        for class_index, class_name in enumerate(sorted(self.gt_counter_per_class.keys())):
            count_true_positives[class_name] = 0

            # load predictions of that class
            predictions_file = self.tmp_pred_files_path + "/" + class_name + "_predictions.json"
            predictions_data = json.load(open(predictions_file))

            # Assign predictions to ground truth objects
            nd = len(predictions_data)  # number of predict data
            tp = [0] * nd  # true positive
            fp = [0] * nd  # false positive
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                gt_file = ground_truth_path + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load prediction bounding-box
                bb = [float(x) for x in prediction["bbox"].split()]
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        # Area of Overlap
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        # compute overlap (IoU) = area of intersection / area of union
                        if iw > 0 and ih > 0:
                            # Area of Union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                                 (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                if ovmax >= self.min_overlap:
                    if not gt_match['used']:
                        tp[idx] = 1
                        gt_match["used"] = True
                        # count_true_positives[predicted_class] += 1
                    else:
                        fp[idx] = 1
                else:
                    fp[idx] = 1

            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / self.gt_counter_per_class[class_name]
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

            ap, mrec, mprec = self.voc_ap(rec, prec)
            sum_AP += ap

        mAP = sum_AP / len(self.gt_counter_per_class)

        return mAP, mrec, mprec

    def read_txt_file(self):
        # Training data
        val_split = 0.2
        with open(self.train_annotation_path) as f:
            train_lines = f.readlines()
        if not self.val_annotation_path == 'nano':
            with open(self.val_annotation_path) as f:
                val_lines = f.readlines()
        else:
            val_lines = []
        lines = train_lines + val_lines
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val

        # Train data
        train_data = lines[:num_train]

        # Val data
        val_data = lines[num_train:]
        val_images = []
        # val_bboxes = []
        for data in val_data:
            val_bboxes = []
            image, *bboxes = data.split()
            file_id = os.path.split(image)[-1].split('.')[0]
            val_images.append(image)
            for bbox in bboxes:
                left, bottom, right, top, class_id = bbox.split(',')
                class_name = self.class_names[int(class_id)]
                bbox = "{} {} {} {}".format(left, bottom, right, top)
                val_bboxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                self.gt_counter_per_class[class_name] += 1

            with open(self.tmp_gt_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
                json.dump(val_bboxes, outfile)

        return train_data, val_data, val_images

    @staticmethod
    def set_trainability(model, trainable=False):
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable

    @staticmethod
    def get_classes(classes_path):
        """loads the classes"""
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    @staticmethod
    def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
        """data generator for fit_generator"""
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if i == 0:
                    np.random.shuffle(annotation_lines)
                image, box = get_random_data(annotation_lines[i], input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i = (i + 1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield image_data, y_true

    @staticmethod
    def get_anchors(anchors_path):
        """loads the anchors from a file"""
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    @staticmethod
    def voc_ap(rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0)  # insert 0.0 at begining of list
        rec.append(1.0)  # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0)  # insert 0.0 at begining of list
        prec.append(0.0)  # insert 0.0 at end of list
        mpre = prec[:]
        """
         This part makes the precision monotonically decreasing
          (goes from the end to the beginning)
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #   range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #   range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        """
         This part creates a list of indexes where the recall changes
        """
        # matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)  # if it was matlab would be i + 1
        """
         The Average Precision (AP) is the area under the curve
          (numerical integration)
        """
        # matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap, mrec, mpre

    @classmethod
    def data_generator_wrapper(cls, annotation_lines, batch_size, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n == 0 or batch_size <= 0:
            return None
        return cls.data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

    def close(self):
        self.sess.close()
        rm_dirs = glob('tmp_*')
        for dir in rm_dirs:
            shutil.rmtree(dir)


if __name__ == "__main__":
    yolo = Yolo()
    yolo.train()
    yolo.close()