import os
import colorsys
import torch
import numpy as np
import time
import cv2
import onnx
import onnxruntime
import os
os.environ['TRIDENT_BACKEND'] = 'pytorch'
import builtins
import trident as T
from trident import *
from trident.models.pytorch_yolo import *

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

def infer_onnx(session,input):
    inp=session.get_inputs()
    input_shape =inp[0].shape
    input_name = inp[0].name
    input_type= inp[0].type
    out=session.get_outputs()
    pred_onnx = session.run(None, {input_name: input.astype(np.float32)})
    return pred_onnx[0]


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    if box_scores is None or len(box_scores) == 0:
        return None, None
    scores = box_scores[:, -1]
    boxes = box_scores[:, :4]
    picked = []
    # _, indexes = scores.sort(descending=True)
    indexes = np.argsort(scores)
    # indexes = indexes[:candidate_size]
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        # current = indexes[0]
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        # indexes = indexes[1:]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(rest_boxes, np.expand_dims(current_box, axis=0), )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :], picked


yolo=onnxruntime.InferenceSession('Models/pretrained_yolov4_mscoco.onnx')


cap = cv2.VideoCapture('MOTORBIKES.mp4')
vw = cv2.VideoWriter('autodrive_v1.avi', cv2.VideoWriter_fourcc('M','J','P','G'),30, (608, 342))
class_names=None
with open( 'pretrained/coco_chinese_classes.txt', 'r', encoding='utf-8-sig') as f:
    labels = [l.rstrip() for l in f]
    class_names = labels

frame_width=1920
frame_height=1080
scale=builtins.min(608.0/frame_width,608.0/frame_height)

detection_threshold=0.4
iou_threshold=0.5
palette = generate_palette(80)
sum = 0
tot_area=None
while True:
    ret, bgr_image = cap.read()

    if bgr_image is None:
        vw.release()
        print("no img")
        break

    bgr_image = cv2.resize(bgr_image.astype(np.float32),None, fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    if tot_area is None:
        tot_area=bgr_image.shape[0] * bgr_image.shape[1]

    #cv2.imwrite('resized.jpg',resized_bgr_image)
    bgr_arr=np.zeros((608,608,3)).astype(np.float32)
    bgr_arr[:bgr_image.shape[0],:bgr_image.shape[1],:]=bgr_image.copy()
    #bgr_arr=bgr_image.copy()
    bgr_arr = bgr_arr /255
    bgr_arr = np.transpose(bgr_arr, [2, 0, 1])
    bgr_arr = np.expand_dims(bgr_arr, axis=0)
    bgr_arr = bgr_arr.astype(np.float32)
    # confidences, boxes = predictor.run(image)
    time_time = time.time()
    boxes = infer_onnx(yolo,bgr_arr)[0]
    print("infer fps:{}".format(1.0/(time.time() - time_time)))
    mask1=boxes[:,2]*boxes[:,3]/tot_area<=0.6
    boxes = boxes[mask1]
    if len(boxes)>0:
        mask2 = boxes[:, 4] > detection_threshold
        boxes = boxes[mask2]
    if boxes is not None and len(boxes) > 0:
        boxes = np.concatenate([xywh2xyxy(boxes[:, :4]), boxes[:, 4:]], axis=-1)
        if len(boxes) > 1:
            box_probs, keep = hard_nms(boxes[:, :5], iou_threshold=iou_threshold, top_k=-1, )
            boxes = boxes[keep]
            print('   {0} bboxes keep after nms!'.format(len(boxes)))
        #boxes[:, :4] /= scale
        boxes[:, :4] = np.round(boxes[:, :4], 0)


        print("======== bbox postprocess time:{0:.5f}".format((time.time() - time_time)))
        time_time = time.time()
        # boxes = boxes * (1 / scale[0])
        locations = boxes[:, :4]
        probs = boxes[:, 4]
        labels = np.argmax(boxes[:, 5:], -1).astype(np.int32)
        if locations is not None:
            for i in range(len(locations)):
                print('         box{0}: {1} prob:{2:.2%} class:{3}'.format(i, [np.round(num, 4) for num in
                                                                               locations[i].tolist()], probs[i],
                                                                           labels[i] if class_names is None or int(
                                                                               labels[i]) >= len(class_names) else
                                                                           class_names[int(labels[i])]))

        #locations, labels, probs
        time_time = time.time()

        if locations is not None and len(locations) > 0:
            if locations.ndim == 1:
                locations = np.expand_dims(locations, 0)
            if labels.ndim == 0:
                labels = np.expand_dims(labels, 0)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            pillow_img=array2image(rgb_image.copy())
            for m in range(len(locations)):
                this_box = locations[m]
                this_label = labels[m]
                thiscolor = tuple([int(c) for c in palette[this_label][:3]])
                pillow_img=plot_bbox(this_box, pillow_img, thiscolor,class_names[int(this_label)], line_thickness=2)  # cv2.rectangle(bgr_image, (this_box[0], this_box[1]), (this_box[2], this_box[3]),  #               self.palette[this_label],  #               1 if bgr_image.shape[1] < 480 else 2 if bgr_image.shape[1] < 640 else 3 if  #               bgr_image.shape[1] < 960 else 4)

            rgb_image =  np.array(pillow_img.copy())
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        print("======== draw image time:{0:.5f}".format((time.time() - time_time)))
    vw.write(bgr_image)
    cv2.imshow('annotated', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("sum:{}".format(sum))

