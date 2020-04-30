import os
import time
os.environ['TRIDENT_BACKEND'] = 'pytorch'
import glob
from trident import *
from pytorch_yolo import *


detector=YoloDetectionModel(input_shape=(3,608,608),output=load('Models/pretrained_yolov4_mscoco.pth'))
#image preprocess
detector.preprocess_flow=[resize((608,608),keep_aspect=True),to_bgr(),normalize(0,255) ]
#detection threshold
detector.detection_threshold=0.4
#iou_threshold
detector.iou_threshold=0.6

#freeze model
detector.trainable=False
#evaluation
detector.eval()


#setting classname
with open( 'pretrained/coco_chinese_classes.txt', 'r', encoding='utf-8-sig') as f:
    labels = [l.rstrip() for l in f]
    detector.class_names = labels
#if u want you also can set detector.palette (list of color)


imgs=glob.glob('images/*.*g')
imgs=[img for img in imgs if   '_infered' not in  img]
print('total {0} images'.format(len(imgs)))

is_drawing=True
for i in range(len(imgs)):
    folder,filename,ext=split_path(imgs[i])
    if is_drawing:
        img=detector.infer_then_draw_single_image(imgs[i],verbose=True)
        array2image(img).save(imgs[i].replace(filename,filename+'_infered'))
    else:
        detector.infer_single_image(imgs[i], verbose=True)