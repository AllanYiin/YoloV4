import os
import time
os.environ['TRIDENT_BACKEND'] =  'tensorflow'
import glob
from trident import *
from tf_yolo import *
from tf_darknet import *


detector=YoloDetectionModel(input_shape=(608,608,3),output=yolo4_body(80, 608))
detector.load_model('Models/pretrained_yolov4_mscoco_tf.pth.tar')



detector.eval()
detector.summary()

#image preprocess
detector.preprocess_flow=[resize((608,608),keep_aspect=True),to_bgr(),normalize(0,255) ]
#detection threshold
detector.detection_threshold=0.4
#iou_threshold
detector.iou_threshold=0.5

is_small_item_enhance=True
#enable small item enhancement
if is_small_item_enhance:
    for module in detector.model.modules():
        if isinstance(module,YoloLayer):
            module.small_item_enhance=True



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


is_drawing = True
for i in range(len(imgs)):
    folder, filename, ext = split_path(imgs[i])
    if is_drawing:
        img = detector.infer_then_draw_single_image(imgs[i], verbose=True)
        if is_small_item_enhance:
            array2image(img).save(imgs[i].replace(filename, filename + '_tf_infered_enhance'))
        else:
            array2image(img).save(imgs[i].replace(filename, filename + '_tf_infered'))
    else:
        detector.infer_single_image(imgs[i], verbose=True)
