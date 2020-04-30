from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs

import os
os.environ['TRIDENT_BACKEND'] = 'pytorch'
import trident as T
from trident import *


__all__ = [ 'resblock_body', 'YoloDetectionModel', 'DarknetConv2D', 'DarknetConv2D_BN_Mish',
           'DarknetConv2D_BN_Leaky', 'YoloLayer']

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_epsilon = _session.epsilon
_trident_dir = _session.trident_dir

anchors = np.array([12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]).reshape(
    (1, 1, 1, -1, 2))



def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'use_bias': True}
    darknet_conv_kwargs['auto_pad'] = False if kwargs.get('strides')==(2,2) else True
    darknet_conv_kwargs['use_bias'] = True
    darknet_conv_kwargs.update(kwargs)
    return Conv2d(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    darknet_conv_kwargs = {'use_bias': False,'normalization':BatchNorm2d(momentum=0.03,eps=1e-4)}
    darknet_conv_kwargs['activation']=LeakyRelu(alpha=0.1)
    darknet_conv_kwargs['auto_pad'] = False if kwargs.get('strides') == (2, 2) else True
    darknet_conv_kwargs.update(kwargs)
    return Conv2d_Block(*args, **darknet_conv_kwargs)



def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    darknet_conv_kwargs = {'use_bias': False, 'normalization':BatchNorm2d(momentum=0.03,eps=1e-4), 'activation': Mish}
    darknet_conv_kwargs['auto_pad'] = False if kwargs.get('strides') == (2, 2) else True
    darknet_conv_kwargs.update(kwargs)
    return Conv2d_Block(*args, **darknet_conv_kwargs)


def resblock_body(num_filters, num_blocks, all_narrow=True,keep_output=False,name=''):
    # block=Sequential()
    # block.add_module(name+'_preconv1',DarknetConv2D_BN_Mish((3, 3),num_filters , strides=(2, 2),auto_pad=False, padding=((1,0),(1,0)),name=name+'_preconv1'))
    # shortconv=DarknetConv2D_BN_Mish((1, 1), num_filters // 2 if all_narrow else num_filters, name=name + '_shortconv')
    # branch=Sequential(
    #             DarknetConv2D_BN_Mish((1, 1), num_filters // 2 if all_narrow else num_filters,name=name+'_mainconv'),
    #             For(range(num_blocks), lambda i:
    #                 ShortCut2d(
    #                     Identity(),
    #                     Sequential(
    #                         DarknetConv2D_BN_Mish((1, 1),num_filters // 2,name=name+'_for{0}_1'.format(i)),
    #                         DarknetConv2D_BN_Mish((3, 3),num_filters // 2 if all_narrow else num_filters,name=name+'_for{0}_2'.format(i))
    #                     ),
    #                     mode='add')
    #             ),
    #             DarknetConv2D_BN_Mish( (1, 1),num_filters // 2 if all_narrow else num_filters,name=name+'_postconv')
    #         )
    # block.add_module(name+'_route',ShortCut2d(branch,shortconv,mode='concate',name=name+'_route'))
    # block.add_module(name+'_convblock5',DarknetConv2D_BN_Mish((1,1),num_filters,name=name+'_convblock5'))
    # return block
    return Sequential(
        DarknetConv2D_BN_Mish((3, 3),num_filters , strides=(2, 2),auto_pad=False, padding=((1,0),(1,0)),name=name+'_preconv1'),
        ShortCut2d(
            {
            1:DarknetConv2D_BN_Mish((1, 1), num_filters // 2 if all_narrow else num_filters, name=name + '_shortconv'),
            0:Sequential(
                DarknetConv2D_BN_Mish((1, 1), num_filters // 2 if all_narrow else num_filters,name=name+'_mainconv'),
                For(range(num_blocks), lambda i:
                    ShortCut2d(
                        Identity(),
                        Sequential(
                            DarknetConv2D_BN_Mish((1, 1),num_filters // 2,name=name+'_for{0}_1'.format(i)),
                            DarknetConv2D_BN_Mish((3, 3),num_filters // 2 if all_narrow else num_filters,name=name+'_for{0}_2'.format(i))
                        ),
                        mode='add')
                ),
                DarknetConv2D_BN_Mish( (1, 1),num_filters // 2 if all_narrow else num_filters,name=name+'_postconv')
            )},
            mode='concate',name=name+'_route'),

        DarknetConv2D_BN_Mish((1,1),num_filters,name=name+'_convblock5')
        )







class YoloLayer(Layer):
    """Detection layer"""

    def __init__(self, anchors, num_classes,grid_size, img_dim=608):
        super(YoloLayer, self).__init__()
        self.register_buffer('grid', None)
        self.register_buffer('anchors', to_tensor(anchors, requires_grad=False).to(get_device()))

        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = grid_size # grid size


        self.compute_grid_offsets(grid_size)

    def compute_grid_offsets(self, grid_size):

        self.stride = self.img_dim / grid_size
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1,self.num_anchors, 1, 1, 2)
        yv, xv = torch.meshgrid([torch.arange(grid_size, device=get_device()), torch.arange(grid_size, device=get_device())])
        self.grid = torch.stack((xv, yv), 2).view((1, 1, grid_size, grid_size, 2)).float()
        self.grid1=meshgrid(grid_size,grid_size,requires_grad=False).view([1, 1, grid_size,grid_size,2])
        # Calculate offsets for each grid

    def forward(self, x, targets=None):
        num_samples = x.size(0)
        grid_size = x.size(2)

        # # If grid size does not match current we compute new offsets
        # if grid_size != self.grid_size:
        #     self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        prediction = x.clone().view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        xy = sigmoid(prediction[..., 0:2])  # Center x
        wh = prediction[..., 2:4]  # Width

        pred_conf = sigmoid(prediction[..., 4])  # Conf
        pred_cls = sigmoid(prediction[..., 5:])  # Cls pred.
        cls_probs=reduce_max(pred_cls,-1,keepdims=True)
        # Add offset and scale with anchors
        pred_boxes = zeros_like(prediction[..., :4])
        pred_boxes[..., 0:2] = xy + self.grid.to(get_device())
        pred_boxes[..., 2:4] = exp(wh) * self.anchor_wh.to(get_device())


        output = torch.cat((pred_boxes.view(num_samples, -1, 4) * self.stride, pred_conf.view(num_samples, -1, 1), pred_cls.view(num_samples, -1, self.num_classes),), -1, )
        return output
        # if targets is None:  #     return output,  # else:  #     iou_scores, class_mask, obj_mask,
        # noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(  #         pred_boxes=pred_boxes,
        #         pred_cls=pred_cls,  #         target=targets,  #         anchors=self.scaled_anchors,
        #         ignore_thres=self.ignore_thres,  #     )  #  #     # Loss : Mask outputs to ignore non-existing
        #         objects (except with conf. loss)  #     loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])  #
        #         loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])  #     loss_w = self.mse_loss(w[obj_mask],
        #         tw[obj_mask])  #     loss_h = self.mse_loss(h[obj_mask], th[obj_mask])  #     loss_conf_obj =
        #         self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])  #     loss_conf_noobj = self.bce_loss(
        #         pred_conf[noobj_mask], tconf[noobj_mask])  #     loss_conf = self.obj_scale * loss_conf_obj +
        #         self.noobj_scale * loss_conf_noobj  #     loss_cls = self.bce_loss(pred_cls[obj_mask],
        #         tcls[obj_mask])  #     total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls  #  #
        # Metrics  #     cls_acc = 100 * class_mask[obj_mask].mean()  #     conf_obj = pred_conf[obj_mask].mean()  #
        # conf_noobj = pred_conf[noobj_mask].mean()  #     conf50 = (pred_conf > 0.5).float()  #     iou50 = (
        # iou_scores > 0.5).float()  #     iou75 = (iou_scores > 0.75).float()  #     detected_mask = conf50 *
        # class_mask * tconf  #     precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)  #
        # recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)  #     recall75 = torch.sum(iou75 *
        # detected_mask) / (obj_mask.sum() + 1e-16)  #  #     self.metrics = {  #         "loss": to_numpy(
        # total_loss).item(),  #         "x":to_numpy(loss_x).item(),  #         "y": to_numpy(loss_y).item(),
        #         "w": to_numpy(loss_w).item(),  #         "h": to_numpy(loss_h).item(),  #         "conf": to_cpu(
        #         loss_conf).item(),  #         "cls": to_numpy(loss_cls).item(),  #         "cls_acc": to_cpu(
        #         cls_acc).item(),  #         "recall50": to_cpu(recall50).item(),  #         "recall75": to_cpu(
        #         recall75).item(),  #         "precision": to_cpu(precision).item(),  #         "conf_obj": to_cpu(
        #         conf_obj).item(),  #         "conf_noobj": to_cpu(conf_noobj).item(),  #         "grid_size": grid_size,  #     }  #  #     return output, total_loss



class YoloDetectionModel(ImageDetectionModel):
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(YoloDetectionModel, self).__init__(inputs, output, input_shape)
        self.preprocess_flow = [resize((input_shape[-2], input_shape[-1]), True), normalize(0, 255)]
        self.detection_threshold = 0.7
        self.iou_threshold = 0.3
        self.class_names = None
        self.palette = generate_palette(80)

    def area_of(self, left_top, right_bottom):
        """Compute the areas of rectangles given two corners.

        Args:
            left_top (N, 2): left top corner.
            right_bottom (N, 2): right bottom corner.

        Returns:
            area (N): return the area.
        """
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def iou_of(self, boxes0, boxes1, eps=1e-5):
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

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
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
            iou = self.iou_of(rest_boxes, np.expand_dims(current_box, axis=0), )
            indexes = indexes[iou <= iou_threshold]

        return box_scores[picked, :], picked

    def nms(self, boxes, threshold=0.3):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list, add the index
            # value to the list of picked indexes, then initialize
            # the suppression list (i.e. indexes that will be deleted)
            # using the last index
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]

            # loop over all indexes in the indexes list
            for pos in range(0, last):
                # grab the current index
                j = idxs[pos]

                # find the largest (x, y) coordinates for the start of
                # the bounding box and the smallest (x, y) coordinates
                # for the end of the bounding box
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])

                # compute the width and height of the bounding box
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                # compute the ratio of overlap between the computed
                # bounding box and the bounding box in the area list
                overlap = float(w * h) / area[j]

                # if there is sufficient overlap, suppress the
                # current bounding box
                if overlap > threshold:
                    suppress.append(pos)

            # delete all indexes from the index list that are in the
            # suppression list
            idxs = np.delete(idxs, suppress)

        # return only the bounding boxes that were picked
        return boxes[pick], pick

    def infer_single_image(self, img, scale=1, verbose=False):
        time_time =None
        if verbose:
            time_time = time.time()
            print("==-----  starting infer {0} -----=======".format(img))
        if self._model.built:
            try:
                self._model.to(self.device)
                self._model.eval()
                img = image2array(img)
                if img.shape[-1] == 4:
                    img = img[:, :, :3]
                img_orig = img.copy()

                for func in self.preprocess_flow:
                    if inspect.isfunction(func):
                        img = func(img)
                        if func.__qualname__ == 'resize.<locals>.img_op':
                            scale = func.scale

                img = image_backend_adaptive(img)
                inp = to_tensor(np.expand_dims(img, 0)).to(self.device).to(self._model.weights[0].data.dtype)

                if verbose:
                    print("======== data preprocess time:{0:.5f}".format((time.time() - time_time)))
                    time_time = time.time()

                boxes = self._model(inp)[0]
                if verbose:
                    print("======== infer  time:{0:.5f}".format((time.time() - time_time)))
                    time_time = time.time()

                mask = boxes[:, 4] > self.detection_threshold
                boxes = boxes[mask]
                if verbose:
                    print('         detection threshold:{0}'.format(self.detection_threshold))
                    print('         {0} bboxes keep!'.format(len(boxes)))
                if boxes is not None and len(boxes) > 0:
                    boxes = to_numpy(boxes)
                    boxes = np.concatenate([xywh2xyxy(boxes[:, :4]), boxes[:, 4:]], axis=-1)
                    if len(boxes) > 1:
                        box_probs, keep = self.hard_nms(boxes[:, :5], iou_threshold=self.iou_threshold, top_k=-1, )
                        boxes = boxes[keep]
                        print('         iou threshold:{0}'.format(self.iou_threshold))
                        print('         {0} bboxes keep!'.format(len(boxes)))
                    boxes[:, :4] /=scale
                    boxes[:, :4]=np.round(boxes[:, :4],0)
                    if verbose:
                        print("======== bbox postprocess time:{0:.5f}".format((time.time() - time_time)))
                        time_time = time.time()
                    # boxes = boxes * (1 / scale[0])
                    locations= boxes[:, :4]
                    probs = boxes[:, 4]
                    labels=np.argmax(boxes[:, 5:], -1).astype(np.int32)

                    if verbose and locations is not None:
                        for i in range(len(locations)):
                            print('         box{0}: {1} prob:{2:.2%} class:{3}'.format(i, [round(num, 4) for num in
                                                                                           locations[i].tolist()], probs[i],
                                                                                       labels[i] if self.class_names is
                                                                                                 None or i >= len(
                                                                                           self.class_names) else
                                                                                       self.class_names[
                                                                                           int(labels[i])]))

                    return img_orig,locations,labels,probs

                else:
                    return img_orig, None, None, None
            except:
                PrintException()
        else:
            raise ValueError('the model is not built yet.')

    def infer_then_draw_single_image(self, img, scale=1, verbose=False):

        rgb_image, boxes, labels, probs = self.infer_single_image(img, scale, verbose)
        time_time = None
        if verbose:
            time_time = time.time()

        if boxes is not None and len(boxes) > 0:
            boxes = np.round(boxes).astype(np.int32)
            if boxes.ndim == 1:
                boxes = np.expand_dims(boxes, 0)
            if labels.ndim == 0:
                labels = np.expand_dims(labels, 0)
            for m in range(len(boxes)):
                this_box = boxes[m]
                this_label = labels[m]
                thiscolor=tuple([int(c) for c in self.palette[this_label][:3]])
                rgb_image=plot_bbox(this_box,rgb_image,thiscolor,self.class_names[int(labels[m])])
                # cv2.rectangle(bgr_image, (this_box[0], this_box[1]), (this_box[2], this_box[3]),
                #               self.palette[this_label],
                #               1 if bgr_image.shape[1] < 480 else 2 if bgr_image.shape[1] < 640 else 3 if
                #               bgr_image.shape[1] < 960 else 4)
        #rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
        if verbose:
            print("======== draw image time:{0:.5f}".format((time.time() - time_time)))
        return rgb_image


