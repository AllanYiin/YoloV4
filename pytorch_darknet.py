import os

os.environ['TRIDENT_BACKEND'] = 'pytorch'

import trident as T
from trident import *
from pytorch_yolo import *



def yolo4_body(num_classes=80,image_size=608):
    anchors=np.array([12, 16,  19, 36,  40, 28,  36, 75,  76, 55,  72, 146,  142, 110,  192, 243,  459, 401]).reshape(-1, 2)
    anchors1 = to_tensor(np.array([12, 16, 19, 36, 40, 28]).reshape(-1, 2),requires_grad=False)
    anchors2 = to_tensor(np.array([36, 75, 76, 55, 72, 146]).reshape(-1, 2),requires_grad=False)
    anchors3 = to_tensor(np.array([142, 110, 192, 243, 459, 401]).reshape(-1, 2),requires_grad=False)
    num_anchors=len(anchors1)
    """Create YOLO_V4 model CNN body in Keras."""
    return Sequential(
            DarknetConv2D_BN_Mish((3, 3), 32,name='first_layer'),
            resblock_body(64, 1, all_narrow=False,name='block64'),
            resblock_body(128, 2,name='block128'),
            resblock_body(256, 8,name='block256'),
            ShortCut2d(
                {
                    1:Sequential(
                        resblock_body(512, 8,name='block512'),
                        ShortCut2d(
                            {
                                1:Sequential(
                                    resblock_body(1024, 4, name='block1024'),
                                    DarknetConv2D_BN_Leaky( (1,1), 512,name='pre_maxpool1'),
                                    DarknetConv2D_BN_Leaky( (3, 3),1024,name='pre_maxpool2'),
                                    DarknetConv2D_BN_Leaky((1,1),512,name='pre_maxpool3'),
                                    ShortCut2d(
                                        MaxPool2d((13,13),strides=(1,1),auto_pad=True),
                                        MaxPool2d((9,9), strides=(1, 1), auto_pad=True),
                                        MaxPool2d((5,5), strides=(1, 1), auto_pad=True),
                                        Identity(),
                                        mode='concate'
                                    ),
                                    DarknetConv2D_BN_Leaky((1, 1), 512,name='pre_y19_1'),
                                    DarknetConv2D_BN_Leaky((3, 3), 1024,name='pre_y19_2'),
                                    DarknetConv2D_BN_Leaky((1, 1), 512,name='y_19',keep_output=True),
                                    DarknetConv2D_BN_Leaky((1, 1),256,name='pre_y19_upsample'),
                                    Upsampling2d(scale_factor=2,name='y19_upsample'),
                                ),
                                0:DarknetConv2D_BN_Leaky((1, 1), 256)
                            },mode='concate'),
                        DarknetConv2D_BN_Leaky((1, 1),256,name='pre_y38_1'),
                        DarknetConv2D_BN_Leaky((3, 3),512,name='pre_y38_2'),
                        DarknetConv2D_BN_Leaky((1, 1),256,name='pre_y38_3'),
                        DarknetConv2D_BN_Leaky((3, 3),512,name='pre_y38_4'),
                        DarknetConv2D_BN_Leaky((1, 1),256,name='y_38',keep_output=True),
                        DarknetConv2D_BN_Leaky((1, 1),128,name='pre_y_38_upsample'),
                        Upsampling2d(scale_factor=2,name='y_38_upsample'),
                    ),
                    0:DarknetConv2D_BN_Leaky((1, 1), 128)
                },
                mode='concate'),
            DarknetConv2D_BN_Leaky((1, 1), 128,name='pre_y76_concate1'),
            DarknetConv2D_BN_Leaky((3, 3), 256,name='pre_y76_concate2'),
            DarknetConv2D_BN_Leaky((1, 1), 128,name='pre_y76_concate3'),
            DarknetConv2D_BN_Leaky((3, 3), 256,name='pre_y76_concate4'),
            DarknetConv2D_BN_Leaky((1, 1), 128,name='pre_y76_concate5'),
            ShortCut2d(
                #y76_output
                Sequential(
                    DarknetConv2D_BN_Leaky( (3, 3),256,name='pre_y76_output'),
                    DarknetConv2D( (1, 1),num_anchors * (num_classes + 5),use_bias=True,name='y76_output'),
                    YoloLayer(anchors=anchors1,num_classes=num_classes,grid_size=76, img_dim=image_size),
                name='y76_output'),
                # y38_output
                Sequential(
                    ShortCut2d(
                        DarknetConv2D_BN_Leaky((3, 3), 256, strides=(2, 2), auto_pad=False, padding=((1, 0), (1, 0)),name='y76_downsample'),
                        branch_from='y_38',mode='concate'),
                    DarknetConv2D_BN_Leaky((1, 1), 256,name='pre_y38_concate1'),
                    DarknetConv2D_BN_Leaky((3, 3), 512,name='pre_y38_concate2'),
                    DarknetConv2D_BN_Leaky((1, 1), 256,name='pre_y38_concate3'),
                    DarknetConv2D_BN_Leaky((3, 3), 512,name='pre_y38_concate4'),
                    DarknetConv2D_BN_Leaky((1, 1), 256,name='pre_y38_concate5'),
                    ShortCut2d(
                        Sequential(
                            DarknetConv2D_BN_Leaky((3, 3), 512, name='pre_y38_output'),
                            DarknetConv2D((1, 1), num_anchors * (num_classes + 5), use_bias=True, name='y38_output'),
                            YoloLayer(anchors=anchors2, num_classes=num_classes,grid_size=38,  img_dim=image_size),
                            name='y38_output'),

                        Sequential(
                            ShortCut2d(
                                DarknetConv2D_BN_Leaky((3, 3), 512, strides=(2, 2), auto_pad=False, padding=((1, 0), (1, 0)),name='y38_downsample'),
                                branch_from='y_19', mode='concate'),
                            DarknetConv2D_BN_Leaky((1, 1), 512,name='pre_y19_concate1'),
                            DarknetConv2D_BN_Leaky((3, 3), 1024,name='pre_y19_concate2'),
                            DarknetConv2D_BN_Leaky((1, 1), 512,name='pre_y19_concate3'),
                            DarknetConv2D_BN_Leaky((3, 3), 1024,name='pre_y19_concate4'),
                            DarknetConv2D_BN_Leaky((1, 1), 512,name='pre_y19_concate5'),
                            Sequential(
                                DarknetConv2D_BN_Leaky((3, 3),1024,name='pre_y19_output'),
                                DarknetConv2D((1, 1), num_anchors * (num_classes + 5),use_bias=True,name='y19_output'),
                                YoloLayer(anchors=anchors3,num_classes=num_classes,grid_size=19, img_dim=image_size),
                            name='y19_output')),

                        mode='concate')
                )
                ,mode = 'concate')
    )



#parse configure
def parse_model_cfg(path):
    # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'
    if not path.endswith('.cfg'):  # add .cfg suffix if omitted
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):  # add cfg/ prefix if omitted
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return nparray
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):  # return array
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():  # return int or float
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string

    # Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh']

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)
    return mdefs



def load_pretrained_weight(yolov4,cfg_path='pretrained/yolov4.cfg',weight_path='pretrained/yolov4.weights'):
    conv_dict=OrderedDict()
    norm_dict=OrderedDict()

    module_defs = parse_model_cfg(cfg_path)
    module_defs=[mdef for mdef in module_defs if mdef['type'] == 'convolutional']  #oly convolution have weights

    #dont't use the ordinal from modules()
    #we need follow the ordinal the same as cfg file
    modules_dict=OrderedDict([(conv.defaultname,conv) for conv in list(yolov4.modules()) if isinstance(conv,(Conv2d,BatchNorm2d))])
    convs_to_load=[(conv.defaultname,int(conv.defaultname.split('_')[-1])) for conv in list(yolov4.modules()) if isinstance(conv,Conv2d)]
    convs_to_load=sorted(convs_to_load, key=itemgetter(1))
    convs=OrderedDict([(conv[0],None) for conv in convs_to_load])

    for module in list(yolov4.modules()):
        if isinstance(module,Conv2d_Block):
            convs[module.conv.defaultname]=module.norm.defaultname

    with open(weight_path, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

        ptr = 0
        for i, (convname,bnname )in enumerate(convs.items()):
                conv = modules_dict[convname]
                mdef=module_defs[i]

                # if (1 if mdef.get('filters')==conv.num_filters  else 0)+(1 if (module_defs[i-1].get('filters',3) if i>0else 3)==conv.input_filters  else 0)+ (1 if mdef.get('size')==conv.kernel_size[0]  else 0)+ (1 if mdef.get('stride')==conv.strides[0]  else 0)==4:
                #     print(i,conv.defaultname,': ', (module_defs[i-1].get('filters',3) if i>0else 3), mdef.get('filters'),'|',conv.input_filters,conv.num_filters,'pass')
                # else:
                #     print(i,conv.defaultname,': ',(module_defs[i-1].get('filters',3) if i>0else 3), mdef.get('filters'),'|',conv.input_filters,conv.num_filters,'fail')
                #     print(conv)
                #     print(mdef)

                if bnname is not None:
                    # Load BN bias, weights, running mean and running variance
                    bn =modules_dict[bnname]
                    nb = bn.bias.numel()  # number of biases
                    # Bias
                    bn.bias.data.copy_(to_tensor(weights[ptr:ptr + nb]).view_as(bn.bias))
                    ptr += nb
                    # Weight
                    bn.weight.data.copy_(to_tensor(weights[ptr:ptr + nb]).view_as(bn.weight))
                    ptr += nb
                    # Running Mean
                    bn.running_mean.data.copy_(to_tensor(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                    ptr += nb
                    # Running Var
                    bn.running_var.data.copy_(to_tensor(weights[ptr:ptr + nb]).view_as(bn.running_var))
                    ptr += nb

                    # bn.running_mean.data.copy_(torch.tensor([0.485, 0.456, 0.406]))
                    # bn.running_var.data.copy_(torch.tensor([0.0524, 0.0502, 0.0506]))
                else:
                    # Load conv. bias
                    nb = conv.bias.numel()
                    conv_b =to_tensor(weights[ptr:ptr + nb]).view_as(conv.bias)
                    conv.bias.data.copy_(conv_b)
                    ptr += nb
                # Load conv. weights
                nw = conv.weight.numel()  # number of weights
                conv.weight.data.copy_(to_tensor(weights[ptr:ptr + nw]).view_as(conv.weight))
                ptr += nw







if __name__ == '__main__':
    #initialize network structure
    yolov4 = yolo4_body(80, 608)

    detector = YoloDetectionModel(input_shape=(3, 608, 608), output=yolov4)

    ####
    ##just use for check
    ####
    weights = OrderedDict()
    n = 0
    m = 0
    print('modual seq.\tweight seq\t ordinal\tmodual default name \tmodual friendly name\tweights name\tshape\r')
    for name, modual in yolov4.named_modules():
        if isinstance(modual, Layer):
            weights[modual.name if hasattr(modual, 'name') else name] = OrderedDict()
            n += 1
            for paraname, w in modual._parameters.items():
                if w is not None:
                    weights[modual.name if hasattr(modual, 'name') else name][
                        modual.name if hasattr(modual, 'name') else modual.name if hasattr(modual,
                                                                                           'name') else name + '/' + paraname] = to_numpy(
                        w)
                    m += 1
                    print('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\r'.format(n, m, int(modual.defaultname.split('_')[-1]),
                                                                       modual.defaultname,
                                                                       modual.name if hasattr(modual, 'name') else name,
                                                                       modual.name + '/' + paraname,
                                                                       to_numpy(w).shape if w is not None else None))
            for paraname, w in modual._buffers.items():
                if 'track' not in paraname:
                    weights[modual.name if hasattr(modual, 'name') else name][
                        modual.name if hasattr(modual, 'name') else name + '/' + paraname] = to_numpy(w)
                    m += 1
                    print('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\r'.format(n, m, int(modual.defaultname.split('_')[-1]),
                                                                       modual.defaultname, name,
                                                                       modual.name if hasattr(modual,
                                                                                              'name') else name + '/' + paraname,
                                                                       to_numpy(w).shape if w is not None else None))


    detector.save_model('Models/yolov4_mscoco.pth.tar')
    # yolo4.save_onnx('Models/yolov4_mscoco.onnx')


    load_pretrained_weight(yolov4=detector.model,cfg_path='pretrained/yolov4.cfg',weight_path='pretrained/yolov4.weights')

    import torch
    torch.save(detector.model, 'Models/pretrained_yolov4_mscoco.pth')
    detector.save_model('Models/pretrained_yolov4_mscoco.pth.tar')















