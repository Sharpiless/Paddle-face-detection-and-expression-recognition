import paddle
import time
from paddle import fluid as fluid
from paddle.fluid import layers as pfl
from yolo3tiny.imageSolver import *
import numpy as np


def sigmoid(x):
    return 1./(1+np.exp(-x))


class YOLOv3_tiny:
    def __init__(self, use_cuda=False):
        self._USE_CUDA = use_cuda

    def build(self, boxNum=64, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, regularization=None, lazy_mode=False):
        dataInput = pfl.data(name='data_input', shape=[
                             3, 416, 416], dtype='float32')
        gtbox = pfl.data(name='data_gtbox', shape=[boxNum, 4], dtype='float32')
        gtlabel = pfl.data(name='data_gtlabel', shape=[boxNum], dtype='int32')
        anchors = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
        layer0_output = _DBL(input=dataInput, num_filters=16,
                             filter_size=3, name='layer0')
        layer1_output = pfl.pool2d(
            input=layer0_output, pool_size=2, pool_type='max', pool_stride=2, name='layer1_max')
        layer2_output = _DBL(input=layer1_output,
                             num_filters=32, filter_size=3, name='layer2')
        layer3_output = pfl.pool2d(
            input=layer2_output, pool_size=2, pool_type='max', pool_stride=2, name='layer3_max')
        layer4_output = _DBL(input=layer3_output,
                             num_filters=64, filter_size=3, name='layer4')
        layer5_output = pfl.pool2d(
            input=layer4_output, pool_size=2, pool_type='max', pool_stride=2, name='layer5_max')
        layer6_output = _DBL(input=layer5_output,
                             num_filters=128, filter_size=3, name='layer6')
        layer7_output = pfl.pool2d(
            input=layer6_output, pool_size=2, pool_type='max', pool_stride=2, name='layer7_max')
        layer8_output = _DBL(input=layer7_output,
                             num_filters=256, filter_size=3, name='layer8')
        layer9_output = pfl.pool2d(
            input=layer8_output, pool_size=2, pool_type='max', pool_stride=2, name='layer9_max')
        layer10_output = _DBL(input=layer9_output,
                              num_filters=512, filter_size=3, name='layer10')
        layer11_output = pfl.pool2d(input=pfl.pad(layer10_output, paddings=[
                                    0, 0, 0, 0, 0, 1, 0, 1]), pool_size=2, pool_type='max', pool_stride=1, name='layer11_max')
        layer12_output = _DBL(input=layer11_output,
                              num_filters=1024, filter_size=3, name='layer12')
        layer13_output = _DBL(
            input=layer12_output, num_filters=256, filter_size=1, padding=0, name='layer13')
        layer14_output = _DBL(input=layer13_output,
                              num_filters=512, filter_size=3, name='layer14')
        layer15_output = pfl.conv2d(
            input=layer14_output, num_filters=18, filter_size=1, name='layer15_conv')
        # layer16_yolo -> -1 x 18 x 13 x 13
        yolo1_loss = pfl.yolov3_loss(name='yolo1_loss', x=layer15_output, gtbox=gtbox, gtlabel=gtlabel,
                                     anchors=anchors, anchor_mask=[3, 4, 5], class_num=1, ignore_thresh=0.5, downsample_ratio=32)
        # layer17_route_13
        layer18_output = _DBL(
            input=layer13_output, num_filters=128, filter_size=1, padding=0, name='layer18')
        layer19_output = pfl.expand(layer18_output, expand_times=[
                                    1, 1, 2, 2], name='layer19_upsample')
        # layer20_route_19_8
        layer20_output = pfl.concat(
            [layer19_output, layer8_output], axis=1, name='layer20_concat')
        layer21_output = _DBL(layer20_output, num_filters=256,
                              filter_size=3, name='layer21')
        layer22_output = pfl.conv2d(
            input=layer21_output, num_filters=18, filter_size=1, name='layer22_conv')
        # layer23_yolo -> -1 x 18 x 26 x 26
        yolo2_loss = pfl.yolov3_loss(name='yolo2_loss', x=layer22_output, gtbox=gtbox, gtlabel=gtlabel,
                                     anchors=anchors, anchor_mask=[0, 1, 2], class_num=1, ignore_thresh=0.5, downsample_ratio=16)
        loss = pfl.reduce_mean(pfl.elementwise_add(
            yolo1_loss, yolo2_loss), name="loss_output")
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate,
                                                  beta1=beta1,
                                                  beta2=beta2,
                                                  epsilon=epsilon,
                                                  regularization=regularization,
                                                  lazy_mode=lazy_mode)
        optimizer.minimize(loss)
        self._netOutput1, self._netOutput2 = layer15_output, layer22_output
        self._loss = loss
        self._trainExe = fluid.Executor(fluid.CUDAPlace(
            0)) if self._USE_CUDA else fluid.Executor(fluid.CPUPlace())

    def train(self, imgClass, epoches=10, batchSize=64, stopRounds=10, initialize=True, confidence_threshold=0.5, nms_threshold=0.3):
        if initialize:
            self._trainExe.run(fluid.default_startup_program())
        imgStream = imgClass.random_batch_img_stream(batchSize=batchSize)
        dataGtlabel = np.zeros([batchSize, imgClass.maxBoxNum], dtype='int32')
        itersPerEpoch = imgClass.imgNum//batchSize
        st = time.time()
        for i in range(1, epoches+1):
            for j in range(1, itersPerEpoch+1):
                dataInput, dataGtbox = next(imgStream)
                outs = self._trainExe.run(feed={'data_input': dataInput, 'data_gtbox': dataGtbox, 'data_gtlabel': dataGtlabel},
                                          fetch_list=[self._loss, self._netOutput1, self._netOutput2])
                if ((i-1)*itersPerEpoch+j) % stopRounds == 0:
                    restNum = ((itersPerEpoch-j)+(epoches-i)
                               * itersPerEpoch)*batchSize
                    trainSpeed = stopRounds*batchSize/(time.time()-st)
                    print('After iterations %d: loss = %.3lf  mAP: %.3lf  %.3lf images//s  Estimated remaining time: %.3lfs' % ((i-1)*itersPerEpoch +
                                                                                                                                j, outs[0][0], _mAP(outs[1], outs[2], dataGtbox, confidence_threshold, nms_threshold), trainSpeed, restNum/trainSpeed))
                    st = time.time()

    def save(self, modelSavePath='./infer_model'):
        fluid.io.save_inference_model(dirname=modelSavePath, feeded_var_names=['data_input'],
                                      target_vars=[self._netOutput1, self._netOutput2], executor=self._trainExe)


class Detector:
    def __init__(self, modelPath='./infer_model', USE_CUDA=False):
        self._exe = fluid.Executor(fluid.CUDAPlace(
            0)) if USE_CUDA else fluid.Executor(fluid.CPUPlace())
        self._inferenceModel, self._feedTargetNames, self._fetchTargets = fluid.io.load_inference_model(
            dirname=modelPath, executor=self._exe)

    def __call__(self, imgList, confidence_threshold=0.5, nms_threshold=0.3):
        if len(imgList) == 0:
            return [], []
        imgArrs = list(map(scale_img, imgList))
        netOutput1, netOutput2 = self._exe.run(program=self._inferenceModel,
                                               feed={self._feedTargetNames[0]: np.array(
                                                   imgArrs, dtype='float32')},
                                               fetch_list=self._fetchTargets)
        # netOutput1 -1 x 18 x 13 x 13
        # netOutput2 -1 x 18 x 26 x 26
        yoloBboxes = _get_bboxes([netOutput1, netOutput2])
        selectedBboxes = _NMS(yoloBboxes, confidence_threshold, nms_threshold)
        selectedBboxes = list(map(toBoundingBox, selectedBboxes))
        return imgArrs, selectedBboxes

    def detect(self, imgArrs, confidence_threshold=0.5, nms_threshold=0.3):
        im = np.expand_dims(imgArrs, axis=0)
        netOutput1, netOutput2 = self._exe.run(program=self._inferenceModel,
                                               feed={self._feedTargetNames[0]: np.array(
                                                   im, dtype='float32')},
                                               fetch_list=self._fetchTargets)
        # netOutput1 -1 x 18 x 13 x 13
        # netOutput2 -1 x 18 x 26 x 26
        yoloBboxes = _get_bboxes([netOutput1, netOutput2])
        selectedBboxes = _NMS(yoloBboxes, confidence_threshold, nms_threshold)
        selectedBboxes = list(map(toBoundingBox, selectedBboxes))
        return selectedBboxes


def toBoundingBox(bbox, W=416, H=416):
    if len(bbox) == 0:
        return bbox
    bbox[:, 0] -= bbox[:, 2]/2
    bbox[:, 1] -= bbox[:, 3]/2
    bbox[:, 2] += bbox[:, 0]
    bbox[:, 3] += bbox[:, 1]
    bbox[:, [0, 2]] *= W
    bbox[:, [1, 3]] *= H
    return bbox


def model_validate(imgClass, batchSize=128, modelSavePath='./infer_model', confidence_threshold=0.5, nms_threshold=0.3, USE_CUDA=False):
    imgStream = imgClass.batch_img_stream(batchSize=batchSize)
    mAP = 0.0
    while True:
        try:
            dataInput, dataGtbox = next(imgStream)
        except:
            break
        exe = fluid.Executor(fluid.CUDAPlace(
            0)) if USE_CUDA else fluid.Executor(fluid.CPUPlace())
        inferenceModel, feedTargetNames, fetchTargets = fluid.io.load_inference_model(
            dirname=modelSavePath, executor=exe)
        netOutput1, netOutput2 = exe.run(inferenceModel,
                                         feed={feedTargetNames[0]: dataInput},
                                         fetch_list=fetchTargets)
        mAP += _mAP(netOutput1, netOutput2, dataGtbox, confidence_threshold,
                    nms_threshold)*len(dataInput)/imgClass.imgNum
    print('model mAP: %.3lf' % mAP)


def _mAP(netOutput1, netOutput2, gtBboxes, confidence_threshold, nms_threshold):
    yoloBboxes = _get_bboxes([netOutput1, netOutput2])
    yoloBboxes = yoloBboxes[:, :, :-1]
    AP = []
    for yoloBbox, gtBbox in zip(yoloBboxes, gtBboxes):
        gtBbox = gtBbox[gtBbox[:, 2] > 0]
        yoloBbox = yoloBbox[yoloBbox[:, -1] > confidence_threshold]
        if len(yoloBbox) == 0:
            AP.append(0.0)
            continue
        IOU_SUM, NUM = 0, len(gtBbox)
        for bbox in gtBbox:
            IOUs = _IOU(bbox, yoloBbox,)
            maxId = np.argmax(IOUs)
            iouMax = IOUs[maxId]
            if iouMax >= nms_threshold:
                IOU_SUM += iouMax
                IOUs = _IOU(yoloBbox[maxId], yoloBbox)
                yoloBbox = yoloBbox[IOUs < nms_threshold]
            if len(yoloBbox) < 1:
                break
        if NUM > 0:
            AP.append(IOU_SUM/(NUM+len(yoloBbox)))
        else:
            AP.append(1.0/(len(yoloBbox)+1))
    return np.mean(AP)


def _DBL(input, num_filters, filter_size, padding=1, name=None):
    conv = pfl.conv2d(input=input, num_filters=num_filters, filter_size=filter_size,
                      padding=padding, name=(name+'_conv2d') if name else None)
    bn = pfl.batch_norm(input=conv, name=(name+'_conv2d') if name else None)
    act = pfl.leaky_relu(bn, name=(name+'_act') if name else None)
    return act


def _get_bboxes(netOutputs, anchors=None, downsample_ratio=[32, 16]):
    anchors = np.array([[[81, 82], [135, 169], [344, 319]], [[10, 14], [23, 27], [
                       37, 58]]], dtype='float32') if anchors == None else anchors
    for i in range(anchors.shape[0]):
        anchors[i] /= downsample_ratio[i]
    bboxes = []
    for netOutput, anchor in zip(netOutputs, anchors):
        N, C, H, W = netOutput.shape
        netOutput[:, [0, 1, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17], :, :] = sigmoid(
            netOutput[:, [0, 1, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17], :, :])
        cx, cy = np.array([i for i in range(W)], dtype='float32').reshape(
            [1, -1]), np.array([i for i in range(H)], dtype='float32').reshape(-1, 1)
        netOutput[:, [0, 6, 12], :, :] += cx
        netOutput[:, [1, 7, 13], :, :] += cy
        netOutput = netOutput.reshape(-1, C, H*W).swapaxes(1, 2)
        netOutput[:, :, [2, 8, 14]] = anchor[:, 0] * \
            np.exp(netOutput[:, :, [2, 8, 14]], dtype='float32')
        netOutput[:, :, [3, 9, 15]] = anchor[:, 1] * \
            np.exp(netOutput[:, :, [3, 9, 15]], dtype='float32')
        netOutput = np.concatenate(
            [i for i in np.split(netOutput, 3, axis=2)], axis=1)
        netOutput[:, :, [0, 2]] /= W
        netOutput[:, :, [1, 3]] /= H
        bboxes.append(netOutput)
    return np.concatenate(bboxes, axis=1).astype('float32')


def _NMS(yoloBboxes, confidence_threshold, nms_threshold):
    selectedBboxes = []
    for bboxesItem in yoloBboxes:
        selected = []
        # ignore the bboxes whose confidence is less than confidence_threshold
        bboxesItem = bboxesItem[bboxesItem[:, 4] > confidence_threshold]
        bboxesItem[:, 4] *= bboxesItem[:, 5]
        bboxesItem = bboxesItem[:, :-1]
        bboxesItem = bboxesItem[np.argsort(-bboxesItem[:, -1])]
        while len(bboxesItem) > 0:
            selectedBbox = bboxesItem[0]
            bboxesItem = bboxesItem[1:]
            selected.append(selectedBbox)
            IOUs = _IOU(selectedBbox, bboxesItem)
            bboxesItem = bboxesItem[IOUs < nms_threshold]
        selectedBboxes.append(np.array(selected, dtype='float32'))
    return selectedBboxes


def _IOU(bboxObj, bboxes):
    def _iou(bbox):
        sx1, sy1, ex1, ey1 = bboxObj[0]-bboxObj[2]/2, bboxObj[1] - \
            bboxObj[3]/2, bboxObj[0]+bboxObj[2]/2, bboxObj[1]+bboxObj[3]/2
        sx2, sy2, ex2, ey2 = bbox[0]-bbox[2]/2, bbox[1] - \
            bbox[3]/2, bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
        sx, sy = max(sx1, sx2), max(sy1, sy2)
        ex, ey = min(ex1, ex2), min(ey1, ey2)
        if ex <= sx or ey <= sy:
            return 0.
        I = (ex-sx)*(ey-sy)
        U = (ex1-sx1)*(ey1-sy1) + (ex2-sx2)*(ey2-sy2) - I
        return I/U
    ious = map(_iou, bboxes)
    return np.array(list(ious), dtype='float32')
