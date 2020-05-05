import os
import re
import random
import cv2
import numpy as np


class WIDER:
    def __init__(self, imgPath, imgBboxGt, maxBoxNum=64):
        with open(imgBboxGt) as f:
            txt = ''.join(f.readlines())
            imgPathArray = np.array(re.findall(r'(.*?\.jpg)\n', txt))
            numArray = np.array(re.findall(r'\.jpg\n(\d+)', txt), dtype='int')
            bboxArray = np.array(re.findall(
                r'(\d+) (\d+) (\d+) (\d+).*?\n', txt), dtype='float')
        numArray, bboxArray = self._expansion_rows_to_maxBoxNum(
            bboxArray, numArray, maxBoxNum)
        self.imgPath, self.imgPathArray, self.bboxArray = imgPath, imgPathArray, bboxArray
        self.maxBoxNum, self.imgNum = maxBoxNum, len(imgPathArray)

    def random_batch_img_stream(self, batchSize=8):
        idList = [i for i in range(self.imgNum)]
        while True:
            samples = random.sample(
                idList, batchSize) if batchSize > 0 else random.sample(idList, len(idList))
            yield self._scale_img_bbox_array(self.imgPath, self.imgPathArray[samples], self.bboxArray[samples])

    def batch_img_stream(self, batchSize=8):
        for i in range(self.imgNum//batchSize+1):
            yield self._scale_img_bbox_array(self.imgPath, self.imgPathArray[i*batchSize:(i+1)*batchSize], self.bboxArray[i*batchSize:(i+1)*batchSize])

    def _expansion_rows_to_maxBoxNum(self, bboxArray, numArray, maxBoxNum):
        obj = 0
        for i in range(numArray.shape[0]):
            if maxBoxNum > numArray[i]:
                obj += numArray[i]
                bboxArray = np.insert(bboxArray, obj=obj, values=np.array(
                    [[0, 0, 0, 0] for j in range(maxBoxNum-numArray[i])]), axis=0)
                obj += maxBoxNum-numArray[i]
            else:
                obj += maxBoxNum
                bboxArray = np.delete(bboxArray, obj=[j for j in range(
                    obj, obj+numArray[i]-maxBoxNum)], axis=0)
                numArray[i] = maxBoxNum
        bboxArray = bboxArray.reshape(-1, maxBoxNum, 4)
        return numArray, bboxArray

    def _scale_img_bbox_array(self, path, imgPathArray, bboxArray, IMG_WH=416):
        imgArray = []
        for i in range(imgPathArray.shape[0]):
            img = cv2.imread(os.path.join(path, imgPathArray[i]))
            h, w, c = img.shape
            dh, dh_e, dw, dw_e = 0, 0, 0, 0
            if w > h:
                dh = (w-h)//2
                dh_e = w-h-dh-dh
            else:
                dw = (h-w)//2
                dw_e = h-w-dw-dw
            img = cv2.copyMakeBorder(
                img, dh, dh+dh_e, dw, dw+dw_e, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            imgArray.append(cv2.resize(img, (IMG_WH, IMG_WH))/255.0)
            bboxArray[i, :, 2] = bboxArray[i, :, 2]/(w+dw+dw+dw_e)
            bboxArray[i, :, 3] = bboxArray[i, :, 3]/(h+dh+dh+dh_e)
            bboxArray[i, :, 0] = (bboxArray[i, :, 0]+dw) / \
                (w+dw+dw+dw_e) + bboxArray[i, :, 2]/2
            bboxArray[i, :, 1] = (bboxArray[i, :, 1]+dh) / \
                (h+dh+dh+dh_e) + bboxArray[i, :, 3]/2

        return np.array(imgArray, dtype='float32').swapaxes(1, 3).swapaxes(2, 3), np.array(bboxArray, dtype='float32')


def draw_bbox(img, bboxes, color=(0, 0, 255)):
    if type(img) == str:
        img = cv2.imread(img)
    else:
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            img = img.swapaxes(0, 2).swapaxes(0, 1)
    if np.max(img) <= 1:
        img *= 255
    img = img.astype('uint8')
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        img = cv2.rectangle(img, (int(round(x1)), int(round(y1))),
                            (int(round(x2)), int(round(y2))), color, 2)

    return img

def recover_img(img):
    if type(img) == str:
        img = cv2.imread(img)
    else:
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            img = img.swapaxes(0, 2).swapaxes(0, 1)
    if np.max(img) <= 1:
        img *= 255
    img = img.astype('uint8')

    return img


def scale_img(img, IMG_WH=416):
    if type(img) == str:
        img = cv2.imread(img)
    else:
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            img = img.swapaxes(0, 2).swapaxes(0, 1)
    h, w, c = img.shape
    dh, dh_e, dw, dw_e = 0, 0, 0, 0
    if w > h:
        dh = (w-h)//2
        dh_e = w-h-dh-dh
    else:
        dw = (h-w)//2
        dw_e = h-w-dw-dw
    img = cv2.copyMakeBorder(img, dh, dh+dh_e, dw,
                             dw+dw_e, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    imgArray = cv2.resize(img, (IMG_WH, IMG_WH))/255.0
    return np.array(imgArray, dtype='float32').swapaxes(0, 2).swapaxes(1, 2)
