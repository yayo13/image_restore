import cv2
import numpy as np
import math
from gaussian_model import gauss_mix_model
from scipy.stats import multivariate_normal
from pulp import *
import pdb

class seg_illum_img(object):
    def __init__(self):
        self._gmm = gauss_mix_model()
        self._color = ((0,255,0), (0,255,255), (0,0,255))
        self._color = np.asarray(self._color)

    def segment(self, img):
        # histogram
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        self._hist = cv2.calcHist([img_gray], [0], None, [256], [0,255])

        # mix gaussian model
        seg_level = self._gmm.modeling(self._hist, 3, 'my')

        # segment image
        img_mask = self.make_mask(img_gray, seg_level)
        img_mask = cv2.medianBlur(img_mask, 5)
        img_mixed = self.mix_mask(img_gray, img_mask)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_ = self.hist_equalize(img_hsv[:,:,2], img_mask, 3)

        img_hsv[:,:,2] = img_
        img_ = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        # # use octm
        # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # img_octm = octm(img_hsv[:,:,2], img_mask)
        # img_hsv[:,:,2] = img_octm
        # img_octm = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('seged', img_mixed)
        cv2.imshow('equal', img_)
        # cv2.imshow('octm', img_octm)
        cv2.imshow('src', img)
        cv2.imshow('gray', img_gray)
        cv2.waitKey(0)

    def make_mask(self, gray_img, level):
        mask = np.zeros((gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
        prob_ = np.zeros((256,len(level)), dtype=np.float32)
        mask_ = np.zeros((256,), dtype=np.uint8)
        for x in range(256):
            for cls in range(len(level)):
                prob_[x,cls] = level[cls][2]*multivariate_normal.pdf(x, level[cls][0], level[cls][1])
            mask_[x] = np.argmax(prob_[x,:])
        
        for row in range(gray_img.shape[0]):
            for col in range(gray_img.shape[1]):
                mask[row, col] = mask_[gray_img[row, col]]
        # mix_mask_ = self.mix_mask(gray_img, mask)
        # cv2.imshow('src_mask', mix_mask_)

        # mask = self.modify_mask(mask, gray_img, level)
        return mask

    def modify_mask(self, mask, gray, hists):
        # only modify dark level
        mean_gray_dark = hists[0][0]
        pos_mean_gray = np.where(gray == int(mean_gray_dark+0.5))
        mask_ = mask.copy()
        for row in range(gray.shape[0]):
            for col in range(gray.shape[1]):
                if mask[row,col] == 1:
                    diff_gray = abs(mean_gray_dark - gray[row,col])
                    diff_dist = max(gray.shape[0], gray.shape[1])

                    for index in range(pos_mean_gray[0].shape[0]):
                        x = pos_mean_gray[0][index]
                        y = pos_mean_gray[1][index]
                        _dist = math.sqrt(math.pow(x-col,2)+math.pow(y-row,2))
                        diff_dist = min(diff_dist, _dist)
                    # if row==136 and col == 87:
                    #     pdb.set_trace()
                    if diff_dist <= 10 and diff_gray <= 50:
                        mask_[row,col] = 0
        return mask_


    def mix_mask(self, img, mask):
        mixed = img.copy()
        if len(img.shape) == 2:
            mixed = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        weight = 0.6
        mmax = np.ones((3,))*255
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                value = (1-weight)*mixed[row,col] + weight*self._color[int(mask[row,col])]
                value = np.minimum(value, mmax)
                mixed[row,col] = value.astype(np.uint8)
        return mixed


    def hist_equalize(self, gray_img, mask, level):
        hist = np.zeros((256, level), dtype=np.float32)
        area = [0]*level
        for row in range(gray_img.shape[0]):
            for col in range(gray_img.shape[1]):
                hist[gray_img[row,col], mask[row,col]] += 1.0
                area[mask[row,col]] += 1
        for i in range(level):
            hist[:,i] /= area[i]

        for i in range(level):
            for ii in range(255):
                hist[ii+1, i] += hist[ii, i]
        
        equalized = np.zeros((gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
        for row in range(gray_img.shape[0]):
            for col in range(gray_img.shape[1]):
                if mask[row, col] == 1 or mask[row, col] == 2:
                    equalized[row, col] = gray_img[row,col]
                else:
                    value = hist[gray_img[row,col], mask[row,col]]
                    equalized[row, col] = max(min(255, int(value*255)),0)
        return equalized

def octm(img, mask):
    pHist = np.zeros((256,), dtype=np.float32)
    count = 0
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if mask[row,col] == 0:
                pHist[int(img[row,col])] += 1
                count += 1
    pHist /= count

    prob = LpProblem('octm', LpMaximize)
    # define dicision variable & target function
    d = 2
    delta = set()
    p = dict()
    for i in range(256):
        var_name = 'x%d'%i
        delta.add(var_name) 
        p[var_name] = pHist[i]
    var = LpVariable.dicts("deltas", delta, lowBound=1.0/2, upBound=255, cat='Continuous')

    prob += lpSum([p[i]*var[i] for i in delta])

    prob += lpSum([var[i] for i in delta]) <= 255

    prob.writeLP('octm_prob.lp')
    prob.solve()

    s = np.zeros((256,), dtype=np.float32)
    for v in prob.variables():
        var_index = int(v.name[8:])
        s[var_index] = v.varValue
        print 's[%d]=%.3f'%(var_index, v.varValue)
    
    print 'maxium = ',value(prob.objective),'\n'

    # map image
    T = np.zeros((256,), dtype=np.uint8)
    mapped = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    T[0] = min(255, int(s[0]+0.5))
    for i in range(1,255):
        for j in range(i+1):
            T[i] += min(255, int(s[j]+0.5))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if mask[row,col] == 0:
                mapped[row,col] = T[img[row,col]]
            else:
                mapped[row,col] = img[row,col]
    return mapped


if __name__ == '__main__':
    PATH_IMG_FILE = 'imgs/01.jpg'
    # PATH_IMG_FILE = 'H:/DATA/CIDI/360/cidi20191204/20191204201838734_043221_HB.jpg'
    img = cv2.imread(PATH_IMG_FILE)

    seg = seg_illum_img()
    seg.segment(img)
    # gmm = gauss_mix_model()
    # gmm.test_em()