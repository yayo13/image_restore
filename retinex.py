#! /usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import sys
import numpy as np
import random
import pdb

class retinex(object):
    def repair(self, img, sigma, type):
        if type == 0:
            return self.repair_SSR(img, sigma)
        if type == 1:
            return self.repair_MSR(img, sigma)
        if type == 2:
            return self.repair_MSRCR(img, sigma, 5, 25, 125, 46, 0.01, 0.8)

    def repair_SSR(self, img, sigma):
        # 单尺度
        # 其实感觉跟形态学顶帽差不多的意思
        temp = cv2.GaussianBlur(img, (0,0), sigma)
        gaussian = np.where(temp == 0, 0.01, temp)
        retinex = np.log10(img+0.01) - np.log10(gaussian)
        return retinex

    def repair_MSR(self, img, sigma_list):
        # 多尺度
        retinex = np.zeros_like(img*1.0)
        for sigma in sigma_list:
            retinex += self.repair_SSR(img, sigma)
        retinex = retinex / len(sigma_list)
        return retinex

    def repair_MSRCR(self, img, sigma_list, gain, offset, alpha, beta, low_clip, high_clip):
        # 带颜色恢复的多尺度
        img = np.float64(img) + 1.0
        img_msr = self.repair_MSR(img, sigma_list)
        img_color = self.color_restor(img, alpha, beta)
        img_msrcr = gain * (img_msr * img_color + offset)

        for ch in range(img_msrcr.shape[2]):
            img_msrcr[:,:,ch] = (img_msrcr[:,:,ch] - np.min(img_msrcr[:,:,ch])) / \
                                (np.max(img_msrcr[:,:,ch]) - np.min(img_msrcr[:,:,ch]))*255
        img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
        img_msrcr = self.color_balance(img_msrcr, low_clip, high_clip)
        return img_msrcr

 
    def color_restor(self, img, alpha, beta):
        img_sum = np.sum(img, axis=2, keepdims=True)
        color_res = beta * (np.log10(alpha*img) - np.log10(img_sum))
        return color_res


    def color_balance(self, img, low, high):
        area = img.shape[0]*img.shape[1]
        for ch in range(img.shape[2]):
            unique, counts = np.unique(img[:,:,ch], return_counts=True)
            current = 0
            low_val = 0
            high_val = 0
            for u, c in zip(unique, counts):
                if float(current) / area < low:
                    low_val = u
                if float(current) / area < high:
                    high_val = u
                current += c
            img[:,:,ch] = np.maximum(np.minimum(img[:,:,ch], high_val), low_val)
        return img


def calc_saturation(diff, slope, limit):
    ret = diff*slope
    if ret > limit:
        ret = limit
    elif ret < -limit:
        ret = -limit
    return ret

def auto_color_equaliz(img, slope=10, limit=1000, samples=500):
    wid = img.shape[1]
    hei = img.shape[0]

    cary = []
    for i in range(0, samples):
        _x = random.randint(0, wid)%wid
        _y = random.randint(0, hei)%hei
        dict_ = {"x":_x, "y":_y}
        cary.append(dict_)

    res = np.zeros((hei, wid, 3), dtype=np.float)
    b_max = sys.float_info.min
    b_min = sys.float_info.max
    g_max = sys.float_info.min
    g_min = sys.float_info.max
    r_max = sys.float_info.min
    r_min = sys.float_info.max
    for row in range(hei):
        for col in range(wid):
            b_ = img[row, col, 0]
            g_ = img[row, col, 1]
            r_ = img[row, col, 2]

            b_rscore_sum = 0
            g_rscore_sum = 0
            r_rscore_sum = 0
            denominator = 0

            for _dict in cary:
                _x = _dict["x"]
                _y = _dict["y"]

                dist = np.sqrt(np.square(_x-col)+np.square(_y-row))
                if dist < hei/5.0:
                    continue

                _sb = img[_y, _x, 0]
                _sg = img[_y, _x, 1]
                _sr = img[_y, _x, 2]

                b_rscore_sum += calc_saturation(int(b_) - int(_sb), slope, limit) / dist
                g_rscore_sum += calc_saturation(int(g_) - int(_sg), slope, limit) / dist
                r_rscore_sum += calc_saturation(int(r_) - int(_sr), slope, limit) / dist

                denominator += limit/dist

            b_rscore_sum /= denominator
            g_rscore_sum /= denominator
            r_rscore_sum /= denominator

            res[row, col, 0] = b_rscore_sum
            res[row, col, 1] = g_rscore_sum
            res[row, col, 2] = r_rscore_sum

            b_max = max(b_max, b_rscore_sum)
            b_min = min(b_min, b_rscore_sum)
            g_max = max(g_max, g_rscore_sum)
            g_min = min(g_min, g_rscore_sum)
            r_max = max(r_max, r_rscore_sum)
            r_min = min(r_min, r_rscore_sum)

    for row in range(hei):
        for col in range(wid):
            res[row, col, 0] = (res[row, col, 0] - b_min)*255 / (b_max-b_min)
            res[row, col, 1] = (res[row, col, 1] - g_min)*255 / (g_max-g_min)
            res[row, col, 2] = (res[row, col, 2] - r_min)*255 / (r_max-r_min)
    return res.astype(np.uint8)

if __name__ == '__main__':
    # PATH_IMG_FILE = 'imgs/03.jpg'
    PATH_IMG_FILE = '/media/yuan/备份数据/DATA/CIDI/360/cidi20191204/20191205101333660_076432_HB.jpg'
    img = cv2.imread(PATH_IMG_FILE)

    reti = retinex()
    img_ = reti.repair(img, (65, 180, 200), 2)
    # img_ = reti.repair(img, 200, 0)
    # img__ = auto_color_equaliz(img)
    cv2.imshow('src', img)
    cv2.imshow('repaired', img_)
    # cv2.imshow('repaired2', img__)
    cv2.waitKey(0)