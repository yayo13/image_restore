#! /usr/bin/env python
# -*- coding:utf-8 -*-

'''
He K, Sun J, Tang X. 
Single image haze removal using dark channel prior[J]. 
IEEE transactions on pattern analysis and machine intelligence, 2010, 33(12): 2341-2353.
'''

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splina
import cv2
# import pdb

class dehazer(object):
    def __init__(self, win=9, ap=0.001, omiga=0.95, max_t=0.05):
        self._ap = ap
        self._omiga = omiga
        self._max_t = max_t
        self._win_size = win
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win,win))

    def dehaze(self, img):
        img_float = img.astype(np.float32)/255.0

        dark_img = self.dark_channel(img_float)
        atmos = self.atmospheric_light(img_float, dark_img)
        trans = self.transmission(img_float, atmos)
        trans = self.guide_filter(img_float, trans)
        trans = np.maximum(trans, self._max_t)
        dehazed = np.zeros(img.shape, img_float.dtype)
        for ch in range(3):
            dehazed[:,:,ch] = (img_float[:,:,ch] - atmos[0,ch]) / trans + atmos[0,ch]
        dehazed = np.maximum(dehazed, 0)
        dehazed = np.minimum(dehazed, 1)
        dehazed *= 255

        cv2.imshow('dark', dark_img)
        cv2.imshow('tran', trans)
        return dehazed.astype(np.uint8)

    def dark_channel(self, img, use_win=True):
        dark_img = img.min(axis=2)
        if use_win:
            dark_img = cv2.erode(dark_img, self._kernel)
        return dark_img

    def atmospheric_light(self, srcimg, darkimg):
        vec_sz = darkimg.shape[0]*darkimg.shape[1]
        dark_vec = darkimg.reshape(vec_sz,1)
        src_vec = srcimg.reshape(vec_sz, 3)

        num_ap = int(vec_sz*self._ap)
        indx = dark_vec[:,0].argsort()[vec_sz-num_ap:]

        atmos = np.zeros((1,3), dtype=np.float32)
        for i in range(num_ap):
            atmos += src_vec[indx[i]]
        atmos /= num_ap
        return atmos

    def transmission(self, srcimg, atmos):
        img_ = np.zeros(srcimg.shape, srcimg.dtype)
        for i in range(3):
            img_[:,:,i] = srcimg[:,:,i] / atmos[0,i]

        trans = 1 - self._omiga*self.dark_channel(img_, False)
        # trans = self.soft_matting(srcimg, trans)
        return trans

    def soft_matting(self, srcimg, trans):
        epsilon = 10**-8
        lambda_ = 10**-4
        
        window_size = 3
        num_window_pixels = window_size * window_size
        inv_num_window_pixels = 1.0 / num_window_pixels

        im_height = srcimg.shape[0]
        im_width  = srcimg.shape[1]
        num_image_pixels = im_height*im_width

        # matting_laplacian = np.zeros((num_image_pixels, num_image_pixels))
        matting_laplacian = sparse.lil_matrix((num_image_pixels, num_image_pixels))
        for row in range(window_size/2, im_height-window_size/2):
            for col in range(window_size/2, im_width-window_size/2):
                window_indice = (row-window_size/2, col-window_size/2, 
                                 row+window_size/2+1, col+window_size/2+1)

                window = srcimg[window_indice[0]:window_indice[2], 
                                window_indice[1]:window_indice[3],:]
                
                window_flat = window.reshape(num_window_pixels, 3)
                window_mean = np.mean(window_flat, 0).reshape(1,3) #1*3
                window_conv = np.cov(window_flat.T, bias=True) #3*3
                window_inv_cov = window_conv + (epsilon / num_window_pixels)*np.eye(3)
                window_inv_cov = np.linalg.inv(window_inv_cov)

                for sub_row_1 in range(window_indice[0], window_indice[2]):
                    for sub_col_1 in range(window_indice[1], window_indice[3]):
                        matting_laplace_row = sub_row_1*im_width + sub_col_1
                        
                        for sub_row_2 in range(window_indice[0], window_indice[2]):
                            for sub_col_2 in range(window_indice[1], window_indice[3]):
                                matting_laplace_col = sub_row_2*im_width + sub_col_2
                                
                                ker_delta = 0
                                if matting_laplace_row == matting_laplace_col:
                                    ker_delta = 1
                                
                                row_pixel_var = srcimg[sub_row_1, sub_col_1, :] - window_mean
                                col_pixel_var = srcimg[sub_row_2, sub_col_2, :] - window_mean
                                
                                matting_laplacian[matting_laplace_row, matting_laplace_col] += ker_delta - inv_num_window_pixels*(1+np.dot(np.dot(row_pixel_var,window_inv_cov),col_pixel_var.T))[0,0]

        trans_flat = trans.reshape(num_image_pixels, 1)
        matting_laplacian_inv = matting_laplacian + (lambda_*sparse.eye(num_image_pixels))
        matting_laplacian_inv = splina.inv(matting_laplacian_inv.tocsc())
        refined_trans_flat = np.dot(lambda_*trans_flat, matting_laplacian_inv)
        refined_trans = refined_trans_flat.reshape(im_height, im_width)

        return refined_trans

    def guide_filter(self, srcimg, trans, radius=11, eps=0.01):
        srcimg_gray = srcimg
        if len(srcimg.shape) == 3:
            srcimg_gray = cv2.cvtColor(srcimg, cv2.COLOR_BGR2GRAY) 

        mean_gui = cv2.boxFilter(srcimg_gray, -1, (radius,radius), normalize=True)
        mean_fil = cv2.boxFilter(trans, -1, (radius,radius), normalize=True)
        mean_gf  = cv2.boxFilter(trans*srcimg_gray, -1, (radius,radius), normalize=True)
            
        cov_gf   = mean_gf - mean_gui*mean_fil
        mean_gui_gui = cv2.boxFilter(srcimg_gray*srcimg_gray, -1, (radius,radius), normalize=True)
        var_gui = mean_gui_gui - mean_gui * mean_gui

        a = cov_gf / (var_gui + eps)
        b = mean_fil - a * mean_gui

        mean_a = cv2.boxFilter(a, -1, (radius,radius), normalize=True)
        mean_b = cv2.boxFilter(b, -1, (radius,radius), normalize=True)
        trans_ = mean_a * srcimg_gray + mean_b
        return trans_

    
def main():
    IMG_FILE = 'imgs/09.bmp'
    src_img = cv2.imread(IMG_FILE)

    recover = dehazer()
    recovered = recover.dehaze(src_img)

    cv2.imshow('src', src_img)
    cv2.imshow('dehazed', recovered)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()