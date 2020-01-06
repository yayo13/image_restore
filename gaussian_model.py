import numpy as np
import cv2
import math
import random
from sklearn import mixture
from scipy.stats import multivariate_normal
import pdb

PI = 3.14159265

class gauss_mix_model(object):
    def __init__(self):
        self._color = ((0,255,0), (255,0,0), (0,0,255))

    def modeling(self, hist_, nclass, flag):
        if flag == 'sklearn':
            return self.modeling_sklearn(hist_, nclass)
        return self.modeling_my(hist_, nclass)

    def gen_sample_from_hist(self, hist_):
        samples = []
        hist_norm = []
        
        maxVal = hist_[0,0]
        minVal = hist_[0,0]
        for i in range(len(hist_)):
            if hist_[i,0] > maxVal:
                maxVal = hist_[i,0]
            if hist_[i,0] < minVal:
                minVal = hist_[i,0]

        for i in range(len(hist_)):
            num_ = hist_[i,0]
            for ii in range(num_):
                samples.append(i)
            hist_norm.append(1.0*(num_-minVal)/(maxVal-minVal))
        random.shuffle(samples)
        # pdb.set_trace()
        samples_np = np.zeros((len(samples),))
        for i in range(len(samples)):
            samples_np[i] = samples[i]
        # pdb.set_trace()
        return samples_np, hist_norm

    def modeling_sklearn(self, sample_, nclass):
        gmm = mixture.GaussianMixture(n_components=nclass, covariance_type='full').fit(sample_)
        labels = gmm.predict(sample_)
        return labels

    def modeling_my(self, sample_, nclass):
        hist_sample, hist_norm = self.gen_sample_from_hist(sample_)
        # initial gmm params
        mu_ecls    = np.zeros((nclass,1), dtype=np.float32)
        sigma_ecls = np.zeros((nclass,1), dtype=np.float32)
        pi_ecls    = np.zeros((nclass,1), dtype=np.float32)

        nPerClass = 255.0/nclass
        for i in range(nclass):
            pi_ecls[i,0]    = 1.0/nclass
            mu_ecls[i,0]    = nPerClass/2.0 + i*nPerClass
            sigma_ecls[i,0] = 200
            # print 'pi[%d] = %.4f'%(i, pi_ecls[i,0])
            # print 'mu_ecls[%d] = %.4f'%(i, mu_ecls[i,0])
            # print 'sigma_ecls[%d] = %.4f'%(i, sigma_ecls[i,0])
        
        gamma = np.zeros((nclass, len(hist_sample)), dtype=np.float32)
        for echo in range(10):
            print '***************** echo: %d *****************'%(echo+1)

            txt_cls_prob  = 'prob:'
            txt_cls_mu    = 'mu:'
            txt_cls_sigma = 'sigma:'
            for cls_index in range(nclass):
                txt_cls_prob += ' %.3f '%pi_ecls[cls_index,0]
                txt_cls_mu += ' %.3f '%mu_ecls[cls_index,0]
                txt_cls_sigma += ' %.3f '%sigma_ecls[cls_index,0]
            print txt_cls_prob
            print txt_cls_mu
            print txt_cls_sigma

            # E STEP
            # pdb.set_trace()
            gamma_sum = np.zeros((len(hist_sample),1), dtype=np.float32)

            for cls_index in range(nclass):
                gamma[cls_index, :] = pi_ecls[cls_index,0]*multivariate_normal.pdf(hist_sample, mean=mu_ecls[cls_index,0], cov=sigma_ecls[cls_index,0])
            # pdb.set_trace()
            for x_index in range(len(hist_sample)):
                gamma_sum[x_index,0] = np.sum(gamma[:,x_index])
                if gamma_sum[x_index,0] >= 0.00001:
                    gamma[:,x_index] /= gamma_sum[x_index,0]
                    
            # pdb.set_trace()
            # M STEP
            num_ecls   = np.zeros((nclass,1), dtype=np.float32)
            mu_ecls    = np.zeros((nclass,1), dtype=np.float32)
            sigma_ecls = np.zeros((nclass,1), dtype=np.float32)
            pi_ecls    = np.zeros((nclass,1), dtype=np.float32)
            for cls_index in range(nclass):
                num_ecls[cls_index,0] = np.sum(gamma[cls_index])
                print 'num_ecls[%d] = %.4f'%(cls_index,num_ecls[cls_index,0])

                for x_index in range(len(hist_sample)):
                    mu_ecls[cls_index,0] += gamma[cls_index][x_index] * hist_sample[x_index]
                mu_ecls[cls_index,0] /= num_ecls[cls_index,0]

                for x_index in range(len(hist_sample)):
                    sigma_ecls[cls_index,0] += gamma[cls_index][x_index] * math.pow(hist_sample[x_index]-mu_ecls[cls_index,0],2)
                # pdb.set_trace()
                sigma_ecls[cls_index,0] /= num_ecls[cls_index,0]

                pi_ecls[cls_index,0] = num_ecls[cls_index,0] / len(hist_sample)
            
            # txt_cls_prob  = 'prob:'
            # txt_cls_mu    = 'mu:'
            # txt_cls_sigma = 'sigma:'
            # for cls_index in range(nclass):
            #     txt_cls_prob += ' %.3f '%pi_ecls[cls_index,0]
            #     txt_cls_mu += ' %.3f '%mu_ecls[cls_index,0]
            #     txt_cls_sigma += ' %.3f '%sigma_ecls[cls_index,0]
            # print txt_cls_prob
            # print txt_cls_mu
            # print txt_cls_sigma

            models = []
            for cls_index in range(nclass):
                models.append((mu_ecls[cls_index,0], sigma_ecls[cls_index,0], pi_ecls[cls_index,0]))
            hist_img = self.draw_hist(hist_norm, models)
            cv2.imshow('hist', hist_img)
            cv2.waitKey(10)
        return models


    def gauss(self, x, mu_, sigma_):
        # bb = -math.pow(x-mu_,2)/(2*sigma_*sigma_)
        # pdb.set_trace()
        return 1.0/(math.sqrt(2*PI) * sigma_) * math.exp(-math.pow(x-mu_,2)/(2*sigma_*sigma_))

    def draw_hist(self, hist_, models_):
        # draw gray hist
        hist_img = np.zeros((256, 256, 3), dtype=np.uint8)
        rate = 256-10

        pos_y_last = int(255 - hist_[0]*rate)
        for level in range(255):
            pos_y = int(255 - hist_[level+1]*rate)
            cv2.line(hist_img, (level, pos_y_last), (level+1, pos_y), (255,255,255))
            pos_y_last = pos_y
        
        # draw pdf
        y = np.zeros((256,1),dtype=np.float32)
        max_y = 0
        for model_idx in range(len(models_)):
            for x in range(256):
                y[x,0] += models_[model_idx][2]*multivariate_normal.pdf(x, models_[model_idx][0], models_[model_idx][1])
        for x in range(256):
            max_y = max(max_y, y[x,0])
        for x in range(256):
            y[x,0] = y[x,0]/max_y

        pos_y_ = int(255 - y[0,0]*rate)
        for x in range(255):
            pos_y = int(255 - y[x+1,0]*rate)
            cv2.line(hist_img, (x, pos_y_), (x+1, pos_y), self._color[0])
            pos_y_ = pos_y
        return hist_img

    def test_em(self):
        # generate data
        NUM_DATA = 50000
        PARAM_DATA = ((30, 100, 0.3), (80, 100, 0.5), (120, 100, 0.2))
        P_DATA = np.zeros((256, 3))
        for i in range(len(PARAM_DATA)):
            for x in range(256):
                # P_DATA[x,i] = self.gauss(x, PARAM_DATA[i][0], PARAM_DATA[i][1])
                P_DATA[x,i] = multivariate_normal.pdf(x, PARAM_DATA[i][0], PARAM_DATA[i][1])
        
        buff = np.zeros((256,1), dtype=np.uint32)
        for i in range(len(PARAM_DATA)):
            num = NUM_DATA * PARAM_DATA[i][2]
            for x in range(256):
                buff[x,0] += int(num*P_DATA[x,i])
        # pdb.set_trace()

        self.modeling_my(buff, 3)