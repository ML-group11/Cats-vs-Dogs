# -*- coding: utf-8 -*-

'''
词袋模型BOW+SVM 目标识别

以狗和猫数据集二分类为例
如果是狗 返回True
如果是猫 返回False
'''
import numpy as np
import cv2

import tkinter as tk
from tkinter import filedialog
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans
import math


class BOW(object):

    def __init__(self, ):
        # 创建一个SIFT对象  用于关键点提取
        self.feature_detector = cv2.xfeatures2d.SIFT_create()
        # 创建一个SIFT对象  用于关键点描述符提取
        self.descriptor_extractor = cv2.xfeatures2d.SIFT_create()
        self.voc_x = []
        # source images and lables
        self.X, self.y = [], []

    def path(self, cls, i):
        '''
        用于获取图片的全路径
        '''
        return '%s/%s/%d.jpg' % (self.train_path, cls, i)

    def Fit(self, train_path):
        '''
        开始训练

        args：
            train_path：训练集图片路径  我们使用的数据格式为 train_path/Dog/i.jpg  train_path/Cat/i.jpg
            k：k-means参数k
        '''
        self.train_path = train_path

        # FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
        flann_params = dict(algorithm=1, tree=5)
        flann = cv2.FlannBasedMatcher(flann_params, {})

        pos = 'dogs'
        neg = 'cats'

        # 指定用于提取词汇字典的样本数
        length = 50
        # 合并特征数据  每个类从数据集中读取length张图片(length个狗,length个猫)，通过聚类创建视觉词汇
        for i in range(length):
            self.voc_x.extend(self.sift_descriptor_extractor(self.path(pos, i)))
            self.voc_x.extend(self.sift_descriptor_extractor(self.path(neg, i)))
        self.voc_x = np.array(self.voc_x)
        print('=======================')
        print(self.voc_x)
        print(self.voc_x.shape)


        SSE_mean = []
        SSE_std = []
        K_range = range(5, 50, 5)
        for k in K_range:
            gmm = KMeans(n_clusters=k)
            kf = KFold(n_splits=5)
            m = 0;
            v = 0
            for train, test in kf.split(self.voc_x):
                gmm.fit(train.reshape(-1,1))
                cost =-gmm.score(test.reshape(-1,1))
                m = m + cost
                v = v + cost*cost
            SSE_mean.append(m/5)
            SSE_std.append(math.sqrt(v/5-(m/5)*(m/5)))
        plt.rc('font', size=18)
        plt.errorbar(K_range, SSE_mean, yerr=SSE_std, xerr=None, fmt='bx-')
        plt.ylabel('cost')
        plt.xlabel('number of clusters')
        plt.show()

    def sift_descriptor_extractor(self, img_path):
        '''
        特征提取：提取数据集中每幅图像的特征点，然后提取特征描述符，形成特征数据(如：SIFT或者SURF方法)；
        '''
        im = cv2.imread(img_path, 0)
        return self.descriptor_extractor.compute(im, self.feature_detector.detect(im))[1]

    def bow_descriptor_extractor(self, img_path):
        '''
        提取图像的BOW特征描述(即利用视觉词袋量化图像特征)
        '''
        #print(img_path)
        im = cv2.imread(img_path, 0)
        return self.bow_img_descriptor_extractor.compute(im, self.feature_detector.detect(im))


if __name__ == '__main__':
    # 训练集图片路径  狗和猫两类  进行训练
    train_path = './images'
    bow = BOW()
    bow.Fit(train_path)


