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


class BOW(object):

    def __init__(self, ):
        # 创建一个SIFT对象  用于关键点提取
        self.feature_detector = cv2.xfeatures2d.SIFT_create()
        # 创建一个SIFT对象  用于关键点描述符提取
        self.descriptor_extractor = cv2.xfeatures2d.SIFT_create()
        # source images and lables
        self.X, self.y = [], []

    def path(self, cls, i):
        '''
        用于获取图片的全路径
        '''
        return '%s/%s/%d.jpg' % (self.train_path, cls, i)

    def Fit(self, train_path, k):
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

        # 创建BOW训练器，指定k-means参数k   把处理好的特征数据全部合并，利用聚类把特征词分为若干类，此若干类的数目由自己设定，每一类相当于一个视觉词汇
        bow_kmeans_trainer = cv2.BOWKMeansTrainer(k)

        pos = 'dogs'
        neg = 'cats'

        # 指定用于提取词汇字典的样本数
        length = 10
        # 合并特征数据  每个类从数据集中读取length张图片(length个狗,length个猫)，通过聚类创建视觉词汇
        for i in range(length):
            bow_kmeans_trainer.add(self.sift_descriptor_extractor(self.path(pos, i)))
            bow_kmeans_trainer.add(self.sift_descriptor_extractor(self.path(neg, i)))

        # 进行k-means聚类，返回词汇字典 也就是聚类中心
        voc = bow_kmeans_trainer.cluster()

        # 输出词汇字典  <class 'numpy.ndarray'> (40, 128)
        print('-------vocabulary, vocabulary.shape----------')
        print(voc, voc.shape)

        # 初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
        self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.descriptor_extractor, flann)
        self.bow_img_descriptor_extractor.setVocabulary(voc)

        # 创建两个数组，分别对应训练数据和标签，并用BOWImgDescriptorExtractor产生的描述符填充
        # 按照下面的方法生成相应的正负样本图片的标签 1：正匹配  -1：负匹配
        for i in range(10000):  # 狗和猫分别取1000张图像
            self.X.extend(self.bow_descriptor_extractor(self.path(pos, i)))
            self.y.append(1)
            self.X.extend(self.bow_descriptor_extractor(self.path(neg, i)))
            self.y.append(-1)
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=0.2)
        print('==========train data============')
        print(np.array(self.Xtrain).shape)
        print(np.array(self.ytrain).shape)
        print(np.array(self.Xtest).shape)
        print(np.array(self.ytest).shape)

        self.linerSVC()
        self.KNN()

    def linerSVC(self):

        # select  hyperparameter C use cross-validation (evaluate by F1 score)
        mean_error = []
        std_error = []
        Ci_range = [0.01, 0.1, 1, 5, 10, 25]
        for Ci in Ci_range:
            from sklearn.svm import LinearSVC
            model = LinearSVC(C=Ci)
            scores = cross_val_score(model, self.Xtrain, self.ytrain, cv=5, scoring='f1')
            mean_error.append(np.array(scores).mean())
            std_error.append(np.array(scores).std())
        plt.rc('font', size=18)
        plt.rc('font', size = 18); plt.rcParams['figure.constrained_layout.use'] = True
        plt.errorbar(Ci_range, mean_error, yerr=std_error, linewidth=3)
        plt.xlabel('Ci'); plt.ylabel('F1Score')
        plt.show()

    def KNN(self):

        # select  hyperparameter k use cross-validation (evaluate by F1 score)
        mean_error = []
        std_error = []
        K_range = [1, 5, 10, 25, 50]
        for k in K_range:
            model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
            scores = cross_val_score(model, self.Xtrain, self.ytrain, cv=5, scoring='f1')
            mean_error.append(np.array(scores).mean())
            std_error.append(np.array(scores).std())
        plt.rc('font', size=18)
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.errorbar(K_range, mean_error, yerr=std_error, linewidth=3)
        plt.xlabel('K');
        plt.ylabel('F1Score')
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
        print(img_path)
        im = cv2.imread(img_path, 0)
        return self.bow_img_descriptor_extractor.compute(im, self.feature_detector.detect(im))


if __name__ == '__main__':

    train_path = './images'
    bow = BOW()
    bow.Fit(train_path, 40)
