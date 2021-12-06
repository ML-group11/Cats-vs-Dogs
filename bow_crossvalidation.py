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
from sklearn.metrics import accuracy_score


class BOW(object):

    def __init__(self, ):
        # 创建一个SIFT对象  用于关键点提取
        self.feature_detector = cv2.xfeatures2d.SIFT_create()
        # 创建一个SIFT对象  用于关键点描述符提取
        self.descriptor_extractor = cv2.xfeatures2d.SIFT_create()

    def path(self, cls, i):
        '''
        用于获取图片的全路径
        '''
        return '%s/%s/%d.jpg' % (self.train_path, cls, i)

    def Fit(self, train_path, k):
        '''
        开始训练

        args：
            train_path：训练集图片路径  我们使用的数据格式为 train_path/dogs/i.jpg  train_path/cats/i.jpg
            k：k-means参数k
        '''
        self.train_path = train_path

        # FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
        flann_params = dict(algorithm=1, tree=5)
        flann = cv2.FlannBasedMatcher(flann_params, {})

        pos = 'dogs'
        neg = 'cats'
        accuracy_svm = []
        accuracy_knn = []
        # 指定用于提取词汇字典的样本数
        length_range = [1,5,10,15,20,25,30]
        for length in length_range:
            # 用于提取词汇字典的样本
            self.voc_X = []
            # 合并特征数据  读取length张图片(length个狗,length个猫)，通过聚类创建视觉词汇
            #length: The number of samples used to build the vocabulary
            for i in range(length):
                self.voc_X.extend(self.sift_descriptor_extractor(self.path(pos, i)))
                self.voc_X.extend(self.sift_descriptor_extractor(self.path(neg, i)))
            self.voc_X = np.array(self.voc_X)

            # 进行k - means聚类
            gmm = KMeans(n_clusters=k).fit(self.voc_X)
            # 返回词汇字典 也就是聚类中心
            voc = gmm.cluster_centers_

            # 输出词汇字典  <class 'numpy.ndarray'> (40, 128)
            print('-------vocabulary, vocabulary.shape----------')
            print(voc, voc.shape)

            # 初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
            self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.descriptor_extractor, flann)
            self.bow_img_descriptor_extractor.setVocabulary(voc)


            # source images and lables
            self.X, self.y = [], []
            # 创建两个数组，分别对应训练数据和标签，并用BOWImgDescriptorExtractor产生的描述符填充
            # 按照下面的方法生成相应的正负样本图片的标签 1：正匹配(dog)  -1：负匹配(cat)
            for i in range(500):  # 狗和猫分别取...张图像
                self.X.extend(self.bow_descriptor_extractor(self.path(pos, i)))
                self.y.append(1)
                self.X.extend(self.bow_descriptor_extractor(self.path(neg, i)))
                self.y.append(-1)
            self.X = np.array(self.X)
            self.y = np.array(self.y)
            self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=0.2)

            accuracy_svm_temp = []
            accuracy_knn_temp = []
            kf = KFold(n_splits=5)
            for train, test in kf.split(self.Xtrain):
                self.linerSVC(train)
                self.KNN(train)
                self.Predict(test, accuracy_svm_temp, accuracy_knn_temp)
            accuracy_svm.append(np.array(accuracy_svm_temp).mean())
            accuracy_knn.append(np.array(accuracy_knn_temp).mean())
        plt.figure()
        plt.rc('font', size=22)
        plt.subplot(121)
        plt.title('linerSVC')
        plt.errorbar(length_range, accuracy_svm)
        plt.xlabel('The number of samples used to build the vocabulary')
        plt.ylabel('accuracy')

        plt.subplot(122)
        plt.title('KNN')
        plt.errorbar(length_range,accuracy_knn)
        plt.xlabel('The number of samples used to build the vocabulary')
        plt.ylabel('accuracy')

        plt.show()
    def linerSVC(self, train):
        # 创建一个SVM对象,C=5
        self.svm = LinearSVC(C=1)
        # 使用训练数据和标签进行训练
        self.svm.fit(np.array(self.X[train]), np.array(self.y[train]))

    def KNN(self, train):
        # 创建一个KNN对象,k=25
        self.knn = KNeighborsClassifier(n_neighbors=25, weights='uniform')
        # 使用训练数据和标签进行训练
        self.knn.fit(np.array(self.X[train]), np.array(self.y[train]))

    def Predict(self, test, accuracy_svm_temp, accuracy_knn_temp):
        self.ypred_svm = self.svm.predict(self.X[test])
        self.ypred_knn = self.knn.predict(self.X[test])

        print('=========== svm confusion_matrix===================')
        print(confusion_matrix(self.y[test], self.ypred_svm))
        print(classification_report(self.y[test], self.ypred_svm))
        accuracy_svm_temp.append(accuracy_score(self.y[test], self.ypred_svm))
        print(accuracy_score(self.y[test], self.ypred_svm))

        print('=========== knn confusion_matrix===================')
        print(confusion_matrix(self.y[test], self.ypred_knn))
        print(classification_report(self.y[test], self.ypred_knn))
        accuracy_knn_temp.append(accuracy_score(self.y[test], self.ypred_knn))
        print(accuracy_score(self.y[test], self.ypred_knn))
    def predict_one_svm(self, img_path):
        '''
        进行预测样本
        '''
        # 提取图片的BOW特征描述
        data = self.bow_descriptor_extractor(img_path)
        res = self.svm.predict(data)
        print(img_path, '\t', res[0])

        # 如果是狗 返回True
        if res[0] == 1.0:
            return True
        # 如果是猫，返回False
        else:
            return False

    def predict_one_knn(self, img_path):
        '''
        进行预测样本
        '''
        # 提取图片的BOW特征描述
        data = self.bow_descriptor_extractor(img_path)
        res = self.knn.predict(data)
        print(img_path, '\t', res[0])

        # 如果是狗 返回True
        if res[0] == 1.0:
            return True
        # 如果是猫，返回False
        else:
            return False

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
    bow.Fit(train_path, 15)
