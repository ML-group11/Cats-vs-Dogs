# -*- coding: utf-8 -*-

import numpy as np
import cv2
import CNNtest

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
import joblib
from sklearn.dummy import DummyClassifier

class BOW(object):

    def __init__(self, ):
        # 创建一个SIFT对象  用于关键点提取
        self.feature_detector = cv2.xfeatures2d.SIFT_create()
        # 创建一个SIFT对象  用于关键点描述符提取
        self.descriptor_extractor = cv2.xfeatures2d.SIFT_create()
        #用于提取词汇字典的样本
        self.voc_X = []
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
            train_path：训练集图片路径  我们使用的数据格式为 train_path/dogs/i.jpg  train_path/cats/i.jpg
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
        length = 15
        # 合并特征数据  读取length张图片(length个狗,length个猫)，通过聚类创建视觉词汇
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


        # 创建两个数组，分别对应训练数据和标签，并用BOWImgDescriptorExtractor产生的描述符填充
        # 按照下面的方法生成相应的正负样本图片的标签 1：正匹配(dog)  -1：负匹配(cat)
        for i in range(10000):  # 狗和猫分别取...张图像
            self.X.extend(self.bow_descriptor_extractor(self.path(pos, i)))
            self.y.append(1)
            self.X.extend(self.bow_descriptor_extractor(self.path(neg, i)))
            self.y.append(-1)
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=0.2)


        self.linerSVC(self.Xtrain)
        self.KNN(self.Xtrain)

        self.cnn = CNNtest.load_model(CNNtest.model_file_name)
        '''
        self.svm = joblib.load('linersvc.model')
        self.knn = joblib.load('knn.model')
        '''
        self.Dummy(self.Xtrain)

        self.Predict(self.Xtest)
    def linerSVC(self,train):
        # 创建一个SVM对象,C=5
        self.svm = LinearSVC(C=1)

        # 使用训练数据和标签进行训练
        self.svm.fit(np.array(self.Xtrain), np.array(self.ytrain))

        joblib.dump(self.svm, 'linersvc.model')
        
    def KNN(self,train):
        # 创建一个KNN对象,k=25
        self.knn = KNeighborsClassifier(n_neighbors=25, weights='uniform')

        # 使用训练数据和标签进行训练
        self.knn.fit(np.array(self.Xtrain), np.array(self.ytrain))

        joblib.dump(self.knn, 'knn.model')
    def Dummy(self,train):

        self.dummy = DummyClassifier(strategy="most_frequent")
        self.dummy.fit(np.array(self.Xtrain), np.array(self.ytrain))

    def Predict(self,test):
        self.ypred_svm = self.svm.predict(self.Xtest)
        self.ypred_knn = self.knn.predict(self.Xtest)
        self.ypred_dummy = self.dummy.predict(self.Xtest)

        print('=========== svm confusion_matrix===================')
        print(confusion_matrix(self.ytest, self.ypred_svm))
        print(classification_report(self.ytest, self.ypred_svm))
        print('======== svm mean accuracy=======')
        print(accuracy_score(self.ytest, self.ypred_svm))


        print('=========== knn confusion_matrix===================')
        print(confusion_matrix(self.ytest, self.ypred_knn))
        print(classification_report(self.ytest, self.ypred_knn))
        print('======== knn mean accuracy=======')
        print(accuracy_score(self.ytest, self.ypred_knn))


        # ROC
        plt.rc('font', size=18)
        fpr, tpr, _ = roc_curve(self.ytest, self.svm.decision_function(self.Xtest))
        plt.plot(fpr, tpr, label='LinearSVC', linewidth=5)

        fpr, tpr, _ = roc_curve(self.ytest, self.ypred_knn)
        plt.plot(fpr, tpr, label='KNN', linewidth=5)

        fpr, tpr, _ = roc_curve(self.ytest, self.ypred_dummy)
        plt.plot(fpr, tpr, label='Baseline', linestyle='--', linewidth=5)

        for i in range(self.ypred_dummy.shape[0]):
            if self.ypred_dummy[i] > 0.5:
                self.ypred_dummy[i] = 1
            else:
                self.ypred_dummy[i] = -1

        self.ypred_dummy = np.squeeze(self.ypred_dummy)

        print('=========== dummy confusion_matrix===================')
        print(confusion_matrix(self.ytest, self.ypred_dummy))
        print(classification_report(self.ytest, self.ypred_dummy))
        print('======== dummy mean accuracy=======')
        print(accuracy_score(self.ytest, self.ypred_dummy))

        CNNtest.predictCNN(self.cnn)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')

        plt.legend(loc='lower right')
        plt.show()

    def predict_one_svm(self, img_path):
        '''
        进行预测样本
        '''
        # 提取图片的BOW特征描述
        data = self.bow_descriptor_extractor(img_path)
        res = self.svm.predict(data)

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

        # 如果是狗 返回True
        if res[0] == 1.0:
            return True
        # 如果是猫，返回False
        else:
            return False
    def predict_one_cnn(self, img_path):
        '''
        进行预测样本
        '''
        img = CNNtest.image.load_img(img_path, target_size=(150, 150))
        img_tensor = CNNtest.image.img_to_array(img)
        img_tensor = img_tensor / 255
        img_tensor = np.expand_dims(img_tensor, axis=0)

        result = self.cnn.predict(img_tensor)
        if result> 0.5:
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
        print(img_path)
        im = cv2.imread(img_path, 0)
        return self.bow_img_descriptor_extractor.compute(im, self.feature_detector.detect(im))


if __name__ == '__main__':
    # 训练集图片路径  狗和猫两类  进行训练
    train_path = './images'
    bow = BOW()
    bow.Fit(train_path, 15)


    # select one image and show the result
    root = tk.Tk()
    root.withdraw()
    file_path1 = filedialog.askopenfilename()  # 选择图片
    dog_img_svm = cv2.imread(file_path1)
    dog_img_knn = cv2.imread(file_path1)
    dog_img_cnn = cv2.imread(file_path1)
    result_svm = bow.predict_one_svm(file_path1)
    result_knn = bow.predict_one_knn(file_path1)
    result_cnn = bow.predict_one_cnn(file_path1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.namedWindow("linerSVC", cv2.WINDOW_NORMAL)
    if result_svm:
        cv2.putText(dog_img_svm, 'is a dog --linerSVC', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(dog_img_svm, 'is not a dog --linerSVC', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('linerSVC', dog_img_svm)

    cv2.namedWindow("KNN", cv2.WINDOW_NORMAL)
    if result_knn:
        cv2.putText(dog_img_knn, 'is a dog --KNN', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(dog_img_knn, 'is not a dog --KNN', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('KNN', dog_img_knn)

    cv2.namedWindow("KNN", cv2.WINDOW_NORMAL)
    if result_knn:
        cv2.putText(dog_img_cnn, 'is a dog --CNN', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(dog_img_cnn, 'is not a dog --CNN', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('CNN', dog_img_cnn)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
