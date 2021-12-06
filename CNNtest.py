# -*- coding: utf-8 -*- 
# @Time : 2021/12/4 23:14 
# @Author : Tianyi  
# @File : CNNtest.py
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras import layers
from keras import models
from keras.layers import Dropout
from keras import optimizers
from keras.models import load_model
from sklearn.metrics import roc_curve

# 增加一个CNN的data文件夹，分类写
train_dir = './data/train/'
validation_dir = './data/validation'
test_dir = './data/test'
model_file_name = 'cat_dog_model.h5'

X_test = []
y_test = []

def read_img(cls):
    src_path = './data/test/{0}'.format(cls)
    fileList = os.listdir(src_path)

    for file in fileList:
        img_path = os.path.join(src_path, file)
        img = image.load_img(img_path, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor = img_tensor / 255

        X_test.append(img_tensor)
        if cls == 'dogs':
            y_test.append(1)
        else:
            y_test.append(-1)
        #print(img_path)

def init_model():
    model = models.Sequential()

    KERNEL_SIZE = (3, 3)

    model.add(layers.Conv2D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  # optimizer=optimizers.RMSprop(lr=0.0003),
                  metrics=['accuracy'])
    model.summary()

    return model


def fig_loss(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


def fig_acc(history):
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()


def fit(model):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=256,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=256,
        class_mode='binary')


    history = model.fit_generator(
        train_generator,
        # steps_per_epoch=,
        epochs=10,
        validation_data=validation_generator,
        # validation_steps=,
    )

    model.save(model_file_name)

    fig_loss(history)
    fig_acc(history)


def predictCNN(model):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=256,
        class_mode='binary')

    read_img('dogs')
    read_img('cats')

    global X_test
    global y_test
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred)

    # ROC
    plt.rc('font', size=18)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, color='black', label='CNN', linewidth=5)
    plt.legend(loc='lower right')

    for i in range(y_pred.shape[0]):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = -1

    y_pred = np.squeeze(y_pred)

    print(y_test.shape)
    print(y_pred.shape)
    print('=========== cnn confusion_matrix===================')
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print('======== cnn mean accuracy=======')
    print(accuracy_score(y_test,y_pred))

    #plt.show()


# 画出count个预测结果和图像
def fig_predict_result(model, count):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        './data/test/',
        target_size=(150, 150),
        batch_size=256,
        class_mode='binary')

    text_labels = []
    plt.figure(figsize=(30, 20))
    # 迭代器可以迭代很多条数据，但这里只取第一个结果看看
    for batch, label in test_generator:
        print(type(batch))
        print(batch.shape)
        pred = model.predict(batch)
        for i in range(count):
            true_reuslt = label[i]
            print(true_reuslt)
            if pred[i] > 0.5:
                text_labels.append('dog')
            else:
                text_labels.append('cat')

            # 4列，若干行的图
            plt.subplot(count / 4 + 1, 4, i + 1)
            plt.title('This is a ' + text_labels[i])
            plt.axis('off') # 关闭坐标轴
            imgplot = plt.imshow(batch[i])

        plt.show()

        # 可以接着画很多，但是只是随机看看几条结果。所以这里停下来。
        break


if __name__ == '__main__':
    model = init_model()
    fit(model)

    model = load_model(model_file_name)
    predictCNN(model)

    # 随机查看10个预测结果并画出它们
    #fig_predict_result(model, 10)
