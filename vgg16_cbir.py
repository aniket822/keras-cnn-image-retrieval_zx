# -*- coding:utf-8 -*-
"""
@author:HuangJie
@time:18-9-19 上午11:05

"""
import numpy as np
import cv2
import cPickle
import os
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.utils import conv_utils, np_utils, plot_model
from keras.optimizers import SGD
from keras.callbacks import TensorBoard


class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights=self.weight, input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling=self.pooling, include_top=False)

    def define_model(self):
        for layer in self.model.layers:
            layer.trainable = False
        x = Flatten(name='flatten')(self.model.output)
        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', name='fc7')(x)
        x = Dense(101, activation='softmax', name='fc8')(x)
        model = Model(inputs=self.model.input, outputs=x, name='vgg16')
        return model

    @staticmethod
    def get_name_list(filepath):
        pathdir = os.listdir(filepath)
        out = []
        for alldir in pathdir:
            if os.path.isdir(os.path.join(filepath, alldir)):
                child = alldir.decode('gbk')
                out.append(child)
        return out

    @staticmethod
    def eachfile(filepath):
        pathdir = os.listdir(filepath)
        out = []
        for alldir in pathdir:
            child = alldir.decode('gbk')
            out.append(child)
        return out

    def get_data(self, data_name, train_precentage, resize, data_format):
        file_name = os.path.join(pic_dir_out, data_name+str(Width)+'x'+str(Height)+'.pkl')
        if os.path.exists(file_name):
            (x_train, y_train), (x_test, y_test) = cPickle.load(open(file_name, 'rb'))
            return (x_train, y_train), (x_test, y_test)
        data_format = conv_utils.normalize_data_format(data_format)
        pic_dir_set = VGGNet.eachfile(pic_dir_data)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        label = 0
        for pic_dir in pic_dir_set:
            print pic_dir_data+pic_dir
            if not os.path.isdir(os.path.join(pic_dir_data, pic_dir)):
                continue
            pic_set = VGGNet.eachfile(os.path.join(pic_dir_data, pic_dir))
            pic_index = 0
            train_count = int(len(pic_set)*train_precentage)
            for pic_name in pic_set:
                if not os.path.isfile(os.path.join(pic_dir_data, pic_dir, pic_name)):
                    continue
                img = cv2.imread(os.path.join(pic_dir_data, pic_dir, pic_name))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if resize:
                    img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
                if data_format == 'channels_last':
                    img = img.reshape(-1, Width, Height, 3)
                elif data_format == 'channels_first':
                    img = img.reshape(-1, 3, Width, Height)
                if pic_index < train_count:
                    x_train.append(img)
                    y_train.append(label)
                else:
                    x_test.append(img)
                    y_test.append(label)
                pic_index += 1
            if len(pic_set) != 0:
                label += 1
            x_train = np.concatenate(x_train, axis=0)
            x_test = np.concatenate(x_test, axis=0)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            cPickle.dump([(x_train, y_train), (x_test, y_test)], open(file_name, 'wb'))
            return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def train_model():
        (x_train, y_train), (x_test, y_test) = \
            VGGNet().get_data(data_name='commodity101_data_', train_precentage=0.7, resize=True, data_format='channels_last')
        x_train = x_train/255.
        x_test = x_test/255.
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)
        model = VGGNet.define_model()
        sgd = SGD(lr=0.05, decay=1e-5, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print "\n-------------Training------------------"
        model.fit(x_train, y_train, epochs=10, batch_size=64)
        model.save_weights(os.path.join(pic_dir_out,'vgg16_cbir_model.h5'))
        print "\n-------------Testing-------------------"
        loss, accuracy = model.evaluate(x_test, y_test)
        print "\n"
        print "test loss:", loss
        print "test accuracy:", accuracy
        class_name_list = VGGNet.get_name_list(pic_dir_data)
        pred = model.predict(x_test, batch_size=64)
        pred_list = []
        for row in pred:
            pred_list.append(row.argsort()[-5:][::-1])
        pred_array = np.array(pred_list)
        test_arg = np.argmax(y_test, axis=1)
        class_count = [0 for _ in xrange(num_classes)]
        class_acc = [0 for _ in xrange(num_classes)]
        for i in xrange(len(test_arg)):
            class_count[test_arg[i]] += 1
            if test_arg[i] in pred_array[i]:
                class_acc[test_arg[i]] += 1
        print "top-" + str(5)+"all acc:", \
                              str(sum(class_acc))+"/"+str(len(test_arg)), \
                              sum(class_acc)/float(len(test_arg))
        for i in xrange(num_classes):
            print i, class_name_list[i], 'acc:'+str(class_acc[i])+'/'+str(class_count[i])
        model.summary()
        print(model.summary())
        plot_model(model, show_shapes=True, to_file=os.path.join(pic_dir_out, 'model.png'))

        return


if __name__ == '__main__':
    global Width, Height, pic_dir_out, pic_dir_data, num_classes
    Width = 224
    Height = 224
    num_classes = 101
    pic_dir_data = '/home/hj/PycharmProjects/keras-cnn-images-retrieval_zx/ClassesFiles/'
    pic_dir_out = '/home/hj/PycharmProjects/keras-cnn-images-retrieval_zx/result/'
    VGGNet().train_model()
