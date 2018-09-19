# -*- coding:utf-8 -*-
"""
@author:HuangJie
@time:18-9-19 上午11:05

"""
import numpy as np
from numpy import linalg as LA
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Flatten,Dense,Dropout
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import SGD


class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights=self.weight, input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling=self.pooling, include_top=False)
        self.model.predict(np.zeros((1, 224, 224, 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        print(self.model.summary())
        plot_model(self.model, to_file='a simple convnet.png')
        for layer in self.model.layers:
            layer.trainable = False
        x = Flatten(name='flatten')(self.model.output)
        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', name='fc7')(x)
        x = Dense(101, activation='softmax', name='fc8')(x)
        model_vgg16_cbir_pretrain = Model(inputs=self.model.input, outputs=x, name='vgg16')
        model_vgg16_cbir_pretrain.summary()
        sgd = SGD(lr=0.05, decay=1e-5)
        model_vgg16_cbir_pretrain.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        feat = self.model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat

if __name__ == '__main__':
    VGGNet.extract_feat(image_path='/home/hj/PycharmProjects/keras-cnn-image-retrieval_zx/database/001_accordion_image_0001.jpg')
