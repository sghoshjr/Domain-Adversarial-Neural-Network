"""
MNIST - MNIST-M Domain Adaptation
"""

import tensorflow as tf

print(tf.__version__)

import numpy as np

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, Dropout

import os
import shutil
import sys
import matplotlib.pyplot as plt
import h5py


#CONSTANTS
MNIST_M_PATH = './Datasets/MNIST_M/mnistm.h5'

BATCH_SIZE = 32
CHANNELS = 3
EPOCH = 100


#Load MNIST Data (Source)
(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = tf.keras.datasets.mnist.load_data()

#Convert to 3 Channel and One_hot labels
mnist_train_x, mnist_test_x = mnist_train_x.reshape((60000, 28, 28, 1)), mnist_test_x.reshape((10000, 28, 28, 1))

mnist_train_x, mnist_test_x = mnist_train_x / 255.0, mnist_test_x / 255.0
mnist_train_x, mnist_test_x = mnist_train_x.astype('float32'), mnist_test_x.astype('float32')

mnist_train_x = np.repeat(mnist_train_x, CHANNELS, axis=3)
mnist_test_x = np.repeat(mnist_test_x, CHANNELS, axis=3)

mnist_train_y = tf.one_hot(mnist_train_y, depth=10)
mnist_test_y = tf.one_hot(mnist_test_y, depth=10)

assert(mnist_train_x.shape == (60000,28,28,3))
assert(mnist_test_x.shape == (10000,28,28,3))
assert(mnist_train_y.shape == (60000,10))
assert(mnist_test_y.shape == (10000,10))





#Load MNIST-M [Target]

with h5py.File(MNIST_M_PATH, 'r') as mnist_m:
    mnist_m_train_x, mnist_m_test_x = mnist_m['train']['X'][()], mnist_m['test']['X'][()]


mnist_m_train_x, mnist_m_test_x = mnist_m_train_x / 255.0, mnist_m_test_x / 255.0
mnist_m_train_x, mnist_m_test_x = mnist_m_train_x.astype('float32'), mnist_m_test_x.astype('float32')

mnist_m_train_y, mnist_m_test_y = mnist_train_y, mnist_test_y

assert(mnist_m_train_x.shape == (60000,28,28,3))
assert(mnist_m_test_x.shape == (10000,28,28,3))
assert(mnist_m_train_y.shape == (60000,10))
assert(mnist_m_test_y.shape == (10000,10))




#Prepare Datasets

source_dataset = tf.data.Dataset.from_tensor_slices((mnist_train_x, mnist_train_y)).shuffle(1000).batch(BATCH_SIZE*2)
da_dataset = tf.data.Dataset.from_tensor_slices((mnist_train_x, mnist_train_y, mnist_m_train_x, mnist_m_train_y)).shuffle(1000).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((mnist_m_test_x, mnist_m_test_y)).shuffle(1000).batch(BATCH_SIZE*2) #Test Dataset over Target Domain
test_dataset2 = tf.data.Dataset.from_tensor_slices((mnist_m_train_x, mnist_m_train_y)).shuffle(1000).batch(BATCH_SIZE*2) #Test Dataset over Target (used for training)


#Gradient Reversal Layer
@tf.custom_gradient
def gradient_reverse(x, lamda=1.0):
    y = tf.identity(x)
    
    def grad(dy):
        return lamda * -dy, None
    
    return y, grad


class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x, lamda=1.0):
        return gradient_reverse(x, lamda)


class DANN(Model):
    def __init__(self):
        super().__init__()
        
        #Feature Extractor
        self.feature_extractor_layer0 = Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.feature_extractor_layer1 = BatchNormalization()
        self.feature_extractor_layer2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        
        self.feature_extractor_layer3 = Conv2D(64, kernel_size=(5, 5), activation='relu')
        self.feature_extractor_layer4 = Dropout(0.5)
        self.feature_extractor_layer5 = BatchNormalization()
        self.feature_extractor_layer6 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        
        #Label Predictor
        self.label_predictor_layer0 = Dense(100, activation='relu')
        self.label_predictor_layer1 = Dense(100, activation='relu')
        self.label_predictor_layer2 = Dense(10, activation=None)
        
        #Domain Predictor
        self.domain_predictor_layer0 = GradientReversalLayer()
        self.domain_predictor_layer1 = Dense(100, activation='relu')
        self.domain_predictor_layer2 = Dense(2, activation=None)
        
    def call(self, x, train=False, source_train=True, lamda=1.0):
        #Feature Extractor
        x = self.feature_extractor_layer0(x)
        x = self.feature_extractor_layer1(x, training=train)
        x = self.feature_extractor_layer2(x)
        
        x = self.feature_extractor_layer3(x)
        x = self.feature_extractor_layer4(x, training=train)
        x = self.feature_extractor_layer5(x, training=train)
        x = self.feature_extractor_layer6(x)
        
        feature = tf.reshape(x, [-1, 4 * 4 * 64])
        
        #Label Predictor
        if source_train is True:
            feature_slice = feature
        else:
            feature_slice = tf.slice(feature, [0, 0], [feature.shape[0] // 2, -1])
        
        lp_x = self.label_predictor_layer0(feature_slice)
        lp_x = self.label_predictor_layer1(lp_x)
        l_logits = self.label_predictor_layer2(lp_x)
        
        #Domain Predictor
        if source_train is True:
            return l_logits
        else:
            dp_x = self.domain_predictor_layer0(feature, lamda)    #GradientReversalLayer
            dp_x = self.domain_predictor_layer1(dp_x)
            d_logits = self.domain_predictor_layer2(dp_x)
            
            return l_logits, d_logits


model = DANN()



def loss_func(input_logits, target_labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=input_logits, labels=target_labels))


def get_loss(l_logits, labels, d_logits=None, domain=None):
    if d_logits is None:
        return loss_func(l_logits, labels)
    else:
        return loss_func(l_logits, labels) + loss_func(d_logits, domain)


model_optimizer = tf.optimizers.SGD()



domain_labels = np.vstack([np.tile([1., 0.], [BATCH_SIZE, 1]),
                           np.tile([0., 1.], [BATCH_SIZE, 1])])
domain_labels = domain_labels.astype('float32')


epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
source_acc = []  # Source Domain Accuracy while Source-only Training
da_acc = []      # Source Domain Accuracy while DA-training
test_acc = []    # Testing Dataset (Target Domain) Accuracy 
test2_acc = []   # Target Domain (used for Training) Accuracy


@tf.function
def train_step_source(s_images, s_labels, lamda=1.0):
    images = s_images
    labels = s_labels
    
    with tf.GradientTape() as tape:
        output = model(images, train=True, source_train=True, lamda=lamda)
        
        model_loss = get_loss(output, labels)
        epoch_accuracy(output, labels)
        
    gradients_mdan = tape.gradient(model_loss, model.trainable_variables)
    model_optimizer.apply_gradients(zip(gradients_mdan, model.trainable_variables))


@tf.function
def train_step_da(s_images, s_labels, t_images=None, t_labels=None, lamda=1.0):
    images = tf.concat([s_images, t_images], 0)
    labels = s_labels
    
    with tf.GradientTape() as tape:
        output = model(images, train=True, source_train=False, lamda=lamda)
        
        model_loss = get_loss(output[0], labels, output[1], domain_labels)
        epoch_accuracy(output[0], labels)
        
    gradients_mdan = tape.gradient(model_loss, model.trainable_variables)
    model_optimizer.apply_gradients(zip(gradients_mdan, model.trainable_variables))


@tf.function
def test_step(t_images, t_labels):
    images = t_images
    labels = t_labels
    
    output = model(images, train=False, source_train=True)
    epoch_accuracy(output, labels)


def train(train_mode, epochs=EPOCH):
    
    if train_mode == 'source':
        dataset = source_dataset
        train_func = train_step_source
        acc_list = source_acc
    elif train_mode == 'domain-adaptation':
        dataset = da_dataset
        train_func = train_step_da
        acc_list = da_acc
    else:
        raise ValueError("Unknown training Mode")
    
    for epoch in range(epochs):
        p = float(epoch) / epochs
        lamda = 2 / (1 + np.exp(-100 * p, dtype=np.float32)) - 1
        lamda = lamda.astype('float32')

        for batch in dataset:
            train_func(*batch, lamda=lamda)
        
        print("Training: Epoch {} :\t Source Accuracy : {:.3%}".format(epoch, epoch_accuracy.result()), end='  |  ')
        acc_list.append(epoch_accuracy.result())
        test()
        epoch_accuracy.reset_states()


def test():
    epoch_accuracy.reset_states()
    
    #Testing Dataset (Target Domain)
    for batch in test_dataset:
        test_step(*batch)
        
    print("Testing Accuracy : {:.3%}".format(epoch_accuracy.result()), end='  |  ')
    test_acc.append(epoch_accuracy.result())
    epoch_accuracy.reset_states()
    
    #Target Domain (used for Training)
    for batch in test_dataset2:
        test_step(*batch)
    
    print("Target Domain Accuracy : {:.3%}".format(epoch_accuracy.result()))
    test2_acc.append(epoch_accuracy.result())
    epoch_accuracy.reset_states()


#Training
#train('source', 5)

train('domain-adaptation', EPOCH)


#Plot Results
x_axis = [i for i in range(0, EPOCH)]

plt.plot(x_axis, da_acc, label="source accuracy")
plt.plot(x_axis, test_acc, label="testing accuracy")
plt.plot(x_axis, test2_acc, label="target accuracy")
plt.legend()