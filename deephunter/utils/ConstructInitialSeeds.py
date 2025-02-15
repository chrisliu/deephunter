#!/usr/bin/env python2.7
import argparse
import os
import sys
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import MobileNet, VGG19, ResNet50
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import Input
from PIL import Image

sys.path.append('../')
model_weight_path = {
    'vgg16': "./profile/cifar10/models/vgg16.h5",
    'resnet20': "/data/dnntest/zpengac/models/resnet/cifar10_resnet20v1_keras_py2.h5",
    'lenet1': "./profile/mnist/models/lenet1.h5",
    'lenet4': "./profile/mnist/models/lenet4.h5",
    'lenet5': "/data/dnntest/zpengac/models/lenet/mnist_lenet5_keras_32_py2.h5"
}
DATADIR = "/data/dnntest/zpengac/datasets/imagenet/"

def mnist_preprocessing(x):
    x = x.reshape(x_test.shape[0], 28, 28)
    new_x = []
    for img in x:
        img = Image.fromarray(img.astype('uint8'), 'L')
        img = img.resize(size=(32, 32))
        img = np.asarray(img).astype(np.float32) / 255.0
        new_x.append(img)
    new_x = np.stack(new_x)
    new_x = np.expand_dims(new_x, axis=-1)
    return new_x

def color_preprocessing(x_test):
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_test
def preprocessing_test_batch(x_test):
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test

def imgnt_preprocessing(x_test):
    x_test = x_test.astype('float32')
    mean = [0.47829157, 0.45374238, 0.403489]
    std = [0.27419248, 0.26739293, 0.2803236]
    for i in range(3):
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_test

def createBatch(x_batch, batch_size, output_path, prefix):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    batch_num = len(x_batch) / batch_size
    batches = np.split(x_batch, batch_num, axis=0)
    for i, batch in enumerate(batches):
        test = np.append(batch, batch, axis=0)
        saved_name = prefix + str(i) + '.npy'
        np.save(os.path.join(output_path, saved_name), test)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='control experiment')


    parser.add_argument('-model_type', help='Model type', choices=['lenet1','lenet4','lenet5','resnet20', 'resnet50', 'vgg16', 'vgg19'], default='lenet5')
    parser.add_argument('-output_path', help='Out path')
    parser.add_argument('-batch_size', type=int, help='Number of images in one batch', default=1)
    parser.add_argument('-batch_num', type=int, help='Number of batches', default=30)
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.model_type in ['lenet1','lenet4','lenet5']:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        batch = mnist_preprocessing(x_test)
        model = load_model(model_weight_path[args.model_type])
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    elif args.model_type in ['resnet20', 'vgg16']:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        batch = color_preprocessing(x_test)
        model = load_model(model_weight_path[args.model_type])
    else:
        x_test = np.load(DATADIR + 'x_test.npy')
        y_test = np.load(DATADIR + 'y_test.npy')
        batch = x_test
        input_tensor = Input(shape=(256, 256, 3))
        model = ResNet50(input_tensor=input_tensor)

    num_in_each_class = args.batch_num / 1000
    result = np.argmax(model.predict(batch),axis=1)
    new_label = np.reshape(y_test, result.shape)
    idx_good = np.where(new_label == result)[0]

    #for i in range(args.batch_num):
    #for cl in range(10):
    #cl_indexes  = [i for i in idx_good if new_label[i] == cl]
    #selected = random.sample(cl_indexes, num_in_each_class)
    #selected = random.sample(cl_indexes, num_in_each_class)
    #createBatch(x_test, args.batch_size, args.output_path, 'Imgnt_')
    with open('imagenet_indices.txt', 'w') as f:
        for cl in range(1000):
            cl_indexes  = [i for i in idx_good if new_label[i] == cl]
            selected = random.sample(cl_indexes, num_in_each_class)
            createBatch(x_test[selected], args.batch_size, args.output_path, str(cl)+'_')
            selected = [str(i)+'\n' for i in selected]
            f.writelines(selected)

print('finish')
