from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import argparse, pickle
import shutil

from tensorflow.keras.models import load_model
import tensorflow as tf
import os

sys.path.append('../')

import tensorflow.keras as keras
from tensorflow.keras import Input
from deephunter.coverage import Coverage

from tensorflow.keras.applications import MobileNet, VGG19, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input

import random
import time
import numpy as np
from PIL import Image
from deephunter.image_queue import ImageInputCorpus, TensorInputCorpus
from deephunter.fuzzone import build_fetch_function

from lib.queue import Seed
from lib.fuzzer import Fuzzer

from deephunter.mutators import Mutators
from tensorflow.keras.utils import CustomObjectScope

from keras_bert import get_custom_objects
import io
from text_models import (
    PretrainedList, download_imdb, 
    load_imdb, get_BERTClassifier, preprocess_imdb
)

#_, (x_test, y_test) = keras.datasets.cifar10.load_data()
#x_test=x_test/255.0
#x_test=x_test.reshape(10000,32,32,3)
#y_test=y_test.reshape(-1)

def dry_run_text(texts, labels, fetch_function, coverage_function, queue):
    for i, (text, label) in enumerate(zip(texts, labels)):
        tf.logging.info(f'Attempting dry run iteration {i}')
        coverage_batches, metadata_batches = fetch_function([None, [text], None, None, None])
        
        coverage_list = coverage_function(coverage_batches)
        metadata_list = metadata_function(metadata_batches)

        input = Seed(0, coverage_list[0], f'seed{i}', None, metadata_list[0], label)
        queue.save_if_interesting(input, [text], False, True, f'seed{i}')

def build_text_random_mutate_function(texts, batch_num):
    def text_random_mutate(seed):
        ref_text = texts[int(seed.root_seed.replace('seed', ''))]
        text = ref_text
        
        cl = seed.clss
        ref_batches = []
        batches = []
        cl_batches = []
        l0_ref_batches = []
        linf_ref_batches = []
        for i in range(batch_num):
            ref_out, text_out, cl_out, changed, l0_ref, linf_ref = Mutators.text_mutate_one(ref_text, text, cl, seed.l0_ref, seed.linf_ref)
            if changed:
                ref_batches.append(ref_out)
                batches.append(text_out)
                cl_batches.append(cl_out)
                l0_ref_batches.append(l0_ref)
                linf_ref_batches.append(linf_ref)

        return np.asarray(ref_batches), np.asarray(batches), cl_batches, l0_ref_batches, linf_ref_batches

    return text_random_mutate

def build_bert_fetch(handler, preprocess):
    def fetch_function(input_batches):
        _, text_batches, _, _, _  = input_batches
        preprocessed = preprocess(text_batches)
        layer_outputs = handler.predict(preprocessed)
        return layer_outputs, np.expand_dims(np.argmax(layer_outputs[-1], axis=1),axis=0)
    return fetch_function

def imagenet_preprocessing(input_img_data):
    #print("imgenet preprocessing")
    temp = np.copy(input_img_data)
    temp = np.float32(temp)
    qq = preprocess_input(temp)
    return qq

def imgnt_preprocessing(x_test):
    #print("imgnt preprocessing")
    return x_test

def mnist_preprocessing(x):
    #print("mnist preprocessing")
    x = x.reshape(x.shape[0], 28, 28)
    new_x = []
    for img in x:
        img = Image.fromarray(img.astype('uint8'), 'L')
        #img = img.resize(size=(32, 32))
        img = np.asarray(img).astype(np.float32) / 255.0 - 0.1306604762738431
        new_x.append(img)
    new_x = np.stack(new_x)
    new_x = np.expand_dims(new_x, axis=-1)
    return new_x


def cifar_preprocessing(x_test):
    #print("cifar preprocessing")
    x_test = x_test.reshape(x_test.shape[0], 28, 28)
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp


model_weight_path = {
    'vgg16': "./profile/cifar10/models/vgg16.h5",
    'resnet20': "./profile/cifar10/models/resnet.h5",
    'lenet1': "./profile/mnist/models/lenet1.h5",
    'lenet4': "./profile/mnist/models/lenet4.h5",
    'lenet5': "./profile/mnist/models/lenet5.h5",
    'bert' : '/home2/chrisliu/bert.h5'
}

model_profile_path = {
    'vgg16': "./profile/cifar10/profiling/vgg16/0_50000.pickle",
    'resnet20': "./profile/cifar10/profiling/resnet20/0_50000.pickle",
    'lenet1': "./profile/mnist/profiling/lenet1/0_60000.pickle",
    'lenet4': "./profile/mnist/profiling/lenet4/0_60000.pickle",
    'lenet5': "./profile/mnist/profiling/lenet5/0_60000.pickle",
    'mobilenet': "./profile/imagenet/profiling/mobilenet_merged.pickle",
    'vgg19': "./profile/imagenet/profiling/vgg19_merged.pickle",
    'resnet50': "./profile/imagenet/profiling/resnet50_merged.pickle",
    'bert': '/home2/chrisliu/bert_40.pickle'
}

preprocess_dic = {
    'vgg16': cifar_preprocessing,
    'resnet20': cifar_preprocessing,
    'lenet1': mnist_preprocessing,
    'lenet4': mnist_preprocessing,
    'lenet5': mnist_preprocessing,
    'mobilenet': imagenet_preprocessing,
    'vgg19': imagenet_preprocessing,
    'resnet50': imgnt_preprocessing
}

shape_dic = {
    'vgg16': (32, 32, 3),
    'resnet20': (32, 32, 3),
    'lenet1': (28, 28, 1),
    'lenet4': (28, 28, 1),
    'lenet5': (32, 32, 1),
    'mobilenet': (224, 224, 3),
    'vgg19': (224, 224, 3),
    'resnet50': (256, 256, 3)
}

metrics_para = {
    'kmnc': 1000,
    'bknc': 10,
    'tknc': 10,
    'nbc': 10,
    'newnc': 10,
    'nc': 0.75,
    'fann': 1.0,
    'snac': 10
}

execlude_layer_dic = {
    'vgg16': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'resnet20': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet1': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet4': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet5': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'mobilenet': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout',
                  'bn', 'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'vgg19': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
              'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'resnet50': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
                 'reshape', 'relu', 'pool', 'concat', 'add', 'res4', 'res5'],
    'bert' : ['Input-Token', 'Input-Segment', 'Embedding-Token']
}


def metadata_function(meta_batches):
    return meta_batches


def image_mutation_function(batch_num):
    # Given a seed, randomly generate a batch of mutants
    def func(seed):
        return Mutators.image_random_mutate(seed, batch_num)

    return func

def text_mutation_function(batch_num):
    # Given a seed, randomly generate a batch of mutants
    def func(seed):
        return Mutators.text_random_mutate(seed, batch_num)

    return func


def objective_function(seed, names):
    metadata = seed.metadata
    ground_truth = seed.ground_truth
    assert (names is not None)
    results = []
    if len(metadata) == 1:
        # To check whether it is an adversarial sample
        if metadata[0] != ground_truth:
            results.append('')
    else:
        # To check whether it has different results between original model and quantized model
        # metadata[0] is the result of original model while metadata[1:] is the results of other models.
        # We use the string adv to represent different types;
        # adv = '' means the seed is not an adversarial sample in original model but has a different result in the
        # quantized version.  adv = 'a' means the seed is adversarial sample and has a different results in quan models.
        if metadata[0] == ground_truth:
            adv = ''
        else:
            adv = 'a'
        count = 1
        while count < len(metadata):
            if metadata[count] != metadata[0]:
                results.append(names[count] + adv)
            count += 1

    # results records the suffix for the name of the failed tests
    return results


def iterate_function(names):
    def func(queue, root_seed, parent, mutated_coverage_list, mutated_data_batches, mutated_metadata_list,
             objective_function):

        ref_batches, batches, cl_batches, l0_batches, linf_batches = mutated_data_batches

        successed = False
        bug_found = False
        # For each mutant in the batch, we will check the coverage and whether it is a failed test
        for idx in range(len(mutated_coverage_list)):

            input = Seed(cl_batches[idx], mutated_coverage_list[idx], root_seed, parent, mutated_metadata_list[:, idx],
                         parent.ground_truth, l0_batches[idx], linf_batches[idx])

            # The implementation for the isFailedTest() in Algorithm 1 of the paper
            results = objective_function(input, names)

            if len(results) > 0:
                # We have find the failed test and save it in the crash dir.
                for i in results:
                    queue.save_if_interesting(input, batches[idx], True, suffix=i)
                bug_found = True
            else:

                new_img = np.append(ref_batches[idx:idx + 1], batches[idx:idx + 1], axis=0)
                # If it is not a failed test, we will check whether it has a coverage gain
                result = queue.save_if_interesting(input, new_img, False)
                successed = successed or result
        return bug_found, successed

    return func


def dry_run(indir, fetch_function, coverage_function, queue):
    seed_lis = os.listdir(indir)
    # Read each initial seed and analyze the coverage
    for seed_name in seed_lis:
        tf.logging.info("Attempting dry run with '%s'...", seed_name)
        path = os.path.join(indir, seed_name)
        img = np.load(path)
        # Each seed will contain two images, i.e., the reference image and mutant (see the paper)
        input_batches = img[1:2]
        # Predict the mutant and obtain the outputs
        # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
        coverage_batches, metadata_batches = fetch_function((0, input_batches, 0, 0, 0))
        # Based on the output, compute the coverage information
        coverage_list = coverage_function(coverage_batches)
        metadata_list = metadata_function(metadata_batches)
        # Create a new seed
        input = Seed(0, coverage_list[0], seed_name, None, metadata_list[0][0], metadata_list[0][0])
        new_img = np.append(input_batches, input_batches, axis=0)
        # Put the seed in the queue and save the npy file in the queue dir
        queue.save_if_interesting(input, new_img, False, True, seed_name)

import time
class Timer:
    def __init__(self):
        self.__start = None
        self.__end = None
        self.__name = ""

    def start(self, name=""):
        if name == "":
            print("Starting timer")
        else:
            print(name)

        self.__start = time.time()
        self.__name = name
        return self

    def end(self):
        self.__end = time.time()
        return self

    def elapsed(self):
        if self.__name == "":
            return "Took {:0.2f} seconds".format(self.__end - self.__start)
        else:
            return "{} took {:0.2f} seconds".format(self.__name,
                                                    self.__end - self.__start)
        return self

if __name__ == '__main__':

    start_time = time.time()

    tf.logging.set_verbosity(tf.logging.INFO)
    random.seed(time.time())

    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')

    parser.add_argument('-i', help='input seed directory')
    parser.add_argument('-o', help='output directory')

    # TODO: add text model here
    parser.add_argument('-model', help="target model to fuzz", choices=['vgg16', 'resnet20', 'mobilenet', 'vgg19',
                                                                        'resnet50', 'lenet1', 'lenet4', 'lenet5',
                                                                        'bert'], default='lenet5')
    parser.add_argument('-criteria', help="set the criteria to guide the fuzzing",
                        choices=['nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc', 'fann'], default='kmnc')
    parser.add_argument('-batch_num', help="the number of mutants generated for each seed", type=int, default=20)
    parser.add_argument('-max_iteration', help="maximum number of fuzz iterations", type=int, default=10000000)
    parser.add_argument('-metric_para', help="set the parameter for different metrics", type=float)
    parser.add_argument('-quantize_test', help="fuzzer for quantization", default=0, type=int)
    # parser.add_argument('-ann_threshold', help="Distance below which we consider something new coverage.", type=float,
    #                     default=1.0)
    parser.add_argument('-quan_model_dir', help="directory including the quantized models for testing")
    parser.add_argument('-random', help="whether to adopt random testing strategy", type=int, default=0)
    parser.add_argument('-select', help="test selection strategy",
                        choices=['uniform', 'tensorfuzz', 'deeptest', 'prob'], default='prob')

    args = parser.parse_args()

    timer = Timer()
    timer.start('Setup')

    # TODO: figure out text size
    img_rows, img_cols = 256, 256
    input_shape = (img_rows, img_cols, 3)
    input_tensor = Input(shape=input_shape)

    # Get the layers which will be excluded during the coverage computation
    exclude_layer_list = execlude_layer_dic[args.model]

    # Create the output directory including seed queue and crash dir, it is like AFL
    if os.path.exists(args.o):
        shutil.rmtree(args.o)
    os.makedirs(os.path.join(args.o, 'queue'))
    os.makedirs(os.path.join(args.o, 'crashes'))

    # Load model. For ImageNet, we use the default models from Keras framework.
    # For other models, we load the model from the h5 file.
    model = None
    if args.model == 'mobilenet':
        model = MobileNet(input_tensor=input_tensor)
    elif args.model == 'vgg19':
        model = VGG19(input_tensor=input_tensor, input_shape=input_shape)
    elif args.model == 'resnet50':
        model = ResNet50(input_tensor=input_tensor)
    elif args.model == 'bert':
        pass
        model = load_model(model_weight_path[args.model], custom_objects=get_custom_objects())
    else:
        model = load_model(model_weight_path[args.model])

    # Get the preprocess function based on different dataset
    if args.model == 'bert': # Dynamically instantiate for bert (need tokenizer)
        _, tokenizer = get_BERTClassifier(PretrainedList.base_cased)

        def build_bert_preprocessor(tokenizer):
            def bert_preprocessor(text_batch):
                SEQ_LEN = 512
                tokenized = [tokenizer.encode(t, max_len=SEQ_LEN)[0] for t in text_batch]
                tokenized = np.array(tokenized)
                return [tokenized, np.zeros_like(tokenized)]
            
            return bert_preprocessor
  
        preprocess_dic['bert'] = build_bert_preprocessor(tokenizer)


    preprocess = preprocess_dic[args.model]

    # Load the profiling information which is needed by the metrics in DeepGauge
    profile_dict = pickle.load(open(model_profile_path[args.model], 'rb'))

    # Load the configuration for the selected metrics.
    if args.metric_para is None:
        cri = metrics_para[args.criteria]
    elif args.criteria == 'nc':
        cri = args.metric_para
    else:
        cri = int(args.metric_para)

    # The coverage computer
    coverage_handler = Coverage(model=model, criteria=args.criteria, k=cri,
                                profiling_dict=profile_dict, exclude_layer=exclude_layer_list)

    # The log file which records the plot data after each iteration of the fuzzing
    plot_file = open(os.path.join(args.o, 'plot.log'), 'a+')

    # Load dataset
    if args.model == 'bert':
        dataset_dir = download_imdb()
        _, _, texts, labels = load_imdb(dataset_dir, only='test')
        p = np.random.permutation(len(texts))
        size = 20
        texts = [texts[i] for i in p][:size]
        labels = [labels[i] for i in p][:size]

    # If testing for quantization, we will load the quantized versions
    # fetch_function is to perform the prediction and obtain the outputs of each layers
    if args.quantize_test == 1:
        model_names = os.listdir(args.quan_model_dir)
        model_paths = [os.path.join(args.quan_model_dir, name) for name in model_names]
        if args.model == 'mobilenet':
            import keras

            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                                    'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                models = [load_model(m) for m in model_paths]
        else:
            models = [load_model(m) for m in model_paths]
        fetch_function = build_fetch_function(coverage_handler, preprocess, models)
        model_names.insert(0, args.model)
    else:
        if args.model == 'bert':
            fetch_function = build_bert_fetch(coverage_handler, preprocess)
        else:
            fetch_function = build_fetch_function(coverage_handler, preprocess)
        model_names = [args.model]

    # Like AFL, dry_run will run all initial seeds and keep all initial seeds in the seed queue

    dry_run_fetch = build_fetch_function(coverage_handler, preprocess)

    # The function to update coverage
    coverage_function = coverage_handler.update_coverage
    # The function to perform the mutation from one seed
    # TODO: replace with name of text model
    if args.model == 'bert':
        mutation_function = build_text_random_mutate_function(texts, args.batch_num)
    else:
        mutation_function = image_mutation_function(args.batch_num)

    # The seed queue
    if args.criteria == 'fann':
        queue = TensorInputCorpus(args.o, args.random, args.select, cri, "kdtree")
    else:
        queue = ImageInputCorpus(args.o, args.random, args.select, coverage_handler.total_size, args.criteria)

    print(timer.end().elapsed())

    # Perform the dry_run process from the initial seeds
    if args.model == 'bert':
        dry_run_text(texts, labels, fetch_function, coverage_function, queue)
    else:
        dry_run(args.i, dry_run_fetch, coverage_function, queue)

    # For each seed, compute the coverage and check whether it is a "bug", i.e., adversarial example
    image_iterate_function = iterate_function(model_names)

    # The main fuzzer class
    fuzzer = Fuzzer(queue, coverage_function, metadata_function, objective_function, mutation_function, fetch_function,
                    image_iterate_function, args.select)

    # The fuzzing process
    fuzzer.loop(args.max_iteration)
    
    #x_test = cifar_preprocessing(x_test)
    #model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    #print(model.metrics_names)
    #print(model.evaluate(x_test, yy, verbose=1))

    spent_time = time.time() - start_time
    print('finish',  spent_time)
    f = open('time.txt', 'a+')
    f.write(args.model + '\t' + args.criteria + '\t' + args.select + '\t' + str(spent_time) + '\n')
    f.close()


