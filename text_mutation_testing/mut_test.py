from __future__ import print_function
import sys
import cv2
import numpy as np
import random
import time
import copy
reload(sys)
sys.setdefaultencoding('utf8')
from nltk import download as nltk_download, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

from test_strings import *

def rearrange_sentences(text, params):
    pst = PunktSentenceTokenizer()
    sentences = pst.tokenize(text)
    if (len(sentences) <= 1):
        return text
    ind1 = random.randrange(len(sentences))
    ind2 = random.randrange(len(sentences))
    while (ind1 == ind2):
        ind2 = random.randrange(len(sentences))
    tmp = sentences[ind1]
    sentences[ind1] = sentences[ind2]
    sentences[ind2] = tmp
    return " ".join(sentences)

# Generate a random synonym for a given word, if nltk has one.
# If not, return the word.

def get_rand_synonym(word):
    try:
        synsets = wordnet.synsets(word)
    except LookupError:
        # Need to download wordnet once.
        nltk_download('wordnet')
        synsets = wordnet.synsets(word)
    synonym_list = map(lambda x: str(x.lemmas()[0].name()), synsets)
    synonym_list = [w for w in synonym_list if w != word]
    if (len(synonym_list) == 0):
        # There are no synonyms, so just return the original word.
        return word
    return random.sample(synonym_list, 1)[0]

# Replace one word with a synonym provided by nltk.
def sub_synonym(text, params):
    # break up text into words
    try:
        words = word_tokenize(text)
    except LookupError:
        # Need to download nltk punkt once.
        nltk_download('punkt')
        words = word_tokenize(text)
    try:
        stoplist = set(stopwords.words('english'))
    except LookupError:
        # Need to download nltk stoplist once.
        nltk_download('stopwords')
        stoplist = set(stopwords.words('english'))

    # This approach is guaranteed to replace a word if one can be
    # replaced, but it takes longer because we have to check if
    # each word is replaceable.
    def word_at_index_replaceable(index):
        # A word is "replaceable" if it's not in the stoplist.
        # This also filters out punctuation.
        return (words[index] not in stoplist
                and (len(words[index]) > 1 or words[index].isalpha()))
    replaceable_indices = [i for i in range(len(words)) if word_at_index_replaceable(i)]
    if (len(replaceable_indices) == 0):
        return text
    replace_word_index = random.sample(replaceable_indices, 1)[0]

    # This approach is faster but is not guaranteed to replace a word.
    # replace_word_index = random.randrange(0, len(words))
    # i = 0
    # while (words[replace_word_index] in stoplist):
    #     if (i == 10):
    #         # Give up guessing indices
    #         return text
    #     replace_word_index = random.randrange(0, len(words))
    #     i += 1

    # Swap word with synonym.
    words[replace_word_index] = get_rand_synonym(words[replace_word_index]).replace('_', ' ')
    detokenizer = TreebankWordDetokenizer()
    return detokenizer.detokenize(words)

# Character-level mutations: add, swap, delete.

def get_rand_char():
    return chr(97 + np.random.randint(0, 26))


def add_rand_char(text):
    rand_char = get_rand_char()
    # Include index past last char so we can add new char at end.
    rand_loc = np.random.randint(0, len(text) + 1)
    return text[:rand_loc] + rand_char + text[rand_loc:]


def del_rand_char(text):
    rand_loc = np.random.randint(0, len(text))
    return text[:rand_loc] + text[rand_loc+1:]


def sub_rand_char(text):
    rand_char = get_rand_char()
    rand_loc = np.random.randint(0, len(text))
    return text[:rand_loc] + rand_char + text[rand_loc+1:]

# Either add, delete, or substitute a random character.
def mutate_char(text, params):
    mutation_funcs = [
        add_rand_char,
        del_rand_char,
        sub_rand_char]
    return mutation_funcs[params](text)


text_mutation_ids = [0, 1]
text_char_mut_ids = [2]
# Fll in with method names of text transformations
text_transformations = [rearrange_sentences, sub_synonym, mutate_char]

text_params = []
text_params.append([None])  # rearrange_sentences
text_params.append([None])  # sub_synonym
text_params.append(list(xrange(0, 3)))  # mutate_char


def text_mutate_one(text, try_num=50):

    for ii in range(try_num):
        random.seed(time.time())
        tid = random.sample(
            text_mutation_ids + text_char_mut_ids, 1)[0]
        transformation = text_transformations[tid]
        params = text_params[tid]
        param = random.sample(params, 1)[0]
        text_new = transformation(text, param)

        if (text_new != text):
            return text_new

    # Otherwise the mutation is failed. Line 20 in Algo 2
    return text

def run_test_single_mutation():
    test_out_file = open("mutation_test_single_mutation.out", "w")
    tests = [
        test1, test2, test3, test4, test5, test6, test7, test8,
        test9, test10, test11, test12, test13, test14, test15, test16
    ]
    ground_truth_sentiments = [
        sent1, sent2, sent3, sent4, sent5, sent6, sent7, sent8,
        sent9, sent10, sent11, sent12, sent13, sent14, sent15, sent16
    ]
    for i in range(0, 16):
        test_out_file.write("Test " + str(i + 1) + ":\n")
        test_out_file.write(tests[i] + "\n")
        test_out_file.write("Original sentiment: " + str(ground_truth_sentiments[i]) + "\n")
        for trial in range(10):
            test_out_file.write("Mutation " + str(trial + 1) + ":\n")
            test_out_file.write(text_mutate_one(tests[i]) + "\n")
        test_out_file.write("\n")
    test_out_file.close()

def run_test_repeated_mutation():
    test_out_file = open("mutation_test_10x_mutation.out", "w")
    tests = [
        test1, test2, test3, test4, test5, test6, test7, test8,
        test9, test10, test11, test12, test13, test14, test15, test16
    ]
    ground_truth_sentiments = [
        sent1, sent2, sent3, sent4, sent5, sent6, sent7, sent8,
        sent9, sent10, sent11, sent12, sent13, sent14, sent15, sent16
    ]
    for i in range(0, 16):
        test_out_file.write("Test " + str(i + 1) + ":\n")
        test_out_file.write(tests[i] + "\n")
        test_out_file.write("Original sentiment: " + str(ground_truth_sentiments[i]) + "\n")
        for trial in range(10):
            test_out_file.write("Mutation " + str(trial + 1) + ":\n")
            text = tests[i]
            for mut in range(10):
                text = text_mutate_one(text)
            test_out_file.write(text + "\n")
        test_out_file.write("\n")
    test_out_file.close()

if __name__ == '__main__':
    run_test_single_mutation()
    run_test_repeated_mutation()
