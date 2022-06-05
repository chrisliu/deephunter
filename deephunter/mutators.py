from __future__ import print_function
import sys
import cv2
import numpy as np
import random
import time
import copy
from nltk import download as nltk_download, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer


# keras 1.2.2 tf:1.2.0
class Mutators():
    def image_translation(img, params):

        rows, cols, ch = img.shape
        # rows, cols = img.shape

        # M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
        M = np.float32([[1, 0, params], [0, 1, params]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def image_scale(img, params):

        # res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
        rows, cols, ch = img.shape
        res = cv2.resize(img, None, fx=params, fy=params, interpolation=cv2.INTER_CUBIC)
        res = res.reshape((res.shape[0],res.shape[1],ch))
        y, x, z = res.shape
        if params > 1:  # need to crop
            startx = x // 2 - cols // 2
            starty = y // 2 - rows // 2
            return res[starty:starty + rows, startx:startx + cols]
        elif params < 1:  # need to pad
            sty = (rows - y) / 2
            stx = (cols - x) / 2
            return np.pad(res, [(sty, rows - y - sty), (stx, cols - x - stx), (0, 0)], mode='constant',
                          constant_values=0)
        return res

    def image_shear(img, params):
        rows, cols, ch = img.shape
        # rows, cols = img.shape
        factor = params * (-1.0)
        M = np.float32([[1, factor, 0], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def image_rotation(img, params):
        rows, cols, ch = img.shape
        # rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
        dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_AREA)
        return dst

    def image_contrast(img, params):
        alpha = params
        new_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha
        # new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

        return new_img

    def image_brightness(img, params):
        beta = params
        new_img = cv2.add(img, beta)  # new_img = img*alpha + beta
        return new_img

    def image_blur(img, params):

        # print("blur")
        blur = []
        if params == 1:
            blur = cv2.blur(img, (3, 3))
        if params == 2:
            blur = cv2.blur(img, (4, 4))
        if params == 3:
            blur = cv2.blur(img, (5, 5))
        if params == 4:
            blur = cv2.GaussianBlur(img, (3, 3), 0)
        if params == 5:
            blur = cv2.GaussianBlur(img, (5, 5), 0)
        if params == 6:
            blur = cv2.GaussianBlur(img, (7, 7), 0)
        if params == 7:
            blur = cv2.medianBlur(img, 3)
        if params == 8:
            blur = cv2.medianBlur(img, 5)
        # if params == 9:
        #     blur = cv2.blur(img, (6, 6))
        if params == 9:
            blur = cv2.bilateralFilter(img, 6, 50, 50)
            # blur = cv2.bilateralFilter(img, 9, 75, 75)
        return blur

    def image_pixel_change(img, params):
        # random change 1 - 5 pixels from 0 -255
        img_shape = img.shape
        img1d = np.ravel(img)
        arr = np.random.randint(0, len(img1d), params)
        for i in arr:
            img1d[i] = np.random.randint(0, 256)
        new_img = img1d.reshape(img_shape)
        return new_img

    def image_noise(img, params):
        if params == 1:  # Gaussian-distributed additive noise.
            row, col, ch = img.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = img + gauss
            return noisy.astype(np.uint8)
        elif params == 2:  # Replaces random pixels with 0 or 1.
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(img)
            # Salt mode
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt))
                      for i in img.shape]
            out[tuple(coords)] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper))
                      for i in img.shape]
            out[tuple(coords)] = 0
            return out
        elif params == 3:  # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
            row, col, ch = img.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = img + img * gauss
            return noisy.astype(np.uint8)















    transformations = [image_translation, image_scale, image_shear, image_rotation,
                       image_contrast, image_brightness, image_blur, image_pixel_change, image_noise]

    # these parameters need to be carefullly considered in the experiment
    # to consider the feedbacks
    params = []
    params.append(list(range(-3, 3)))  # image_translation
    params.append(list(map(lambda x: x * 0.1, list(range(7, 12)))))  # image_scale
    params.append(list(map(lambda x: x * 0.1, list(range(-6, 6)))))  # image_shear
    params.append(list(range(-50, 50)))  # image_rotation
    params.append(list(map(lambda x: x * 0.1, list(range(5, 13)))))  # image_contrast
    params.append(list(range(-20, 20)))  # image_brightness
    params.append(list(range(1, 10)))  # image_blur
    params.append(list(range(1, 10)))  # image_pixel_change
    params.append(list(range(1, 4)))  # image_noise

    classA = [7, 8]  # pixel value transformation
    classB = [0, 1, 2, 3, 4, 5, 6] # Affine transformation

    @staticmethod
    def mutate_one(ref_img, img, cl, l0_ref, linf_ref, try_num=50):

        # ref_img is the reference image, img is the seed

        # cl means the current state of transformation
        # 0 means it can select both of Affine and Pixel transformations
        # 1 means it only select pixel transformation because an Affine transformation has been used before

        # l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
        # between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1}) in Equation 2 of the paper

        # tyr_num is the maximum number of trials in Algorithm 2


        x, y, z = img.shape

        # a, b is the alpha and beta in Equation 1 in the paper
        a = 0.02
        b = 0.20

        # l0: alpha * size(s), l_infinity: beta * 255 in Equation 1
        l0 = int(a * x * y * z)
        l_infinity = int(b * 255)

        ori_shape = ref_img.shape
        for ii in range(try_num):
            random.seed(time.time())
            if cl == 0:  # 0: can choose class A and B
                tid = random.sample(Mutators.classA + Mutators.classB, 1)[0]
                # Randomly select one transformation   Line-7 in Algorithm2
                transformation = Mutators.transformations[tid]
                params = Mutators.params[tid]
                # Randomly select one parameter Line 10 in Algo2
                param = random.sample(params, 1)[0]

                # Perform the transformation  Line 11 in Algo2
                img_new = transformation(copy.deepcopy(img), param)
                img_new = img_new.reshape(ori_shape)

                if tid in Mutators.classA:
                    sub = ref_img - img_new
                    # check whether it is a valid mutation. i.e., Equation 1 and Line 12 in Algo2
                    l0_ref = np.sum(sub != 0)
                    linf_ref = np.max(abs(sub))
                    if l0_ref < l0 or linf_ref < l_infinity:
                        return ref_img, img_new, 0, 1, l0_ref, linf_ref
                else:  # B, C
                    # If the current transformation is an Affine trans, we will update the reference image and
                    # the transformation state of the seed.
                    ref_img = transformation(copy.deepcopy(ref_img), param)
                    ref_img = ref_img.reshape(ori_shape)
                    return ref_img, img_new, 1, 1, l0_ref, linf_ref
            if cl == 1: # 0: can choose class A
                tid = random.sample(Mutators.classA, 1)[0]
                transformation = Mutators.transformations[tid]
                params = Mutators.params[tid]
                param = random.sample(params, 1)[0]
                img_new = transformation(copy.deepcopy(img), param)
                sub = ref_img - img_new

                # To compute the value in Equation 2 in the paper.
                l0_new = l0_ref +  np.sum(sub != 0)
                linf_new = max(linf_ref , np.max(abs(sub)))

                if  l0_new < l0 or linf_new < l_infinity:
                    return ref_img, img_new, 1, 1, l0_ref, linf_ref
        # Otherwise the mutation is failed. Line 20 in Algo 2
        return ref_img, img, cl, 0, l0_ref, linf_ref

    @staticmethod
    def mutate_without_limitation(ref_img):

        tid = random.sample(Mutators.classA + Mutators.classB, 1)[0]
        transformation = Mutators.transformations[tid]
        ori_shape = ref_img.shape
        params = Mutators.params[tid]
        param = random.sample(params, 1)[0]
        img_new = transformation(ref_img, param)
        img_new = img_new.reshape(ori_shape)
        return img_new
    @staticmethod
    #Algorithm 2
    def image_random_mutate(seed, batch_num):

        test = np.load(seed.fname)
        ref_img = test[0]
        img = test[1]
        cl = seed.clss
        ref_batches = []
        batches = []
        cl_batches = []
        l0_ref_batches = []
        linf_ref_batches = []
        for i in range(batch_num):
            ref_out, img_out, cl_out, changed, l0_ref, linf_ref = Mutators.mutate_one(ref_img, img, cl, seed.l0_ref, seed.linf_ref)
            if changed:
                ref_batches.append(ref_out)
                batches.append(img_out)
                cl_batches.append(cl_out)
                l0_ref_batches.append(l0_ref)
                linf_ref_batches.append(linf_ref)

        return np.asarray(ref_batches), np.asarray(batches), cl_batches, l0_ref_batches, linf_ref_batches


    # Text mutations
    # Swap two sentences.
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
    @staticmethod
    def get_rand_synonym(word):
        try:
            synsets = wordnet.synsets(word)
        except LookupError:
            # Need to download wordnet once.
            nltk_download('wordnet')
            synsets = wordnet.synsets(word)
        synonym_list = map(lambda x: x.lemmas()[0].name().encode(), synsets)
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
        words[replace_word_index] = Mutators.get_rand_synonym(words[replace_word_index]).replace('_', ' ')
        detokenizer = TreebankWordDetokenizer()
        return detokenizer.detokenize(words)

    # Character-level mutations: add, swap, delete.
    @staticmethod
    def get_rand_char():
        return chr(97 + np.random.randint(0, 26))

    @staticmethod
    def add_rand_char(text):
        rand_char = Mutators.get_rand_char()
        # Include index past last char so we can add new char at end.
        rand_loc = np.random.randint(0, len(text) + 1)
        return text[:rand_loc] + rand_char + text[rand_loc:]

    @staticmethod
    def del_rand_char(text):
        rand_loc = np.random.randint(0, len(text))
        return text[:rand_loc] + text[rand_loc+1:]

    @staticmethod
    def sub_rand_char(text):
        rand_char = Mutators.get_rand_char()
        rand_loc = np.random.randint(0, len(text))
        return text[:rand_loc] + rand_char + text[rand_loc+1:]

    # Either add, delete, or substitute a random character.
    def mutate_char(text, params):
        mutation_funcs = [
            Mutators.add_rand_char,
            Mutators.del_rand_char,
            Mutators.sub_rand_char]
        return mutation_funcs[params](text)


    text_mutation_ids = [0, 1]
    text_char_mut_ids = [2]
    # Fll in with method names of text transformations
    text_transformations = [rearrange_sentences, sub_synonym, mutate_char]

    text_params = []
    text_params.append([None])  # rearrange_sentences
    text_params.append([None])  # sub_synonym
    text_params.append(list(range(0, 3)))  # mutate_char


    @staticmethod
    def text_mutate_one(ref_text, text, cl, l0_ref, linf_ref, try_num=50):

        # ref_text is the reference text, text is the seed

        # Note - This variable is from image mutations. Might use it for some kind of text transformation or might ignore it.
        # cl means the current state of transformation
        # 0 means it can select both of Affine and Pixel transformations
        # 1 means it only select pixel transformation because an Affine transformation has been used before

        # l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
        # between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1}) in Equation 2 of the paper

        # tyr_num is the maximum number of trials in Algorithm 2

        # a, b is the alpha and beta in Equation 1 in the paper
        # a is the fraction of pixels we are allowing to change
        # b is the fraction of the color space that we allow a pixel value to change by
        # a = 0.02
        # b = 0.20

        # l0: alpha * size(s), l_infinity: beta * 255 in Equation 1
        # l0 is the max number of changed pixels between an image and its mutant
        # Note: Might use l0 as the number of words changed.
        # l0 = int(a * x * y * z)
        # l_infinity is the max value that a pixel can change by
        # l_infinity = int(b * 255)

        # Analog for a, b, l0, l_infinity:
        # Will ignore them for now, but could use them for
        # max number of words allowed to change between two strings
        # or any measure of how much a seed has been changed from the original

        for ii in range(try_num):
            random.seed(time.time())
            tid = random.sample(
                Mutators.text_mutation_ids + Mutators.text_char_mut_ids, 1)[0]
            transformation = Mutators.text_transformations[tid]
            params = Mutators.text_params[tid]
            param = random.sample(params, 1)[0]
            text_new = transformation(text, param)

            # Optional: compute new l0 and linf values and check if
            # they're within range (less than previous values)

            if (text_new != text):
                return ref_text, text_new, 1, 1, l0_ref, linf_ref

        # Otherwise the mutation is failed. Line 20 in Algo 2
        return ref_text, text, cl, 0, l0_ref, linf_ref

    @staticmethod
    def text_random_mutate(seed, batch_num):

        test = np.load(seed.fname)
        ref_text = test[0]
        text = test[1]
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


if __name__ == '__main__':
    print("main Test.")
