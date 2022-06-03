class PretrainedList:
    '''Latest copies of pre-trained bert models (as of June 2, 2022)

    An updated list is maintained under the official repo:
    https://github.com/google-research/bert
    '''

    # 12-layer, 768-hidden, 12-heads, 110M parameters
    base_uncased = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
    # 24-layer, 1024-hidden, 16-heads, 340M parameters
    large_uncased = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip"
    # 12-layer, 768-hidden, 12-heads, 110M parameters
    base_cased = "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"
    # 24-layer, 1024-hidden, 16-heads, 340M parameters
    large_cased = "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip"
    # 24-layer, 1024-hidden, 16-heads, 340M parameters
    large_uncased_whole = "https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip"
    # 24-layer, 1024-hidden, 16-heads, 340M parameters
    large_cased_whole = "https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip"
    # 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    multilingual_uncased = "https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip"
    # 102 languages, 24-layer, 1024-hidden, 16-heads, 340M parameters
    multilingual_cased = "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip"
    # 12-layer, 768-hidden, 12-heads, 110M parameters
    chinese = "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip"

if __name__ == '__main__':
    from keras_bert import (
        get_pretrained, 
        get_checkpoint_paths, 
        load_trained_model_from_checkpoint
    )

    # Download the model
    model_path = get_pretrained(PretrainedList.base_uncased)
    paths = get_checkpoint_paths(model_path)
    model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint,
                                               training=True)

    print(model.summary())




