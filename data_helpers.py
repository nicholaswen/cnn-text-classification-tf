import numpy as np
import re
import pandas as pd

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_sarc(input_file, training, sample_percent=1.0):
    reddit = pd.read_csv(input_file)
    
    sample_index = int(len(reddit) * sample_percent)

    labels = reddit['label'].values
    labels = labels[:sample_index]
    labels = [[0, 1] if l == 1 else [1, 0] for l in labels]
    split_index = int(len(labels) * 0.7)
    train_labels, test_labels = labels[:split_index], labels[split_index:]
    sarcastic = 0
    for label in test_labels:
        if label == [0, 1]: sarcastic += 1

    # Process data
    text = reddit['comment'].values
    text = [str(x) for x in text]
    text = text[:sample_index]
    
    train_text, test_text = text[:split_index], text[split_index:]

    return [train_text, np.array(train_labels)] if training else [test_text, np.array(test_labels)]

def load_data_ghosh(input_file):
    with open(input_file) as f:
        twitter = f.readlines()
    twitter = [x.strip() for x in twitter]

    twitter = pd.DataFrame(twitter)

    new = twitter[0].str.split("\t", n = 2, expand = True)
    twitter_labels = new[1]
    twitter_text = new[2]
    twitter_text = [tweet for tweet in twitter_text]
    
    twitter_labels = [[0, 1] if l is '1' else [1, 0] for l in twitter_labels]
    sarcastic = 0
    for label in twitter_labels:
        if label == [0, 1]: sarcastic += 1
    #print("Sarcastic Count: %d" % sarcastic)
    #print("Not Sarcastic Count: %d" % (len(twitter_labels)-sarcastic))
    twitter_labels = np.array(twitter_labels)
    return [twitter_text, twitter_labels]


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def batch_iter_one_epoch(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches = int((len(data)-1)/batch_size) + 1
    
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        print("Epoch: %d" % epoch)
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

#load_data_sarc('data/train-balanced-sarcasm.csv', True)
#load_data_ghosh('data/ghosh/train.txt')
