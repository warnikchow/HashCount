#!/usr/bin/env python3

'''
example.py

Benchmark system for the SemEval-2018 Task 3 on Irony detection in English tweets.
The system makes use of token unigrams as features and outputs cross-validated F1-score.

Date: 1/09/2017
Copyright (c) Gilles Jacobs & Cynthia Van Hee, LT3. All rights reserved.
'''

from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import dump_svmlight_file
from sklearn import metrics
import numpy as np
import logging
import codecs

logging.basicConfig(level=logging.INFO)

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.lower().split('\t') for line in f.read().splitlines()]
    return data

def parse_dataset(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with open(fp, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)
    return corpus, y

def parse_dataset_utf(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with open(fp, 'rt',encoding='utf-8',errors='ignore') as data_in:
        for line in data_in:
            if not line.lower().startswith("+tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)
    return corpus, y

def parse_dataset_pred(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    #y = []
    corpus = []
    with open(fp, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                #label = int(line.split("\t")[1])
                tweet = line.split("\t")[1]
                #y.append(label)
                corpus.append(tweet)
    return corpus

def featurize(corpus):
    '''
    Tokenizes and creates TF-IDF BoW vectors.
    :param corpus: A list of strings each string representing document.
    :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
    '''
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names()) # to manually check if the tokens are reasonable
    return X

def loadvector(File):
    print("Loading word vectors")
    f = open(File,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

glove_twit = loadvector('glove100.txt')
wdim = 100
maxlen = 30
maxtaglen = 20
import re
import nltk
from nltk.corpus import stopwords
splithash = re.compile(r"#(\w+)")

def cleanupDoc(s):
    stopset = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(s)
    cleanup = " ".join(filter(lambda word: word not in stopset, s.split()))
    return cleanup

# https://www.researchgate.net/post/Is_there_a_way_in_PreProcessing_data_to_split_words_in_Hashtag

def split_hashtag(hashtagestring):
    fo = re.compile(r'#[A-Z]{2,}(?![a-z])|[A-Z][a-z]+')
    fi = fo.findall(hashtagestring)
    result = ''
    for var in fi:
        result += var + ' '
    return result

def hashtovec(word):
    split = split_hashtag(word)
    split = nltk.word_tokenize(split)
    if len(split)>0:
        result=np.zeros((len(split),wdim))
        for i in range(len(split)):
            if split[i].lower() in glove_twit:
                result[i,:] = glove_twit[split[i].lower()]
                #count = count+1
                #sum = sum+glove_twit[split[i].lower()]
    else:
        sum=np.zeros(wdim)
        count=0
        for i in range(len(word)):
            if word[i].lower() in glove_twit:
                count = count+1
                sum = sum+glove_twit[word[i].lower()]
        if count>0:
            sum = sum/count
        result=sum
    return result

def hashtovec_sum(word):
    split = split_hashtag(word)
    split = nltk.word_tokenize(split)
    sum=np.zeros(wdim)
    count=0
    if len(split)>0:
        for i in range(len(split)):
            if split[i].lower() in glove_twit:
                count = count+1
                sum = sum+glove_twit[split[i].lower()]
        if count>0:
            sum = sum/count
    else:
        for i in range(len(word)):
            if word[i].lower() in glove_twit:
                count = count+1
                sum = sum+glove_twit[word[i].lower()]
        if count>0:
            sum = sum/count
    return sum

def featurize_text(corpus):
    rec = np.zeros((len(corpus),maxlen,wdim))
    conv = np.zeros((len(corpus),maxlen,wdim,1))
    result1 = np.zeros((len(corpus),maxlen,wdim))
    result2 = np.zeros((len(corpus),maxlen,wdim,1))
    for i in range(len(corpus)):
        #if i%1000 ==0:
        print(i)
        s = nltk.word_tokenize(corpus[i])
        stag = splithash.findall(corpus[i])
        for j in range(len(s)):
            if s[j].lower() in glove_twit and j<maxlen:
                print(s[j])
                rec[i][j,:]=glove_twit[s[j].lower()]
                conv[i][j,:,0]=glove_twit[s[j].lower()]
    for i in range(len(corpus)):
        result1[i] = rec[i]
        result2[i] = conv[i]
    return result1, result2

def featurize_seq(corpus):
    rec = np.zeros((len(corpus),maxlen,wdim))
    rectag = np.zeros((len(corpus),maxtaglen,wdim))
    conv = np.zeros((len(corpus),maxlen,wdim,1))
    convtag = np.zeros((len(corpus),maxtaglen,wdim,1))
    result1 = np.zeros((len(corpus),maxlen+maxtaglen,wdim))
    result2 = np.zeros((len(corpus),maxlen+maxtaglen,wdim,1))
    for i in range(len(corpus)):
        #if i%1000 ==0:
        print(i)
        s = nltk.word_tokenize(corpus[i])
    	stag = splithash.findall(corpus[i])    
        for j in range(len(s)):
            if s[j].lower() in glove_twit and j<maxlen:
                print(s[j])
                rec[i][j,:]=glove_twit[s[j].lower()]
                conv[i][j,:,0]=glove_twit[s[j].lower()]
        count=0
        for j in range(len(stag)):
            if stag[j].lower() in glove_twit and count<maxtaglen:
                print(stag[j])
                rectag[i][count,:]=glove_twit[stag[j].lower()]
                convtag[i][count,:,0]=glove_twit[stag[j].lower()]
                count=count+1
            else:
                split = split_hashtag(stag[j])
                split = nltk.word_tokenize(split)
                print(stag[j])
                if len(split)==0 and count<maxtaglen:
                    rectag[i][count,:]=hashtovec(stag[j])
                    convtag[i][count,:,0]=hashtovec(stag[j])
                    count=count+1
                elif len(split)>0 and count+len(split)-1<maxtaglen:
                    rectag[i][count:count+len(split),:]=hashtovec(stag[j])
                    convtag[i][count:count+len(split),:,0]=hashtovec(stag[j])
                    count=count+len(split)
    for i in range(len(corpus)):
        result1[i] = np.concatenate((rec[i],rectag[i]))
        result2[i] = np.concatenate((conv[i],convtag[i]))
    return result1, result2
   
def featurize_sum_seq(corpus):
    rec = np.zeros((len(corpus),maxlen,wdim))
    rectag = np.zeros((len(corpus),maxtaglen,wdim))
    conv = np.zeros((len(corpus),maxlen,wdim,1))
    convtag = np.zeros((len(corpus),maxtaglen,wdim,1))
    result1 = np.zeros((len(corpus),maxlen+maxtaglen,wdim))
    result2 = np.zeros((len(corpus),maxlen+maxtaglen,wdim,1))
    for i in range(len(corpus)):
        #if i%1000 ==0:
        print(i)
        s = nltk.word_tokenize(corpus[i])
        stag = splithash.findall(corpus[i])
        for j in range(len(s)):
            if s[j].lower() in glove_twit and j<maxlen:
                print(s[j])
                rec[i][j,:]=glove_twit[s[j].lower()]
                conv[i][j,:,0]=glove_twit[s[j].lower()]
        for j in range(len(stag)):
            if stag[j].lower() in glove_twit and j<maxtaglen:
                print(stag[j])
                rectag[i][j,:]=glove_twit[stag[j].lower()]
                convtag[i][j,:,0]=glove_twit[stag[j].lower()]
            elif j<maxtaglen:
                split = split_hashtag(stag[j])
                split = nltk.word_tokenize(split)
                print(stag[j])
                rectag[i][j,:]=hashtovec_sum(stag[j])
                convtag[i][j,:,0]=hashtovec_sum(stag[j])
    for i in range(len(corpus)):
        result1[i] = np.concatenate((rec[i],rectag[i]))
        result2[i] = np.concatenate((conv[i],convtag[i]))
    return result1, result2

result1 = featurize_seq(corpus)
result2_rec,result2_conv = featurize_sum_seq(corpus)
result2_rec_emoji,result2_conv_emoji = featurize_sum_seq(corpus)
result2_rec_pred, result2_conv_pred = featurize_sum_seq(corpus_pred)
result_multi_rec, result_multi_conv = featurize_sum_seq(corpus)
result_multi_rec_emoji, result_multi_conv_emoji = featurize_sum_seq(corpus)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)
adam_half_2 = optimizers.Adam(lr=0.0002)

from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
import keras.layers as layers

from keras.layers import GRU, LSTM, SimpleRNN

from keras.callbacks import ModelCheckpoint

##### f1 score ftn.
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2

class Metrics(Callback):
 def on_train_begin(self, logs={}):
  self.val_f1s = []
  self.val_recalls = []
  self.val_precisions = []
 def on_epoch_end(self, epoch, logs={}):
  val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
  val_targ = self.validation_data[1]
  _val_f1 = f1_score(val_targ, val_predict)
  _val_recall = recall_score(val_targ, val_predict)
  _val_precision = precision_score(val_targ, val_predict)
  self.val_f1s.append(_val_f1)
  self.val_recalls.append(_val_recall)
  self.val_precisions.append(_val_precision)
  print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))

metricsf1 = Metrics()

def y_onehot(y,label):
    y_onehot = np.zeros((len(y),label))
    for i in range(len(y)):
        y_onehot[i,y[i]]=1
    return y_onehot

from keras import backend as K

class Metricsf1macro(Callback):
 def on_train_begin(self, logs={}):
  self.val_f1s = []
  self.val_recalls = []
  self.val_precisions = []
 def on_epoch_end(self, epoch, logs={}):
  val_predict = np.asarray(self.model.predict(self.validation_data[0]))
  val_predict = np.argmax(val_predict,axis=1)
  val_targ = self.validation_data[1]
  _val_f1 = metrics.f1_score(val_targ, val_predict, average="macro")
  _val_recall = metrics.recall_score(val_targ, val_predict, average="macro")
  _val_precision = metrics.precision_score(val_targ, val_predict, average="macro")
  self.val_f1s.append(_val_f1)
  self.val_recalls.append(_val_recall)
  self.val_precisions.append(_val_precision)
  print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))
  #print("— val_f1: %f"%(_val_f1))

metricsf1macro = Metricsf1macro()

from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

def validate_bilstm(result,y,filename):
    model_bi = Sequential()
    model_bi.add(Bidirectional(LSTM(32), input_shape=(len(result[0]), wdim)))
    model_bi.add(Dense(1, activation='sigmoid'))
    model_bi.compile(optimizer=adam_half, loss="binary_crossentropy", metrics=['accuracy'])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1,checkpoint]
    model_bi.fit(result,y,validation_split=0.1,epochs=20,batch_size=16,callbacks=callbacks_list)

def validate_bilstm4(result,y,filename):
    model_bi = Sequential()
    model_bi.add(Bidirectional(LSTM(32), input_shape=(len(result[0]), wdim)))
    model_bi.add(Dense(4, activation='softmax'))
    model_bi.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro,checkpoint]
    model_bi.fit(result,y,validation_split=0.1,epochs=20,batch_size=16,callbacks=callbacks_list)

def validate_cnn(result,y,filename):
    model_conv = Sequential()
    model_conv.add(layers.Conv2D(32,(3,wdim),activation= 'relu',input_shape = (len(result[0]),wdim,1)))
    model_conv.add(layers.MaxPooling2D((2,1)))
    model_conv.add(layers.Conv2D(32,(3,1),activation='relu'))
    model_conv.add(layers.Flatten())
    model_conv.add(layers.Dense(32,activation='relu'))
    model_conv.add(Dense(1, activation='sigmoid'))
    model_conv.summary()
    model_conv.compile(optimizer=adam_half, loss="binary_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1,checkpoint]
    model_conv.fit(result,y,validation_split=0.1,epochs=20,batch_size=16,callbacks=callbacks_list)

def validate_cnn4(result,y,filename):
    model_conv = Sequential()
    model_conv.add(layers.Conv2D(32,(3,wdim),activation= 'relu',input_shape = (len(result[0]),wdim,1)))
    model_conv.add(layers.MaxPooling2D((2,1)))
    model_conv.add(layers.Conv2D(32,(3,1),activation='relu'))
    model_conv.add(layers.Flatten())
    model_conv.add(layers.Dense(32,activation='relu'))
    model_conv.add(Dense(4, activation='softmax'))
    model_conv.summary()
    model_conv.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro,checkpoint]
    model_conv.fit(result,y,validation_split=0.1,epochs=20,batch_size=16,callbacks=callbacks_list)

from keras.models import load_model

def make_prediction_A(corpus):
    result_rec, result_conv = featurize_sum_seq(corpus)
    model = load_model('models/bilstm_sum_best_6824_6849.hdf5')
    pred = model.predict(result_rec)
    TASK = "A" # Define, A or B
    FNAME = './predictions-task' + TASK + '.txt'
    PREDICTIONSFILE = open(FNAME, "w")
    for i in range(len(pred)):
        if pred[i] >=0.5:
            PREDICTIONSFILE.write('1\n')
        else:
            PREDICTIONSFILE.write('0\n')
    PREDICTIONSFILE.close()  

def make_prediction_B(corpus):
    result_rec, result_conv = featurize_sum_seq(corpus)
    model = load_model('models/conv4_sum_best_3937_5521.hdf5')
    pred = model.predict(result_conv)
    pred = np.argmax(pred,axis=1)
    TASK = "B"
    FNAME = './predictions-task'+TASK+'.txt'
    PREDICTIONSFILE = open(FNAME, "w")
    for i in range(len(pred)):
        PREDICTIONSFILE.write(str(pred[i])+'\n')
    PREDICTIONSFILE.close()

if __name__ == "__main__":
    # Experiment settings

    # Dataset: SemEval2018-T4-train-taskA.txt or SemEval2018-T4-train-taskB.txt
    DATASET_FP = "./SemEval2018-T3-train-taskA.txt"
    TASK = "A" # Define, A or B
    FNAME = './predictions-task' + TASK + '.txt'
    PREDICTIONSFILE = open(FNAME, "w")

    K_FOLDS = 10 # 10-fold crossvalidation
    CLF = LinearSVC() # the default, non-parameter optimized linear-kernel SVM

    # Loading dataset and featurised simple Tfidf-BoW model
    corpus, y = parse_dataset(DATASET_FP)
    X = featurize(corpus)

    class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()
    print (class_counts)
    
    # Returns an array of the same size as 'y' where each entry is a prediction obtained by cross validated
    predicted = cross_val_predict(CLF, X, y, cv=K_FOLDS)
    
    # Modify F1-score calculation depending on the task
    if TASK.lower() == 'a':
        score = metrics.f1_score(y, predicted, pos_label=1)
    elif TASK.lower() == 'b':
        score = metrics.f1_score(y, predicted, average="macro")
    print ("F1-score Task", TASK, score)
    for p in predicted:
        PREDICTIONSFILE.write("{}\n".format(p))
    PREDICTIONSFILE.close()
    
    
    
