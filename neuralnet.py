from importlib import reload
import os

import time
import datetime
import csv
import json

# lybraries
import pandas as pd
import numpy as np

# natural language toolkit imports
#import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# metrics
from metrics import map_score_filtered, map_score
from sklearn.metrics import roc_auc_score

from itertools import chain

# embedding models
from gensim.models import KeyedVectors

# Keras imports
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras.layers import (Input, Embedding, Convolution1D, Dropout, SpatialDropout1D, dot,
                            Reshape, Merge, Flatten, Concatenate,
                            GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, Dense)

max_len_q = 40 #25
max_len_a = 40 # 100

models_dir = ('project/models/')
data_dir = ('project/data/')
results_dir = ('project/results/')
if not os.path.isdir(models_dir):
    os.makedirs(models_dir)
    print("Home directory %s was created." %models_dir)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
    print("Home directory %s was created." %results_dir)


def intarray(a):
    a = a.replace('\n', '')
    return list(map(int,a.split(',')))

d = 'Unnamed: 0'
# LOAD DATASETS AS PANDAS DATAFRAMES
train = pd.read_csv(data_dir+'train_embeddings.csv', sep=',')
if d in train:
    train = train.drop(columns=d)
train['Q_W2V'] = train['Q_W2V'].map(intarray)
train['A_W2V'] = train['A_W2V'].map(intarray)
train['Q_FT'] = train['Q_FT'].map(intarray)
train['A_FT'] = train['A_FT'].map(intarray)
train['Q_OV'] = train['Q_OV'].map(intarray)
train['A_OV'] = train['A_OV'].map(intarray)
train['Q_POS'] = train['Q_POS'].map(intarray)
train['A_POS'] = train['A_POS'].map(intarray)
train['Q_BC'] = train['Q_BC'].map(intarray)
train['A_BC'] = train['A_BC'].map(intarray)

# print("AFTER")
# print(train['Q_W2V'])

dev = pd.read_csv(data_dir+'dev_embeddings.csv', sep=',')
if d in dev:
    dev = dev.drop(columns=d)
dev['Q_W2V'] = dev['Q_W2V'].map(intarray)
dev['A_W2V'] = dev['A_W2V'].map(intarray)
dev['Q_FT'] = dev['Q_FT'].map(intarray)
dev['A_FT'] = dev['A_FT'].map(intarray)
dev['Q_OV'] = dev['Q_OV'].map(intarray)
dev['A_OV'] = dev['A_OV'].map(intarray)
dev['Q_POS'] = dev['Q_POS'].map(intarray)
dev['A_POS'] = dev['A_POS'].map(intarray)
dev['Q_BC'] = dev['Q_BC'].map(intarray)
dev['A_BC'] = dev['A_BC'].map(intarray)


test = pd.read_csv(data_dir+'test_embeddings.csv', sep=',')
if d in test:
    test = test.drop(columns=d)
test['Q_W2V'] = test['Q_W2V'].map(intarray)
test['A_W2V'] = test['A_W2V'].map(intarray)
test['Q_FT'] = test['Q_FT'].map(intarray)
test['A_FT'] = test['A_FT'].map(intarray)
test['Q_OV'] = test['Q_OV'].map(intarray)
test['A_OV'] = test['A_OV'].map(intarray)
test['Q_POS'] = test['Q_POS'].map(intarray)
test['A_POS'] = test['A_POS'].map(intarray)
test['Q_BC'] = test['Q_BC'].map(intarray)
test['A_BC'] = test['A_BC'].map(intarray)


# LOAD EMBEDDING MATRICES
em = np.load(data_dir+"matrices.npz")
em_W2V = em['w2v']
em_FT = em['ft']
em_POS = em['pos']
em_BC = em['bc']

with open(data_dir+'dictW2V.json') as f:
    dict_w2v = json.load(f)
with open(data_dir+'dictFT.json') as f:
    dict_ft = json.load(f)
with open(data_dir+'dictPOS.json') as f:
    dict_pos = json.load(f)
with open(data_dir+'dictBC.json') as f:
    dict_bc = json.load(f)



# verticalize and list dataset without overlaps
def data(dataset, col_q, col_a):
    return (dataset['QuestionID'],dataset['SentenceID']), [np.vstack(dataset[col_q]), np.vstack(dataset[col_a])], np.vstack(dataset['Label'].tolist())
# verticalize and list dataset with overlaps
def dataOver(dataset, col_q, col_a):
    return (dataset['QuestionID'],dataset['SentenceID']), [np.vstack(dataset[col_q]), np.vstack(dataset[col_a]),np.vstack(dataset['Q_OV']), np.vstack(dataset['A_OV']), dataset['count'].values], np.vstack(dataset['Label'].tolist())

def dataTwo(dataset, col_q_1, col_a_1, col_q_2, col_a_2):
    return (dataset['QuestionID'],dataset['SentenceID']), [np.vstack(dataset[col_q_1]), np.vstack(dataset[col_a_1]), np.vstack(dataset[col_q_2]), np.vstack(dataset[col_a_2])], np.vstack(dataset['Label'].tolist())

def dataTwoOv(dataset, col_q_1, col_a_1, col_q_2, col_a_2):
    return (dataset['QuestionID'],dataset['SentenceID']), [np.vstack(dataset[col_q_1]), np.vstack(dataset[col_a_1]), np.vstack(dataset[col_q_2]), np.vstack(dataset[col_a_2]), np.vstack(dataset['Q_OV']), np.vstack(dataset['A_OV']), dataset['count'].values], np.vstack(dataset['Label'].tolist())

def dataAll(dataset, col_q, col_a):
    return (dataset['QuestionID'],dataset['SentenceID']), [np.vstack(dataset[col_q]), np.vstack(dataset[col_a]), np.vstack(dataset['Q_POS']), np.vstack(dataset['A_POS']), np.vstack(dataset['Q_BC']), np.vstack(dataset['A_BC'])], np.vstack(dataset['Label'].tolist())

def dataAllOv(dataset, col_q, col_a):
    return (dataset['QuestionID'],dataset['SentenceID']), [np.vstack(dataset[col_q]), np.vstack(dataset[col_a]), np.vstack(dataset['Q_POS']), np.vstack(dataset['A_POS']), np.vstack(dataset['Q_BC']), np.vstack(dataset['A_BC']), np.vstack(dataset['Q_OV']), np.vstack(dataset['A_OV']), dataset['count'].values], np.vstack(dataset['Label'].tolist())


# In this setting, accuracy is not a relevant metric for classification. 
# For this reason, we create a custom Callback to implement early stopping and model saving.
class EpochEval(Callback):

    def __init__(self, validation_data, evaluate, patience=np.Inf, save_model=False, score_c=None):
        #super(Callback, self).__init__()
        super().__init__()

        (self.qids, self.aids), self.X, self.y = validation_data
        self.evaluate = evaluate
        self.best = -np.Inf
        self.patience = patience
        self.wait = 0
        self.waited = False
        self.save_model = save_model


    def on_epoch_end(self, epoch, logs={}):
        print
        prediction = self.model.predict(self.X)
        val = self.evaluate(self.qids, self.y, prediction)
        print("\t{0} = {1:.4f}".format(self.evaluate.__name__, val))
        if val*0.995 > self.best:
            self.model.save('qa.h5')
            print ('\tBest {0}: {1:.4f}'.format(self.evaluate.__name__, val))
            self.best = val
            self.wait = 0
            self.waited = False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
        print


que = Input(shape=(max_len_q,))
ans = Input(shape=(max_len_a,))
feat_q = Input(shape=(max_len_q,))
feat_a = Input(shape=(max_len_a,))
bcf_q = Input(shape=(max_len_q,))
bcf_a = Input(shape=(max_len_a,))

que_ov = Input(shape=(max_len_q,))
ans_ov = Input(shape=(max_len_a,))
cnt = Input(shape=(1,))

que_pos = Input(shape=(max_len_q,))
ans_pos = Input(shape=(max_len_a,))
que_bc = Input(shape=(max_len_q,))
ans_bc = Input(shape=(max_len_a,))


# embeddings for models with no overlap and without update
que_w2v_emb_no = Embedding(len(dict_w2v), 50, input_length=max_len_q, weights=[em_W2V], trainable=False)
ans_w2v_emb_no = Embedding(len(dict_w2v), 50, input_length=max_len_a, weights=[em_W2V], trainable=False)
que_ft_emb_no = Embedding(len(dict_ft), 300, input_length=max_len_q, weights=[em_FT], trainable=False)
ans_ft_emb_no = Embedding(len(dict_ft), 300, input_length=max_len_a, weights=[em_FT], trainable=False)
# embeddings for models without update
que_pos_emb_no = Embedding(len(dict_pos), 20, input_length=max_len_q, weights=[em_POS], trainable=False)
ans_pos_emb_no = Embedding(len(dict_pos), 20, input_length=max_len_a, weights=[em_POS], trainable=False)
que_bc_emb_no = Embedding(len(dict_bc), 20, input_length=max_len_q, weights=[em_BC], trainable=False)
ans_bc_emb_no = Embedding(len(dict_bc), 20, input_length=max_len_a, weights=[em_BC], trainable=False)


# embeddings for models with no overlap and with update
que_w2v_emb_up = Embedding(len(dict_w2v), 50, input_length=max_len_q, weights=[em_W2V], trainable=True)
ans_w2v_emb_up = Embedding(len(dict_w2v), 50, input_length=max_len_a, weights=[em_W2V], trainable=True)
que_ft_emb_up = Embedding(len(dict_ft), 300, input_length=max_len_q, weights=[em_FT], trainable=True)
ans_ft_emb_up = Embedding(len(dict_ft), 300, input_length=max_len_a, weights=[em_FT], trainable=True)
# embeddings for models with update
que_pos_emb_up = Embedding(len(dict_pos), 20, input_length=max_len_q, weights=[em_POS], trainable=True)
ans_pos_emb_up = Embedding(len(dict_pos), 20, input_length=max_len_a, weights=[em_POS], trainable=True)
que_bc_emb_up = Embedding(len(dict_bc), 20, input_length=max_len_q, weights=[em_BC], trainable=True)
ans_bc_emb_up = Embedding(len(dict_bc), 20, input_length=max_len_a, weights=[em_BC], trainable=True)

# embeddings for overlap model
que_ov_emb = Embedding(3, 5,input_length=max_len_q)(que_ov)
ans_ov_emb = Embedding(3, 5,input_length=max_len_a)(ans_ov)

que_w2v_emb_ov = Embedding(len(dict_w2v), 50 , input_length=max_len_q, weights=[em_W2V], trainable=True)(que)
ans_w2v_emb_ov = Embedding(len(dict_w2v), 50 , input_length=max_len_a, weights=[em_W2V], trainable=True)(ans)
que_ft_emb_ov = Embedding(len(dict_ft), 300 , input_length=max_len_q, weights=[em_FT], trainable=True)(que)
ans_ft_emb_ov = Embedding(len(dict_ft), 300 , input_length=max_len_a, weights=[em_FT], trainable=True)(ans)


que_pos_emb_c = Embedding(len(dict_pos), 20, input_length=max_len_q, weights=[em_POS], trainable=True)(feat_q)
ans_pos_emb_c = Embedding(len(dict_pos), 20, input_length=max_len_a, weights=[em_POS], trainable=True)(feat_a)
que_bc_emb_c = Embedding(len(dict_bc), 20, input_length=max_len_q, weights=[em_BC], trainable=True)(bcf_q)
ans_bc_emb_c = Embedding(len(dict_bc), 20, input_length=max_len_a, weights=[em_BC], trainable=True)(bcf_a)

que_w2v_ov_emb = concatenate([que_ov_emb, que_w2v_emb_ov])
ans_w2v_ov_emb = concatenate([ans_ov_emb, ans_w2v_emb_ov])
que_ft_ov_emb = concatenate([que_ov_emb, que_ft_emb_ov])
ans_ft_ov_emb = concatenate([ans_ov_emb, ans_ft_emb_ov])

que_w2v_ov_pos_emb = concatenate([que_ov_emb, que_w2v_emb_ov, que_pos_emb_c])
ans_w2v_ov_pos_emb = concatenate([ans_ov_emb, ans_w2v_emb_ov, ans_pos_emb_c])
que_w2v_ov_bc_emb = concatenate([que_ov_emb, que_w2v_emb_ov, que_bc_emb_c])
ans_w2v_ov_bc_emb = concatenate([ans_ov_emb, ans_w2v_emb_ov, ans_bc_emb_c])


def create_classify(join, cl_dim):
    classify = Sequential()
    classify.add(Dense(100, activation='tanh', input_dim=cl_dim))
    classify.add(Dense(1, activation='sigmoid'))
    return classify(join)

###############################################################################################################
############################################## ONE FEATURE MODEL ##############################################
def create_1feat_model(emb_q, emb_a, col_q, col_a, newname, ep):
    np.random.seed(42)

    que_model = Sequential()
    que_model.add(emb_q) # add question embedding
    que_model.add(Convolution1D(100, 5, activation='tanh'))
    que_model.add(GlobalMaxPooling1D())

    ans_model = Sequential()
    ans_model.add(emb_a) # add question embedding
    ans_model.add(Convolution1D(100, 5, activation='tanh'))
    ans_model.add(GlobalMaxPooling1D())

    q_emb = que_model(que) #var for question model
    a_emb = ans_model(ans) #var for answer model
    join = concatenate([q_emb, a_emb]) #concatenate embedding of question and answers

    out = create_classify(join, 200)


    model = Model(inputs=[que, ans], outputs=[out])
    model.summary()
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    print("MODEL FIT 1 feat")
    model.fit([np.vstack(train[col_q]), np.vstack(train[col_a])], np.vstack(train['Label'].tolist()), 
            batch_size=100, epochs=100000, shuffle=True, verbose=2,
            callbacks=[EpochEval(data(dev, col_q, col_a), map_score_filtered, patience=ep)])

    nameb = models_dir + newname + '.h5'
    os.rename('qa.h5', nameb)
    del model
###############################################################################################################
######################################### ONE FEAT WITH OVERLAP MODEL #########################################
def create_1feat_ov_model(emb_q, emb_a, col_q, col_a, size, newname, ep):
    np.random.seed(42)

    que_model = Sequential()
    que_model.add(Convolution1D(100, 5, activation='tanh', kernel_initializer='lecun_uniform', input_shape=(max_len_q, size)))
    que_model.add(GlobalMaxPooling1D())

    ans_model = Sequential()
    ans_model.add(Convolution1D(100, 5, activation='tanh', kernel_initializer='lecun_uniform', input_shape=(max_len_a, size)))
    ans_model.add(GlobalMaxPooling1D())

    q_emb = que_model(emb_q)# q_ov_w2v) #var for question model
    a_emb = ans_model(emb_a)# a_ov_w2v) #var for answer model
    join = concatenate([q_emb, a_emb, cnt]) #concatenate embedding of question and answers

    out = create_classify(join, 201)

    model_ov = Model(inputs=[que, ans, que_ov, ans_ov, cnt], outputs=[out])
    model_ov.summary()
    model_ov.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    
    model_ov.fit([np.vstack(train[col_q]), np.vstack(train[col_a]), np.vstack(train['Q_OV']), np.vstack(train['A_OV']), train['count'].values],
            np.vstack(train['Label'].tolist()), batch_size=100, epochs=100000, shuffle=True, verbose=2,
            callbacks=[EpochEval(dataOver(dev, col_q, col_a), map_score_filtered, patience=ep)])

    name =models_dir + newname + '.h5'
    os.rename('qa.h5', name)
    del model_ov
###############################################################################################################
############################################## TWO FEATURE MODEL ##############################################
def create_2feat_model(emb_q, emb_a, emb_fq, emb_fa, col_q_1, col_a_1, col_q_2, col_a_2, newname, ep):
    np.random.seed(42)

    que_in = Sequential()
    que_in.add(emb_q)
    que_emb = que_in(que)

    featQ_in = Sequential()
    featQ_in.add(emb_fq)
    fq_emb = featQ_in(feat_q)

    q_matrix = concatenate([que_emb, fq_emb], axis=2)

    que_conc = Sequential()
    que_conc.add(Convolution1D(100, 5, activation='tanh', kernel_initializer='lecun_uniform', input_shape=(max_len_q, 70)))
    que_conc.add(GlobalAveragePooling1D())
    q_emb = que_conc(q_matrix)
    

    ans_in = Sequential()
    ans_in.add(emb_a)
    ans_emb = ans_in(ans)

    featA_in = Sequential()
    featA_in.add(emb_fa)
    fa_emb = featA_in(feat_a)

    a_matrix = concatenate([ans_emb, fa_emb], axis=2)


    ans_conc = Sequential()
    ans_conc.add(Convolution1D(100, 5, activation='tanh', kernel_initializer='lecun_uniform', input_shape=(max_len_a, 70)))
    ans_conc.add(GlobalAveragePooling1D())
    a_emb = ans_conc(a_matrix)

    qa_matrix = concatenate([q_emb, a_emb])
    out = create_classify(qa_matrix, 200)


    model = Model(inputs=[que, ans, feat_q, feat_a], outputs=[out])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

    # print("MODEL FIT")
    model.fit([np.vstack(train[col_q_1]), np.vstack(train[col_a_1]), np.vstack(train[col_q_2]), np.vstack(train[col_a_2])],
            np.vstack(train['Label'].tolist()), batch_size=100, epochs=100000, shuffle=True, verbose=2,
            callbacks=[EpochEval(dataTwo(dev, col_q_1, col_a_1, col_q_2, col_a_2), map_score_filtered, patience=ep)])

    nameb = models_dir + newname + '.h5'
    os.rename('qa.h5', nameb)
    del model
###############################################################################################################
######################################### TWO FEAT WITH OVERLAP MODEL #########################################
def create_2feat_ov_model(emb_q, emb_a, emb_fq, emb_fa, col_q_1, col_a_1, col_q_2, col_a_2, size, newname, ep):
    np.random.seed(42)

    featQ_in = Sequential()
    featQ_in.add(emb_fq)
    fq_emb = featQ_in(feat_q)

    q_matrix = concatenate([emb_q, fq_emb], axis=2)

    que_conc = Sequential()
    que_conc.add(Convolution1D(100, 5, activation='tanh', kernel_initializer='lecun_uniform', input_shape=(max_len_q, size)))
    que_conc.add(GlobalAveragePooling1D())
    q_emb = que_conc(q_matrix)

    featA_in = Sequential()
    featA_in.add(emb_fa)
    fa_emb = featA_in(feat_a)

    a_matrix = concatenate([emb_a, fa_emb], axis=2)


    ans_conc = Sequential()
    ans_conc.add(Convolution1D(100, 5, activation='tanh', kernel_initializer='lecun_uniform', input_shape=(max_len_a, size)))
    ans_conc.add(GlobalAveragePooling1D())
    a_emb = ans_conc(a_matrix)

    qa_matrix = concatenate([q_emb, a_emb, cnt])
    out = create_classify(qa_matrix, 201)


    model = Model(inputs=[que, ans, feat_q, feat_a, que_ov, ans_ov, cnt], outputs=[out])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])    


    model.fit([np.vstack(train[col_q_1]), np.vstack(train[col_a_1]), np.vstack(train[col_q_2]), np.vstack(train[col_a_2]),
            np.vstack(train['Q_OV']), np.vstack(train['A_OV']), train['count'].values], np.vstack(train['Label'].tolist()),
            batch_size=100, epochs=100000, shuffle=True, verbose=2,
            callbacks=[EpochEval(dataTwoOv(dev, col_q_1, col_a_1, col_q_2, col_a_2), map_score_filtered, patience=ep)])

    nameb = models_dir + newname + '.h5'
    os.rename('qa.h5', nameb)
    del model
###############################################################################################################
############################################### COMPLETE MODEL ################################################
def create_complete_model(emb_q, emb_a, pos_q, pos_a, bc_q, bc_a, col_q, col_a, size, newname, ep):
    np.random.seed(42)

    posQ_in = Sequential()
    posQ_in.add(pos_q)
    que_pos = posQ_in(feat_q)

    bcQ_in = Sequential()
    bcQ_in.add(bc_q)
    que_bc = bcQ_in(bcf_q)

    q_matrix = concatenate([emb_q, que_pos, que_bc], axis=2)

    que_conc = Sequential()
    que_conc.add(Convolution1D(100, 5, activation='tanh', kernel_initializer='lecun_uniform', input_shape=(max_len_q, size)))
    que_conc.add(GlobalAveragePooling1D())
    q_emb = que_conc(q_matrix)
    
    posA_in = Sequential()
    posA_in.add(pos_a)
    ans_pos = posA_in(feat_a)

    bcA_in = Sequential()
    bcA_in.add(bc_a)
    ans_bc = bcA_in(bcf_a)

    a_matrix = concatenate([emb_a, ans_pos, ans_bc], axis=2)


    ans_conc = Sequential()
    ans_conc.add(Convolution1D(100, 5, activation='tanh', kernel_initializer='lecun_uniform', input_shape=(max_len_a, size)))
    ans_conc.add(GlobalAveragePooling1D())
    a_emb = ans_conc(a_matrix)

    qa_matrix = concatenate([q_emb, a_emb, cnt])
    out = create_classify(qa_matrix, 201)

    model = Model(inputs=[que, ans, feat_q, feat_a, bcf_q, bcf_a, que_ov, ans_ov, cnt], outputs=[out])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])    


    model.fit([np.vstack(train[col_q]), np.vstack(train[col_a]), np.vstack(train['Q_POS']), np.vstack(train['A_POS']),
            np.vstack(train['Q_BC']), np.vstack(train['A_BC']), np.vstack(train['Q_OV']), np.vstack(train['A_OV']), train['count'].values], 
            np.vstack(train['Label'].tolist()), batch_size=100, epochs=100000, shuffle=True, verbose=2,
            callbacks=[EpochEval(dataAllOv(dev, col_q, col_a), map_score_filtered, patience=ep)])

    nameb = models_dir + newname + '.h5'
    os.rename('qa.h5', nameb)
    del model
###############################################################################################################

def get_results(name, colname, qid, X, lab, test):
    model = load_model(name)
    pred = model.predict(X)
    print()
    print(map_score_filtered(qid, lab, pred))
    print(map_score(qid, lab, pred))
    test[colname] = pd.Series(y for y in pred)
    del model

# ########################################### BASELINE MODEL - W2V ###########################################
# print("\n\n#################################### BASELINE - W2V - NO UPDATE #################################")
# now = datetime.datetime.now()
# print(now.strftime("%H:%M.%S"))
# name = 'w2v'
# # create_1feat_model(que_w2v_emb_no, ans_w2v_emb_no, 'Q_W2V', 'A_W2V', name, 5)
# (qid,_ ), X, lab = data(test, 'Q_W2V', 'A_W2V')
# get_results(models_dir+name+'.h5', 'pred_w2v', qid, X, lab, test)


# print("\n\n####################### BASELINE - W2V - NO UPDATE - INCREMENTED PATIENCE #######################")
# now = datetime.datetime.now()
# print(now.strftime("%H:%M.%S"))
# name = 'w2v_pat'
# # create_1feat_model(que_w2v_emb_no, ans_w2v_emb_no, 'Q_W2V', 'A_W2V', name, 30)
# (qid,_ ), X, lab = data(test, 'Q_W2V', 'A_W2V')
# get_results(models_dir+name+'.h5', name, qid, X, lab, test)

# print("\n\n################################# BASELINE - W2V - WITH UPDATE ##################################")
# now = datetime.datetime.now()
# print(now.strftime("%H:%M.%S"))
# name = 'w2v_up' 
# # create_1feat_model(que_w2v_emb_up, ans_w2v_emb_up, 'Q_W2V', 'A_W2V', name, 5)
# (qid,_ ), X, lab = data(test, 'Q_W2V', 'A_W2V')
# get_results(models_dir+name+'.h5', name , qid, X, lab, test)

# print("\n\n###################### BASELINE - W2V - WITH UPDATE - INCREMENTED PATIENCE ######################")
# now = datetime.datetime.now()
# print(now.strftime("%H:%M.%S"))
# name = 'w2v_up_pat'
# # create_1feat_model(que_w2v_emb_up, ans_w2v_emb_up, 'Q_W2V', 'A_W2V', name, 30)
# (qid,_ ), X, lab = data(test, 'Q_W2V', 'A_W2V')
# get_results(models_dir+name+'.h5', name, qid, X, lab, test)

# ########################################### BASELINE MODEL - W2V ###########################################
# ########################################### BASELINE MODEL - FT ############################################

# print("\n\n################################## BASELINE - FT - NO UPDATE ####################################")
# now = datetime.datetime.now()
# print(now.strftime("%H:%M.%S"))
# name = 'ft'
# # create_1feat_model(que_ft_emb_no, ans_ft_emb_no, 'Q_FT', 'A_FT', name, 5)
# (qid,_ ), X, lab = data(test, 'Q_FT', 'A_FT')
# get_results(models_dir+name+'.h5', 'pred_ft', qid, X, lab, test)

# print("\n\n################################# BASELINE - FT - WITH UPDATE ###################################")
# now = datetime.datetime.now()
# print(now.strftime("%H:%M.%S"))
# name = 'ft_up'
# # create_1feat_model(que_ft_emb_up, ans_ft_emb_up, 'Q_FT', 'A_FT', name, 5)
# (qid,_ ), X, lab = data(test, 'Q_FT', 'A_FT')
# get_results(models_dir+name+'.h5', name, qid, X, lab, test)

# ########################################### BASELINE MODEL - FT ############################################

# test.to_csv(path_or_buf=results_dir+'baseline_test.csv', sep=',', na_rep='', header=1, index=False, index_label=None, mode='w')
# test = test.drop(columns=['pred_w2v', 'w2v_pat', 'w2v_up', 'w2v_up_pat', 'pred_ft','ft_up'])

# ############################################## OVERLAP MODEL ###############################################
# print("\n\n################################## OVERLAP - W2V - WITH UPDATE ##################################")
# now = datetime.datetime.now()
# print(now.strftime("%H:%M.%S"))
# name = 'w2v_ov'
# create_1feat_ov_model(que_w2v_ov_emb, ans_w2v_ov_emb, 'Q_W2V', 'A_W2V', 55, name, 5)
# (qid,_ ), X, lab = dataOver(test, 'Q_W2V', 'A_W2V')
# get_results(models_dir+name+'.h5', name, qid, X, lab, test)

# ############################################## OVERLAP MODEL ###############################################

# test.to_csv(path_or_buf=results_dir+'overlap_test.csv', sep=',', na_rep='', header=1, index=False, index_label=None, mode='w')
# test = test.drop(columns='w2v_ov')

########################################### ONE FEAT MODEL - POS ###########################################
print("\n\n############################### ONE FEAT - W2V & POS - NO UPDATE ################################")
now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
name = 'w2v_pos'
# create_2feat_model(que_w2v_emb_no, ans_w2v_emb_no, que_pos_emb_no, ans_pos_emb_no, 'Q_W2V', 'A_W2V', 'Q_POS', 'A_POS', name, 5)
(qid,_ ), X, lab = dataTwo(test, 'Q_W2V', 'A_W2V', 'Q_POS', 'A_POS')
get_results(models_dir+name+'.h5', name, qid, X, lab, test)

print("\n\n##################### ONE FEAT - W2V & POS - NO UPDATE - INCREMENTED PATIENCE #####################")
now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
name = 'w2v_pos_pat'
# create_2feat_model(que_w2v_emb_no, ans_w2v_emb_no, que_pos_emb_no, ans_pos_emb_no, 'Q_W2V', 'A_W2V', 'Q_POS', 'A_POS', name, 30)
(qid,_ ), X, lab = dataTwo(test, 'Q_W2V', 'A_W2V', 'Q_POS', 'A_POS')
get_results(models_dir + name +'.h5', name, qid, X, lab, test)

print("\n\n############################### ONE FEAT - W2V & POS - WITH UPDATE ################################")
now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
name = 'w2v_pos_up'
# create_2feat_model(que_w2v_emb_up, ans_w2v_emb_up, que_pos_emb_up, ans_pos_emb_up, 'Q_W2V', 'A_W2V', 'Q_POS', 'A_POS', name, 5)
(qid,_ ), X, lab = dataTwo(test, 'Q_W2V', 'A_W2V', 'Q_POS', 'A_POS')
get_results(models_dir+ name +'.h5', name, qid, X, lab, test)

print("\n\n#################### ONE FEAT - W2V & POS - WITH UPDATE - INCREMENTED PATIENCE ####################")
now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
name = 'w2v_pos_up_pat'
# create_2feat_model(que_w2v_emb_up, ans_w2v_emb_up, que_pos_emb_up, ans_pos_emb_up, 'Q_W2V', 'A_W2V', 'Q_POS', 'A_POS', name, 30)
(qid,_ ), X, lab = dataTwo(test, 'Q_W2V', 'A_W2V', 'Q_POS', 'A_POS')
get_results(models_dir + name + '.h5', name, qid, X, lab, test)
########################################### ONE FEAT MODEL - POS ###########################################
########################################### ONE FEAT MODEL - BC ############################################
print("\n\n########################### ONE FEAT - W2V & BROWN CLUSTERS - NO UPDATE ###########################")
now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
name = 'w2v_bc'
# create_2feat_model(que_w2v_emb_no, ans_w2v_emb_no, que_bc_emb_no, ans_bc_emb_no, 'Q_W2V', 'A_W2V', 'Q_BC', 'A_BC', name, 5)
(qid,_ ), X, lab = dataTwo(test, 'Q_W2V', 'A_W2V', 'Q_BC', 'A_BC')
get_results(models_dir+name+'.h5', name, qid, X, lab, test)

print("\n\n################ ONE FEAT - W2V & BROWN CLUSTERS - NO UPDATE - INCREMENTED PATIENCE ###############")
now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
name = 'w2v_bc_pat'
create_2feat_model(que_w2v_emb_no, ans_w2v_emb_no, que_bc_emb_no, ans_bc_emb_no, 'Q_W2V', 'A_W2V', 'Q_BC', 'A_BC', name, 30)
(qid,_ ), X, lab = dataTwo(test, 'Q_W2V', 'A_W2V', 'Q_BC', 'A_BC')
get_results(models_dir+name+'.h5', name, qid, X, lab, test)


print("\n\n########################## ONE FEAT - W2V & BROWN CLUSTERS - WITH UPDATE ##########################")
now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
name = 'w2v_bc_up'
create_2feat_model(que_w2v_emb_up, ans_w2v_emb_up, que_bc_emb_up, ans_bc_emb_up, 'Q_W2V', 'A_W2V', 'Q_BC', 'A_BC', name, 5)
(qid,_ ), X, lab = dataTwo(test, 'Q_W2V', 'A_W2V', 'Q_BC', 'A_BC')
get_results(models_dir+name+'.h5', name, qid, X, lab, test)


print("\n\n############### ONE FEAT - W2V & BROWN CLUSTERS - WITH UPDATE - INCREMENTED PATIENCE ##############")
now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
name = 'w2v_bc_up_pat'
create_2feat_model(que_w2v_emb_up, ans_w2v_emb_up, que_bc_emb_up, ans_bc_emb_up, 'Q_W2V', 'A_W2V', 'Q_BC', 'A_BC', name, 30)
(qid,_ ), X, lab = dataTwo(test, 'Q_W2V', 'A_W2V', 'Q_BC', 'A_BC')
get_results(models_dir+name+'.h5', name, qid, X, lab, test)

########################################### ONE FEAT MODEL - BC ############################################

test.to_csv(path_or_buf=results_dir+'onefeat_test.csv', sep=',', na_rep='', header=1, index=False, index_label=None, mode='w')
test = test.drop(columns=['w2v_pos', 'w2v_pos_pat', 'w2v_pos_up', 'w2v_pos_up_pat', 'w2v_bc', 'w2v_bc_pat', 'w2v_bc_up', 'w2v_bc_up_pat'])

# ####################################### ONE FEAT OVERLAP MODEL - POS #######################################

# print("\n\n########################### ONE FEAT OVERLAP - W2V & POS - WITH UPDATE ############################")
# now = datetime.datetime.now()
# print(now.strftime("%H:%M.%S"))
# name = 'w2v_pos_ov'
# create_2feat_ov_model(que_w2v_ov_emb, ans_w2v_ov_emb, que_pos_emb_up, ans_pos_emb_up, 'Q_W2V', 'A_W2V', 'Q_POS', 'A_POS', 75, name, 5)
# (qid,_ ), X, lab = dataTwoOv(test, 'Q_W2V', 'A_W2V', 'Q_POS', 'A_POS')
# get_results(models_dir+name+'.h5', name, qid, X, lab, test)

# ####################################### ONE FEAT OVERLAP MODEL - POS #######################################
# ####################################### ONE FEAT OVERLAP MODEL - BC ########################################
# print("\n\n###################### ONE FEAT OVERLAP - W2V & BROWN CLUSTERS - WITH UPDATE ######################")
# now = datetime.datetime.now()
# print(now.strftime("%H:%M.%S"))
# name = 'w2v_bc_ov'
# create_2feat_ov_model(que_w2v_ov_emb, ans_w2v_ov_emb, que_bc_emb_up, ans_bc_emb_up, 'Q_W2V', 'A_W2V', 'Q_BC', 'A_BC', 75, name, 5)
# (qid,_ ), X, lab = dataTwoOv(test, 'Q_W2V', 'A_W2V', 'Q_BC', 'A_BC')
# get_results(models_dir+name+'.h5', name, qid, X, lab, test)

# ####################################### ONE FEAT OVERLAP MODEL - BC ########################################

# test.to_csv(path_or_buf=results_dir+'onefeat_ov_test.csv', sep=',', na_rep='', header=1, index=False, index_label=None, mode='w')
# test = test.drop(columns=['w2v_pos_ov', 'w2v_bc_ov'])

# ############################################## COMPLETE MODEL ##############################################
# print("\n\n################################## COMPLETE MODEL - WITH UPDATE ###################################")
# now = datetime.datetime.now()
# print(now.strftime("%H:%M.%S"))
# name = 'w2v_all'
# create_complete_model(que_w2v_ov_emb, ans_w2v_ov_emb, que_pos_emb_up, ans_pos_emb_up, que_bc_emb_up, ans_bc_emb_up, 'Q_W2V', 'A_W2V', 95, name, 5)
# (qid,_ ), X, lab = dataAllOv(test, 'Q_W2V', 'A_W2V')
# get_results(models_dir+name+'.h5', name, qid, X, lab, test)

# ############################################## COMPLETE MODEL ##############################################

# test.to_csv(path_or_buf=results_dir+'complete.csv', sep=',', na_rep='', header=1, index=False, index_label=None, mode='w')