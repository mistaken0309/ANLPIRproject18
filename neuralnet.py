from importlib import reload
import os

import time
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
                            Reshape, Merge, Flatten,
                            GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, Dense)

max_len_q = 40 #25
max_len_a = 40 # 100

models_dir = ('./models')        
if not os.path.isdir(models_dir):
    os.makedirs(models_dir)
    print("Home directory %s was created." %models_dir)

# remove previously created models
# if os.path.exists('./models/qa_W2V_basic.h5'): os.remove('./models/qa_W2V_basic.h5')
# if os.path.exists('./models/qa_W2V_overlap.h5'): os.remove('./models/qa_W2V_overlap.h5')
# if os.path.exists('./models/qa_ft_basic.h5'): os.remove('./models/qa_ft_basic.h5')
# if os.path.exists('./models/qa_ft_overlap.h5'): os.remove('./models/qa_ft_overlap.h5')

def intarray(a):
    a = a.replace('\n', '')
    return list(map(int,a.split(',')))

# LOAD DATASETS AS PANDAS DATAFRAMES
train = pd.read_csv('train_embeddings.csv', sep=',')
train['Q_W2V'] = train['Q_W2V'].map(intarray)
train['A_W2V'] = train['A_W2V'].map(intarray)
# train['Q_FT'] = train['Q_FT'].map(intarray)
# train['A_FT'] = train['A_FT'].map(intarray)
train['Q_OV'] = train['Q_OV'].map(intarray)
train['A_OV'] = train['A_OV'].map(intarray)
train['Q_POS'] = train['Q_POS'].map(intarray)
train['A_POS'] = train['A_POS'].map(intarray)
train['Q_BC'] = train['Q_BC'].map(intarray)
train['A_BC'] = train['A_BC'].map(intarray)

# print("AFTER")
# print(train['Q_W2V'])

dev = pd.read_csv('dev_embeddings.csv', sep=',')
dev['Q_W2V'] = dev['Q_W2V'].map(intarray)
dev['A_W2V'] = dev['A_W2V'].map(intarray)
# dev['Q_FT'] = dev['Q_FT'].map(intarray)
# dev['A_FT'] = dev['A_FT'].map(intarray)
dev['Q_OV'] = dev['Q_OV'].map(intarray)
dev['A_OV'] = dev['A_OV'].map(intarray)
dev['Q_POS'] = dev['Q_POS'].map(intarray)
dev['A_POS'] = dev['A_POS'].map(intarray)
dev['Q_BC'] = dev['Q_BC'].map(intarray)
dev['A_BC'] = dev['A_BC'].map(intarray)


test = pd.read_csv('test_embeddings.csv', sep=',')
test['Q_W2V'] = test['Q_W2V'].map(intarray)
test['A_W2V'] = test['A_W2V'].map(intarray)
# test['Q_FT'] = test['Q_FT'].map(intarray)
# test['A_FT'] = test['A_FT'].map(intarray)
test['Q_OV'] = test['Q_OV'].map(intarray)
test['A_OV'] = test['A_OV'].map(intarray)
test['Q_POS'] = test['Q_POS'].map(intarray)
test['A_POS'] = test['A_POS'].map(intarray)
test['Q_BC'] = test['Q_BC'].map(intarray)
test['A_BC'] = test['A_BC'].map(intarray)


# LOAD EMBEDDING MATRICES
em = np.load("data/results/matrices.npz")
em_W2V = em['w2v']
# em_FT = em['ft']
em_POS = em['pos']
em_BC = em['bc']

with open('data/results/dictW2V.json') as f:
    dict_w2v = json.load(f)
# with open('data/results/dictFT.json') as f:
#     dict_ft = json.load(f)
with open('data/results/dictPOS.json') as f:
    dict_pos = json.load(f)
with open('data/results/dictBC.json') as f:
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
feat_q = Input(shape=(max_len_q,))
ans = Input(shape=(max_len_a,))
feat_a = Input(shape=(max_len_a,))

que_ov = Input(shape=(max_len_q,))
ans_ov = Input(shape=(max_len_a,))
cnt = Input(shape=(1,))

que_pos = Input(shape=(max_len_q,))
ans_pos = Input(shape=(max_len_a,))
que_bc = Input(shape=(max_len_q,))
ans_bc = Input(shape=(max_len_a,))


que_ov_emb = Embedding(3, 5,input_length=max_len_q)(que_ov)
que_w2v_emb_c = Embedding(len(dict_w2v), 50, input_length=max_len_q, weights=[em_W2V], trainable=False)(que)
# que_ft_emb_c = Embedding(len(dict_ft), 300, input_length=max_len_q, weights=[em_FT], trainable=False)(que)
que_pos_emb_c = Embedding(len(dict_pos), 20, input_length=max_len_q, weights=[em_POS], trainable=True)(feat_q)
que_bc_emb_c = Embedding(len(dict_bc), 20, input_length=max_len_q, weights=[em_BC], trainable=True)(feat_q)

q_ov_w2v = concatenate([que_ov_emb, que_w2v_emb_c])
# q_ov_ft = concatenate([que_ov_emb, que_ft_emb_c])
q_ov_pos = concatenate([que_ov_emb, que_pos_emb_c])
q_ov_bc = concatenate([que_ov_emb, que_bc_emb_c])

q_pos_bc = concatenate([que_w2v_emb_c, que_pos_emb_c])

# embeddings for overlap model
ans_ov_emb = Embedding(3, 5,input_length=max_len_a)(ans_ov)
ans_w2v_emb_c = Embedding(len(dict_w2v), 50, input_length=max_len_a, weights=[em_W2V], trainable=False)(ans)
# ans_ft_emb_c = Embedding(len(dict_ft), 300, input_length=max_len_a, weights=[em_FT], trainable=False)(ans)
ans_pos_emb_c = Embedding(len(dict_pos), 20, input_length=max_len_a, weights=[em_POS], trainable=True)(feat_a)
ans_bc_emb_c = Embedding(len(dict_bc), 20, input_length=max_len_a, weights=[em_BC], trainable=True)(feat_a)

a_ov_w2v = concatenate([ans_ov_emb, ans_w2v_emb_c])
# a_ov_ft = concatenate([ans_ov_emb, ans_ft_emb_ov])
a_ov_pos = concatenate([ans_ov_emb, ans_pos_emb_c])
a_ov_bc = concatenate([ans_ov_emb, ans_bc_emb_c])
    
a_pos_bc = concatenate([ans_pos_emb_c, ans_bc_emb_c])

# embeddings for w2v in the basic model
que_w2v_emb = Embedding(len(dict_w2v), 50, input_length=max_len_q, weights=[em_W2V], trainable=False)
ans_w2v_emb = Embedding(len(dict_w2v), 50, input_length=max_len_a, weights=[em_W2V], trainable=False)
# que_ft_emb = Embedding(len(dict_ft), 300, input_length=max_len_q, weights=[em_FT], trainable=False)
# ans_ft_emb = Embedding(len(dict_ft), 300, input_length=max_len_a, weights=[em_FT], trainable=False)
que_pos_emb = Embedding(len(dict_pos), 20, input_length=max_len_q, weights=[em_POS], trainable=True)
ans_pos_emb = Embedding(len(dict_pos), 20, input_length=max_len_a, weights=[em_POS], trainable=True)
que_bc_emb = Embedding(len(dict_bc), 20, input_length=max_len_q, weights=[em_BC], trainable=True)
ans_bc_emb = Embedding(len(dict_bc), 20, input_length=max_len_a, weights=[em_BC], trainable=True)


np.random.seed(42)


def create_classify(join, cl_dim):
    classify = Sequential()
    classify.add(Dense(100, activation='tanh', input_dim=cl_dim))
    classify.add(Dense(1, activation='sigmoid'))
    print("CLASSIFY")
    classify.summary()
    return classify(join)

###############################################################################################################
############################################## ONE FEATURE MODEL ##############################################
def create_1feat_model(emb_q, emb_a, col_q, col_a, newname):
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
            callbacks=[EpochEval(data(dev, col_q, col_a), map_score_filtered, patience=5)])

    nameb ='./models' + newname + '_basic.h5'
    os.rename('qa.h5', nameb)
###############################################################################################################
######################################### ONE FEAT WITH OVERLAP MODEL #########################################
def create_1feat_ov_model(emb_q, emb_a, col_q, col_a, size, newname):
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
    print("MODEL FIT 1 feat with ov")
    model_ov.fit([np.vstack(train[col_q]), np.vstack(train[col_a]), np.vstack(train['Q_OV']), np.vstack(train['A_OV']), train['count'].values],
            np.vstack(train['Label'].tolist()), batch_size=100, epochs=100000, shuffle=True, verbose=2,
            callbacks=[EpochEval(dataOver(dev, col_q, col_a), map_score_filtered, patience=5)])

    name ='./models' + newname + 'overlap.h5'
    os.rename('qa.h5', name)
###############################################################################################################
############################################## TWO FEATURE MODEL ##############################################
def create_2feat_model(emb_q, emb_a, emb_fq, emb_fa, col_q_1, col_a_1, col_q_2, col_a_2, newname):
    
    que_in = Sequential()
    que_in.add(emb_q)
    que_in.add(Convolution1D(100, 5, activation='tanh'))
    que_in.add(GlobalAveragePooling1D())
    # que_in.add(Flatten())

    ans_in = Sequential()
    ans_in.add(emb_a)
    ans_in.add(Convolution1D(100, 5, activation='tanh'))
    ans_in.add(GlobalAveragePooling1D())
    # ans_in.add(Flatten())

    featQ_in = Sequential()
    featQ_in.add(emb_fq)
    # featQ_in.add(Convolution1D(100, 5, activation='tanh'))
    # featQ_in.add(GlobalAveragePooling1D())
    featQ_in.add(Flatten())

    featA_in = Sequential()
    featA_in.add(emb_fa)
    # featA_in.add(Convolution1D(100, 5, activation='tanh'))
    # featA_in.add(GlobalAveragePooling1D())
    featA_in.add(Flatten())

    q_emb = que_in(que)
    a_emb = ans_in(ans)
    fq_emb = featQ_in(feat_q)
    fa_emb = featA_in(feat_a)

    qa_matrix = concatenate([q_emb, a_emb, fq_emb, fa_emb]) 
    # qa_matrix = flatten
    # out = create_classify(qa_matrix, 5600)
    out = create_classify(qa_matrix, 1800)
    # out = create_classify(qa_matrix, 400)


    model = Model(inputs=[que, ans, feat_q, feat_a], outputs=[out])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

    print("MODEL FIT")
    model.fit([np.vstack(train[col_q_1]), np.vstack(train[col_a_1]), np.vstack(train[col_q_2]), np.vstack(train[col_a_2])],
            np.vstack(train['Label'].tolist()), batch_size=100, epochs=100000, shuffle=True, verbose=2,
            callbacks=[EpochEval(dataTwo(dev, col_q_1, col_a_1, col_q_2, col_a_2), map_score_filtered, patience=5)])

    nameb ='./models' + newname + '.h5'
    os.rename('qa.h5', nameb)
###############################################################################################################





create_1feat_model(que_w2v_emb, ans_w2v_emb, 'Q_W2V', 'A_W2V', "/qa_W2V")
# create_1feat_model(que_ft_emb, ans_ft_emb, 'Q_FT', 'A_FT', "/qa_FT")
# create_1feat_model(que_pos_emb, ans_pos_emb, 'Q_POS', 'A_POS', "/qa_POS")
# create_1feat_model(que_bc_emb, ans_bc_emb, 'Q_BC', 'A_BC', "/qa_BC")


create_2feat_model(que_w2v_emb, ans_w2v_emb, que_pos_emb, ans_pos_emb, 'Q_W2V', 'A_W2V', 'Q_POS', 'A_POS', "qa_w2v_pos")

# print("STARTING with overlap")
# create_1feat_ov_model(q_ov_w2v, a_ov_w2v, 'Q_W2V', 'A_W2V', 55, "/qa_w2v_ov")
# create_1feat_ov_model(q_ov_ft, a_ov_ft, 'Q_FT', 'A_FT', 55, "/qa_ft_ov")
# create_1feat_ov_model(q_ov_pos, a_ov_pos, 'Q_POS', 'A_POS', 25, "/qa_pos_ov")
# create_1feat_ov_model(q_ov_bc, a_ov_bc, 'Q_BC', 'A_BC', 25, "/qa_bc_ov")