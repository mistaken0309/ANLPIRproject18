from importlib import reload
import os
import time
import datetime
import csv
import json


# libraries
import pandas as pd
import numpy as np

# natural language toolkit imports
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# metrics
from metrics import map_score_filtered, map_score
from sklearn.metrics import roc_auc_score

from itertools import chain

# embedding models
from gensim.models import KeyedVectors

from keras.preprocessing.sequence import pad_sequences

import spacy


data_dir = ('project/data/')
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
    print("Home directory %s was created." %data_dir)


########################################### READ FILES ###########################################
# we import the datasets in pandas dataframes
train = pd.read_csv('data/WikiQA-train.tsv', sep='\t')
dev = pd.read_csv('data/WikiQA-dev.tsv', sep='\t')
test = pd.read_csv('data/WikiQA-test.tsv', sep='\t')

################################## VARIABLES AND EASY FUNCTIONS ##################################
# Size to pad the sentences to
max_len_q = 40 #25
max_len_a = 40 #100

stop = set(stopwords.words('english'))

nlp = spacy.load('en_core_web_lg')

########################################## TOKENIZATION ##########################################
#sentences are tokenized and lowercased
def preprocess(sent):
    return word_tokenize(sent.lower())

def a2s(a):
    k = ','.join(map(str, a))
    return k

# map word to IDs
def word2id(sent, d):
    return list(map(lambda x: d.get(x, 1), sent))

################################## FUNCTIONS TO COMPUTE OVERLAP ###################################
def count_feat(que, ans):
    return len((set(que)&set(ans))-stop)
def overlap(que, ans):
    return (set(que)&set(ans))
def overlap_feats(st, overlapping):
    return [1 if word not in overlapping else 2 for word in st]

################################# FIRST PREPROCESSING OF DATASET #################################
# train=train.head(int(len(train.index)/100))
# dev=dev.head(int(len(dev.index)/100))
# test=test.head(int(len(test.index)/100))
# print(len(train.index))
# print(len(dev.index))
# print(len(test.index))

# lower and tokenize questions and answers
train['Question_tok'] = train['Question'].map(preprocess) 
train['Sentence_tok'] = train['Sentence'].map(preprocess)
dev['Question_tok'] = dev['Question'].map(preprocess)
dev['Sentence_tok'] = dev['Sentence'].map(preprocess)
test['Question_tok'] = test['Question'].map(preprocess)
test['Sentence_tok'] = test['Sentence'].map(preprocess)

# print(max(len(list(sent)) for sent in train['Question_tok']))
# print(max(len(list(sent)) for sent in train['Sentence_tok']))
# print(max(len(list(sent)) for sent in dev['Question_tok']))
# print(max(len(list(sent)) for sent in dev['Sentence_tok']))
# print(max(len(list(sent)) for sent in test['Question_tok']))
# print(max(len(list(sent)) for sent in test['Sentence_tok']))

# print("going to sleep")
# time.sleep(1000)


# create toks list from dataset
toks = (set(chain.from_iterable(train['Question_tok'])) | set(chain.from_iterable(train['Sentence_tok']))   | \
    set(chain.from_iterable(dev['Question_tok'])) | set(chain.from_iterable(dev['Sentence_tok']))           | \
    set(chain.from_iterable(test['Question_tok'])) | set(chain.from_iterable(test['Sentence_tok'])))

##################################### IMPORT FT AND W2V EMB ######################################
# load the word embedding and prepare the dictionary for our dataset
print("STARTING TO load embeddings {}".format('data/aquaint+wiki.txt.gz.ndim=50.bin'))
embW2V = KeyedVectors.load_word2vec_format('data/aquaint+wiki.txt.gz.ndim=50.bin', binary=True)
print("DONE WITH loading embeddings {}".format('data/aquaint+wiki.txt.gz.ndim=50.bin'))


print("STARTING TO load embeddings {}".format('data/wiki-news-300d-1M.vec'))
embFT = KeyedVectors.load_word2vec_format('data/wiki-news-300d-1M.vec', binary=False)
print("DONE WITH loading embeddings {}".format('data/wiki-news-300d-1M.vec'))

##################################### IMPORT FT AND W2V EMB ######################################
########################################## DICTIONARIES ##########################################
dict_W2V = {'PAD':0, 'UNK':1}
dict_FT = {'PAD':0, 'UNK':1}
dict_POS = {'PAD':0, 'UNK':1}
dict_BC = {'PAD':0, 'UNK':1}

i = 1

######################### CREATE WORD EMBEDDINGS AND RELATED DICTIONARIES ########################    
def update_dict(e, diction):
    j = len(diction)
    if e not in diction:
        diction[e] = j
    return diction[e]

def embeddes(x):
    que_tok = x['Question_tok']
    ans_tok = x['Sentence_tok']

    # CREATE OVERLAP DATA
    x['count'] = count_feat(que_tok, ans_tok)
    overl = overlap(que_tok, ans_tok)
    x['overlap'] = overl
    Q_overl = overlap_feats(que_tok, overl)
    A_overl = overlap_feats(ans_tok, overl)
    x['Q_OV'] = ','.join(map(str, pad_sequences([Q_overl], max_len_q)[0]))
    x['A_OV'] = ','.join(map(str, pad_sequences([A_overl], max_len_a)[0]))
    
    x['Q_W2V'] = ','.join(map(str, pad_sequences([word2id(x['Question_tok'], dict_W2V)], max_len_q)[0]))
    x['A_W2V'] = ','.join(map(str, pad_sequences([word2id(x['Sentence_tok'], dict_W2V)], max_len_a)[0]))
    x['Q_FT'] = ','.join(map(str, pad_sequences([word2id(x['Question_tok'], dict_FT)], max_len_q)[0]))
    x['A_FT'] = ','.join(map(str, pad_sequences([word2id(x['Sentence_tok'], dict_FT)], max_len_a)[0]))

    # CREATE POS TAGS AND BROWN CLUSTERS
    que = nlp(x['Question'])
    tags_Q = list()
    bcs_Q = list()
    
    for w in que:
        p = update_dict(w.tag_, dict_POS)
        bc = update_dict(w.cluster, dict_BC)
        tags_Q.append(p)
        bcs_Q.append(bc)
        

    ans = nlp(x['Sentence'])
    tags_A = list()
    bcs_A = list()
    for w in ans:
        p = update_dict(w.tag_, dict_POS)
        bc = update_dict(w.cluster, dict_BC)
        tags_A.append(p)
        bcs_A.append(bc)

    
    x['Q_POS'] = ','.join(map(str, pad_sequences([tags_Q], max_len_q)[0]))
    x['A_POS'] = ','.join(map(str, pad_sequences([tags_A], max_len_a)[0]))
    
    x['Q_BC'] = ','.join(map(str, pad_sequences([bcs_Q], max_len_q)[0]))
    x['A_BC'] = ','.join(map(str, pad_sequences([bcs_A], max_len_a)[0]))
    
    global i 
    if i %1000 == 0:
        print(i)
    i = i +1
    return x


######################################## CREATE DICTIONARY #######################################
def create_dict(toks, embed, dictionar):
    i = 2
    for _, tok in enumerate(toks):
        if tok in embed:
            dictionar[tok] = i
            i+=1

##################################### CREATE EMBEDDING MATRIX ####################################
def emb_matrix(dictionar, emb_, dim):      
    embedding_matrix = np.zeros((len(dictionar), dim))
    for word in dictionar:
        if word in emb_:
            embedding_matrix[dictionar[word]] = emb_[word]
    return embedding_matrix

def emb_matrix_unk(dictionar, dim):        
    embedding_matrix = np.random.uniform(-1.0, 1.0, (len(dictionar), dim))
    embedding_matrix[0] = np.zeros((1, dim))
    return embedding_matrix

##########################################################################################################################
################################################## CREATE DICTIONARIES ###################################################
create_dict(toks, embW2V, dict_W2V)
create_dict(toks, embFT, dict_FT)

##########################################################################################################################
################################### CREATE EMBEDDING, MODIFY TRAIN, AND STORE TO FILES ###################################
print("EXTRACTING embeddings")

i = 1
print("TRAIN")
print(len(train.index))
now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
train = train.apply(embeddes, axis=1)

now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
print("WRITING TRAIN TO FILE")
train.to_csv(path_or_buf=data_dir+'train_embeddings.csv', sep=',', na_rep='', header=1, index=True, index_label=None, mode='w')

i = 1
print("DEV")
print(len(dev.index))
now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
dev = dev.apply(embeddes, axis=1)

now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
print("WRITING DEV TO FILE")
dev.to_csv(path_or_buf=data_dir+'dev_embeddings.csv', sep=',', na_rep='', header=1, index=True, index_label=None, mode='w')

i = 1
print("TEST")
print(len(test.index))
now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
test = test.apply(embeddes, axis=1)

now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))
print("WRITING TEST TO FILE")
test.to_csv(path_or_buf=data_dir+'test_embeddings.csv', sep=',', na_rep='', header=1, index=True, index_label=None, mode='w')
now = datetime.datetime.now()
print(now.strftime("%H:%M.%S"))


# print("columns of train:")
# print(list(train.columns.values))

##########################################################################################################
####################################### CREATE EMBEDDING MATRICES ########################################
print("CREATING embedding matrices")
matrix_W2V = emb_matrix(dict_W2V, embW2V, 50)
matrix_FT = emb_matrix(dict_FT, embFT, 300)

matrix_POS = emb_matrix_unk(dict_POS, 20)
matrix_BC = emb_matrix_unk(dict_BC, 20)
##########################################################################################################
####################################### WRITE DICTIONARIES TO FILE #######################################
print("WRITING to file dictionaries and embedding matrices")
with open(data_dir+'dictW2V.json', 'w') as file:
     file.write(json.dumps(dict_W2V))
with open(data_dir+'dictFT.json', 'w') as file:
     file.write(json.dumps(dict_FT))
with open(data_dir+'dictPOS.json', 'w') as file:
     file.write(json.dumps(dict_POS))
with open(data_dir+'dictBC.json', 'w') as file:
     file.write(json.dumps(dict_BC))

##########################################################################################################
######################################### WRITE MATRICES TO FILE #########################################
# np.savez_compressed("project/data/matrices", w2v=matrix_W2V)
np.savez_compressed(data_dir+'matrices', w2v=matrix_W2V, ft=matrix_FT, pos=matrix_POS, bc=matrix_BC)
# np.savez_compressed("project/data/matrices", w2v=matrix_W2V, pos=matrix_POS, bc=matrix_BC)
##########################################################################################################

