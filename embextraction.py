from importlib import reload
import os
import time
import csv

# lybraries
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

########################################### READ FILES ###########################################
# we import the datasets in pandas dataframes
train = pd.read_csv('data/WikiQA-train.tsv', sep='\t')
dev = pd.read_csv('data/WikiQA-dev.tsv', sep='\t')
test = pd.read_csv('data/WikiQA-test.tsv', sep='\t')

################################## VARIABLES AND EASY FUNCTIONS ##################################
# Size to pad the sentences to
max_len_q = 25
max_len_a = 100

dim = 300
stop = set(stopwords.words('english'))

nlp = spacy.load('en_core_web_lg')

########################################## TOKENIZATION ##########################################
#sentences are tokenized and lowercased
def preprocess(sent):
    return word_tokenize(sent.lower())

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
train=train.head(5)
dev=dev.head(5)
test=test.head(5)

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
embBC = dict()
embPOS = dict()
dict_POS = {'PAD':0, 'UNK':1}
dict_BC = {'PAD':0, 'UNK':1}
dict_W2V = {'PAD':0, 'UNK':1}
dict_FT = {'PAD':0, 'UNK':1}


def create_dict(toks, embed, dictionar):
    i = 2
    for _, tok in enumerate(toks):
        if tok in embed:
            dictionar[tok] = i
            i+=1

######################### CREATE WORD EMBEDDINGS AND RELATED DICTIONARIES ########################    
def update_dict(w, e, diction):
    if w not in diction:
        s = list()
        s.append(e)
        diction[w] = s
    else:
        s = diction[w]
        if not e in s:
            s.append(e)
            diction[w] = s

# def update_emb(w, e, diction):
#     s = np.zeros((20,), dtype=int)
#     if w in diction:
#         s=diction[w]
#     if e:
#         if e == 102:
#             s[19]=1
#         else:
#             s[(e%83)+1]=1
        
#     else:
#         s[0]=1
#     diction[w]=s


def embeddes(x):
    que_tok = x['Question_tok']
    ans_tok = x['Sentence_tok']

    # CREATE OVERLAP DATA
    x['count'] = count_feat(que_tok, ans_tok)
    overl = overlap(que_tok, ans_tok)
    x['overlap'] = overl
    Q_overl = overlap_feats(que_tok, overl)
    A_overl = overlap_feats(ans_tok, overl)
    x['Q_OV'] = Q_overl
    x['A_OV'] = A_overl
    
    x['Q_W2V'] = word2id(x['Question_tok'], dict_W2V)
    x['A_W2V'] = word2id(x['Sentence_tok'], dict_W2V)
    x['Q_FT'] = word2id(x['Question_tok'], dict_FT)
    x['A_FT'] = word2id(x['Sentence_tok'], dict_FT)

    # CREATE POS TAGS AND BROWN CLUSTERS
    que = nlp(x['Question'])
    tags_Q = list()
    bcs_Q = list()
    
    for w in que:
        tags_Q.append(w.pos)
        bcs_Q.append(w.cluster)
        update_dict(str(w.text), w.pos, embPOS)
        update_dict(str(w.text), w.cluster, embBC)

    ans = nlp(x['Sentence'])
    tags_A = list()
    bcs_A = list()
    for w in ans:
        tags_A.append(w.pos)
        bcs_A.append(w.cluster)
        update_dict(str(w.text), w.pos, embPOS)
        update_dict(str(w.text), w.cluster, embBC)

    x['Q_POS'] = tags_Q
    x['Q_CLUS'] = bcs_Q

    x['A_POS'] = tags_A
    x['A_CLUS'] = bcs_A
    return x

################################# CREATE EMBEDDING ARRAYS FOR BC #################################
# def create_array(values, emb):
#     dim=len(values)
#     for w in emb:
#         s = np.zeros((dim,), dtype=int)
#         l = emb[w]
#         for v in l:
#             s[values.index(v)] = 1
#         emb[w] = s

##################################### CREATE EMBEDDING MATRIX ####################################
def emb_matrix(dictionar, emb_, dim):        
    embedding_matrix = np.zeros((len(dictionar), dim))
    for word in dictionar:
        if word in emb_:
            embedding_matrix[dictionar[word]] = emb_[word]
        else:
            np.random.uniform(-1.0, 1.0, (1, dim))
    return embedding_matrix

def emb_matrix_unk(dictionar, emb_, dim):        
    embedding_matrix = np.random.uniform(-1.0, 1.0, (len(dictionar), dim))
    embedding_matrix[0] = np.zeros((1, dim))
    return embedding_matrix

##########################################################################################################################
################################### CREATE EMBEDDING, MODIFY TRAIN, AND STORE TO FILES ###################################
print("EXTRACTING embeddings")
train = train.apply(embeddes, axis=1)
dev = dev.apply(embeddes, axis=1)
test = test.apply(embeddes, axis=1)

print("APPLYING padding")
train['Q_W2V'] = train['Q_W2V'].apply(lambda s: pad_sequences([s], max_len_q)[0])
train['A_W2V'] = train['A_W2V'].apply(lambda s: pad_sequences([s], max_len_a)[0])
train['Q_FT'] = train['Q_FT'].apply(lambda s: pad_sequences([s], max_len_q)[0])
train['A_FT'] = train['A_FT'].apply(lambda s: pad_sequences([s], max_len_a)[0])
dev['Q_W2V'] = dev['Q_W2V'].apply(lambda s: pad_sequences([s], max_len_q)[0])
dev['A_W2V'] = dev['A_W2V'].apply(lambda s: pad_sequences([s], max_len_a)[0])
dev['Q_FT'] = dev['Q_FT'].apply(lambda s: pad_sequences([s], max_len_q)[0])
dev['A_FT'] = dev['A_FT'].apply(lambda s: pad_sequences([s], max_len_a)[0])
test['Q_W2V'] = test['Q_W2V'].apply(lambda s: pad_sequences([s], max_len_q)[0])
test['A_W2V'] = test['A_W2V'].apply(lambda s: pad_sequences([s], max_len_a)[0])
test['Q_FT'] = test['Q_FT'].apply(lambda s: pad_sequences([s], max_len_q)[0])
test['A_FT'] = test['A_FT'].apply(lambda s: pad_sequences([s], max_len_a)[0])


print("columns of train:")
print(list(train.columns.values))

print("WRITING to file new dataframes")
train.to_csv(path_or_buf='train_embeddings.csv', sep=',', na_rep='', header=1, index=True, index_label=None, mode='w')
test.to_csv(path_or_buf='test_embeddings.csv', sep=',', na_rep='', header=1, index=True, index_label=None, mode='w')
dev.to_csv(path_or_buf='dev_embeddings.csv', sep=',', na_rep='', header=1, index=True, index_label=None, mode='w')

##########################################################################################################################
##################################### CREATE EMBEDDING DICTIONARIES ######################################
print("CREATING dictionaries")
create_dict(toks, embW2V, dict_W2V)
create_dict(toks, embFT, dict_FT)

create_dict(toks, embPOS, dict_POS)
create_dict(toks, embBC, dict_BC)
##########################################################################################################
####################################### CREATE EMBEDDING MATRICES ########################################
print("CREATING embedding matrices")
matrix_W2V = emb_matrix(dict_W2V, embW2V, 50)
matrix_FT = emb_matrix(dict_FT, embFT, 300)

matrix_POS = emb_matrix_unk(dict_POS, embPOS, 20)
matrix_BC = emb_matrix_unk(dict_BC, embBC, 20)
##########################################################################################################
####################################### WRITE DICTIONARIES TO FILE #######################################
print("WRITING to file dictionaries and embedding matrices")
w = csv.writer(open("data/results/embPOS.csv", "w"))
for key, val in embPOS.items():
    w.writerow([key, val])

w = csv.writer(open("data/results/embBC.csv", "w"))
for key, val in embBC.items():
    w.writerow([key, val])

w = csv.writer(open("data/results/dictW2V.csv", "w"))
for key, val in dict_W2V.items():
    w.writerow([key, val])

w = csv.writer(open("data/results/dictFT.csv", "w"))
for key, val in dict_FT.items():
    w.writerow([key, val])

w = csv.writer(open("data/results/dictPOS.csv", "w"))
for key, val in dict_POS.items():
    w.writerow([key, val])

w = csv.writer(open("data/results/dictBC.csv", "w"))
for key, val in dict_BC.items():
    w.writerow([key, val])
##########################################################################################################
######################################### WRITE MATRICES TO FILE #########################################
np.savez_compressed("data/results/matrices", w2v=matrix_W2V, ft=matrix_FT, pos=matrix_POS, bc=matrix_BC)
# np.savez_compressed("data/results/matrices", w2v=matrix_W2V, pos=matrix_POS, bc=matrix_BC)
##########################################################################################################
