# import win_unicode_console # imported to not have problems with the absence of unicode encoding while working on Windows OS
from importlib import reload
import os
import time

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

import spacy

########################################### READ FILES ###########################################
# we import the datasets in pandas dataframes
train = pd.read_csv('data/WikiQA-train.tsv', sep='\t')
dev = pd.read_csv('data/WikiQA-dev.tsv', sep='\t')
test = pd.read_csv('data/WikiQA-test.tsv', sep='\t')
train

################################## VARIABLES AND EASY FUNCTIONS ##################################
# Size to pad the sentences to
max_len_q = 40
max_len_a = 40

dim = 300
stop = set(stopwords.words('english'))

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

############################################## SPACY #############################################
nlp = spacy.load('en_core_web_lg')

############################### EXECUTION OF IMPORT OF EMBEDDINGS ################################
train=train.head(5)

# lower and tokenize questions and answers
train['Question_tok'] = train['Question'].map(preprocess) 
train['Sentence_tok'] = train['Sentence'].map(preprocess)
# dev['Question_tok'] = dev['Question'].map(preprocess)
# dev['Sentence_tok'] = dev['Sentence'].map(preprocess)
# test['Question_tok'] = test['Question'].map(preprocess)
# test['Sentence_tok'] = test['Sentence'].map(preprocess)

# time.sleep(10)

# create toks list from dataset
toks = (set(chain.from_iterable(train['Question_tok'])) | set(chain.from_iterable(train['Sentence_tok']))) # | \
    # set(chain.from_iterable(dev['Question_tok'])) | set(chain.from_iterable(dev['Sentence_tok']))     | \
    # set(chain.from_iterable(test['Question_tok'])) | set(chain.from_iterable(test['Sentence_tok'])))

# load the word embedding and prepare the dictionary for our dataset
print("STARTING TO load embeddings {}".format('data/aquaint+wiki.txt.gz.ndim=50.bin'))
embW2V = KeyedVectors.load_word2vec_format('data/aquaint+wiki.txt.gz.ndim=50.bin', binary=True)
print("DONE WITH loading embeddings {}".format('data/aquaint+wiki.txt.gz.ndim=50.bin'))


# print("STARTING TO load embeddings {}".format('data/wiki-news-300d-1M.vec'))
# embFT = KeyedVectors.load_word2vec_format('data/wiki-news-300d-1M.vec', binary=False)
# print("DONE WITH loading embeddings {}".format('data/wiki-news-300d-1M.vec'))

########################## IMPORT FT AND W2V EMB AND CREATE DICTIONARY ###########################
########################################## DICTIONARIES ##########################################
dictBC = dict()
dictPOS = dict()
dict_POS = {'PAD':0, 'UNK':1}
dict_BC = {'PAD':0, 'UNK':1}
dict_W2V = {'PAD':0, 'UNK':1}
# dict_FT = {'PAD':0, 'UNK':1}


def create_dict(toks, embed, dictionar):
    i = 2
    for _, tok in enumerate(toks):
        if tok in embed:
            dictionar[tok] = i
            i+=1
    len(dictionar)

# create dictionaries
create_dict(toks, embW2V, dict_W2V)
# create_dict(toks, embFT, dict_FT)
# print(dict_W2V)


################################## CREATE WORD OVERLAP EMBEDDING #################################    
def update_dict(w, e, diction):
    if w not in diction:
        s = set()
        s.add(e)
        diction[w] = s
    else:
        s = diction[w]
        if not e in s:
            print("adding ("+str(e)+")")
            s.add(e)
            diction[w] = s

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
    # print(ov)
    
    x['Q_W2V'] = word2id(x['Question_tok'], dict_W2V)
    x['A_W2V'] = word2id(x['Sentence_tok'], dict_W2V)
    # x['Q_FT'] = word2id(x['Question_tok'], dict_FT)
    # x['A_FT'] = word2id(x['Sentence_tok'], dict_FT)

    que = nlp(x['Question'])
    tags_Q = list()
    bcs_Q = list()
    
    # CREATE POS TAGS AND 
    for w in que:
        tags_Q.append(w.pos)
        bcs_Q.append(w.cluster)
        update_dict(str(w.text), w.pos, dictPOS)
        update_dict(str(w.text), w.cluster, dictBC)

    ans = nlp(x['Sentence'])
    tags_A = list()
    bcs_A = list()
    # print(ans)
    for w in ans:
        tags_A.append(w.pos)
        bcs_A.append(w.cluster)
        update_dict(str(w.text), w.pos, dictPOS)
        update_dict(str(w.text), w.cluster, dictBC)

    x['Q_POS'] = tags_Q
    x['Q_CLUS'] = bcs_Q

    x['A_POS'] = tags_A
    x['A_CLUS'] = bcs_A

    return x



def emb_matrix(dictionar, emb_, dim):
    print(len(dictionar))
    print("dictionary")
    print(dictionar)
    
    embedding_matrix = np.zeros((len(dictionar), dim))
    for word in dictionar:
        if word in emb_:
            # embedding_matrix[dictionar[word]] = [emb_[word]]
            embedding_matrix[dictionar[word]] = [word, emb_[word]]
    print("an embedding matrix")
    print(embedding_matrix)
    return embedding_matrix

# print("creating spacy embeddings")
train = train.apply(embeddes, axis=1)

create_dict(toks, dictPOS, dict_POS)
print("dictionary of pos tags")
print(dict_POS)
# print(max(len(list(sent)) for sent in train['Q_W2V']))
print(max(len(dictPOS[x]) for x in dictPOS))
# print(max(dict_POS, key= lambda x: len(set(dict_POS[x]))))

# emb_matrix(dict_POS, dictPOS, dim)

# print(train[['Question_tok', 'count', 'Q_W2V', 'Q_POS']])

# train.to_csv(path_or_buf='train_embeddings.csv', sep=',', na_rep='', header=1, index=True, index_label=None, mode='w')


# train2.to_csv(path_or_buf='train_embeddings.csv', sep=',', na_rep='', header=1, index=True,
#                 index_label=None, mode='w', doublequote=True, escapechar='\\')

# print(train2[['Question_tok', 'Q_W2V', 'Q_clus', 'count', 'overlap', 'Q_overlap', 'A_overlap']])
# print(type(diction))
# print(diction)

# print(train[['Q_W2V', 'A_W2V', 'Q_overlap', 'S_overlap', 'Q_gen_pos', 'A_gen_pos', 'Q_b_clus', 'A_b_clus']].head())