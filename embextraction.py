# import win_unicode_console # imported to not have problems with the absence of unicode encoding while working on Windows OS
from importlib import reload
import os

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

########################################## TOKENIZATION ##########################################
#sentences are tokenized and lowercased
def lower(sent):
    return sent.lower()
def preprocess(sent):
    return word_tokenize(sent)

################################## VARIABLES AND EASY FUNCTIONS ##################################
# Size to pad the sentences to
max_len_q = 40
max_len_a = 40

dim = 300
stop = set(stopwords.words('english'))

################################## FUNCTIONS TO COMPUTE OVERLAP ###################################
def count_feat(que, ans):
    return len((set(que)&set(ans))-stop)
def overlap(que, ans):
    return (set(que)&set(ans))
def overlap_feats(st, overlapping):
    return [1 if word not in overlapping else 2 for word in st]

############################################## SPACY #############################################

nlp = spacy.load('en_core_web_lg')

# def pos_emb(st):
#     st = nlp(st)
#     return [w.pos for w in st]
# def pos_emb_tag(st):
#     st = nlp(st)
#     return [w.tag for w in st]
# def brown_cluster(st):
#     st = nlp(st)
#     return [w.cluster for w in st]     


############################### EXECUTION OF IMPORT OF EMBEDDINGS ################################
train=train.head(5)

# apply preprocessing
train['Q_low'] = train['Question'].map(lower)
train['A_low'] = train['Sentence'].map(lower)
train['Question_tok'] = train['Q_low'].map(preprocess)
train['Sentence_tok'] = train['A_low'].map(preprocess)
# dev['Question_tok'] = dev['Question'].map(preprocess)
# dev['Sentence_tok'] = dev['Sentence'].map(preprocess)
# test['Question_tok'] = test['Question'].map(preprocess)
# test['Sentence_tok'] = test['Sentence'].map(preprocess)

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
dictionary = {'PAD':0, 'UNK':1}
diction = dict()
dict_W2V = {'PAD':0, 'UNK':1}
dict_FT = {'PAD':0, 'UNK':1}

#map word to IDs
def word2id(sent):
    return list(map(lambda x: dictionary.get(x, 1), sent))

# import the embedded data into a dictionary
# path = path of embeddings,
# binary = true if word2vec, = false if fastText
def map_w2ID(binary):
    # add mapping to IDs
    print("Mapping words to IDs")
    if binary:
        train['Q_W2V'] = train['Question_tok'].map(word2id)
        train['A_W2V'] = train['Sentence_tok'].map(word2id)
        # dev['Q_W2V'] = dev['Question_tok'].map(word2id)
        # dev['A_W2V'] = train['Sentence_tok'].map(word2id)
        # train['Q_W2V'] = train['Question_tok'].map(word2id)
        # train['A_W2V'] = train['Sentence_tok'].map(word2id)
        print(max(len(list(sent)) for sent in train['Q_W2V']))
        print(max(len(list(sent)) for sent in train['A_W2V']))
    else:
        train['Q_FT'] = train['Question_tok'].map(word2id)
        train['A_FT'] = train['Sentence_tok'].map(word2id)
        # dev['Q_FT'] = dev['Question_tok'].map(word2id)
        # dev['A_FT'] = train['Sentence_tok'].map(word2id)
        # train['Q_FT'] = train['Question_tok'].map(word2id)
        # train['A_FT'] = train['Sentence_tok'].map(word2id)
        print(max(len(list(sent)) for sent in train['Q_FT']))
        print(max(len(list(sent)) for sent in train['A_FT']))




def create_dict(toks, embed, dictionar):
    i = 2
    for _, tok in enumerate(toks):
        if tok in embed:
            dictionar[tok] = i
            i+=1
    len(dictionar)

# create dictionaries
create_dict(toks, embW2V, dict_W2V)

print(dict_W2V)
# create_dict(toks, embFT, dict_FT)

# create word IDs

## dictionary = dict_\W2V
## map_w2ID(binary=True)

# dictionary = dict_FT
# map_w2ID(binary=False)

################################## CREATE WORD OVERLAP EMBEDDING #################################
def update_dict(w):
    if str(w.text) not in diction:
        # print("A new tuple")
        tt = set()
        clusters = set()
        tt.add(w.pos)
        clusters.add(w.cluster)
        diction[str(w.text)] = (tt, clusters)
        # print("Q - new dictionary:")
    else:
        tup = diction[str(w.text)]
        # print("A existing tup for "+ w.text +":")
        print(tup)
        if not w.pos in tup[0]:
            print("not in tags")
            tup[0].add(w.pos)
        if not w.cluster in tup[1]:
            print("not in brown clusters")
            tup[1].add(w.cluster)
    
def embeddes(x):
    que_tok = x['Question_tok']
    ans_tok = x['Sentence_tok']
    
    x['count'] = count_feat(que_tok, ans_tok)
    ov = overlap(que_tok, ans_tok)
    x['overlap'] = ov
    x['Q_overlap'] = overlap_feats(que_tok, ov)
    x['A_overlap'] = overlap_feats(ans_tok, ov)
    # print(ov)

    que = nlp(x['Question'])
    tags_Q = list()
    bcs_Q = list()

    print(que)
    print(len(que))
    for w in que:
        tags_Q.append(w.pos)
        bcs_Q.append(w.cluster)
        update_dict(w)

    ans = nlp(x['Sentence'])
    tags_A = list()
    bcs_A = list()
    print(ans)
    for w in ans:
        tags_A.append(w.pos)
        bcs_A.append(w.cluster)
        update_dict(w)

    x['Q_pos'] = tags_Q
    x['Q_clus'] = bcs_Q

    x['A_pos'] = tags_A
    x['A_clus'] = bcs_A

    return x

# print("creating spacy embeddings")
# train2 = train.apply(embeddes, axis=1)

# print(train2[['Question_tok', 'Q_W2V', 'Q_clus', 'count', 'overlap', 'Q_overlap', 'A_overlap']])
# print(type(diction))
# print(diction)

# print(train[['Q_W2V', 'A_W2V', 'Q_overlap', 'S_overlap', 'Q_gen_pos', 'A_gen_pos', 'Q_b_clus', 'A_b_clus']].head())


# print("Creating overlaps")
# train['count'] = pd.Series(count_feat(que, ans) for que, ans in zip(train['Question_tok'], train['Sentence_tok']))
# test['count'] = pd.Series(count_feat(que, ans) for que, ans in zip(test['Question_tok'], test['Sentence_tok']))
# dev['count'] = pd.Series(count_feat(que, ans) for que, ans in zip(dev['Question_tok'], dev['Sentence_tok']))

############################## WORD OVERLAP ##############################
# train['overlap'] = pd.Series(overlap(que, ans) for que, ans in zip(train['Question_tok'], train['Sentence_tok']))
# test['overlap'] = pd.Series(overlap(que, ans) for que, ans in zip(test['Question_tok'], test['Sentence_tok']))
# dev['overlap'] = pd.Series(overlap(que, ans) for que, ans in zip(dev['Question_tok'], dev['Sentence_tok']))

## Now we use word overlaps as binary feature at the embedding level
# train['Q_overlap'] = pd.Series(overlap_feats(que, ov) for que, ov in zip(train['Question_tok'], train['overlap']))
# train['A_overlap'] = pd.Series(overlap_feats(que, ov) for que, ov in zip(train['Sentence_tok'], train['overlap']))
# dev['Q_overlap'] = pd.Series(overlap_feats(que, ov) for que, ov in zip(dev['Question_tok'], dev['overlap']))
# dev['A_overlap'] = pd.Series(overlap_feats(que, ov) for que, ov in zip(dev['Sentence_tok'], dev['overlap']))
# test['Q_overlap'] = pd.Series(overlap_feats(que, ov) for que, ov in zip(test['Question_tok'], test['overlap']))
# test['A_overlap'] = pd.Series(overlap_feats(que, ov) for que, ov in zip(test['Sentence_tok'], test['overlap']))