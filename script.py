# import win_unicode_console # imported to not have problems with the absence of unicode encoding while working on Windows OS
from importlib import reload
import os

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
                            GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, Dense)


# added since working on Windows OS
# win_unicode_console.enable()
############## DATA IMPORT AND ANALYSIS ##############

models_dir = ('./project/models')        
if not os.path.isdir(models_dir):
    os.makedirs(models_dir)
    print("Home directory %s was created." %models_dir)

if os.path.exists('./project/models/qa_W2V_basic.h5'): os.remove('./project/models/qa_W2V_basic.h5')
if os.path.exists('./project/models/qa_W2V_overlap.h5'): os.remove('./project/models/qa_W2V_overlap.h5')
if os.path.exists('./project/models/qa_ft_basic.h5'): os.remove('./project/models/qa_ft_basic.h5')
if os.path.exists('./project/models/qa_ft_overlap.h5'): os.remove('./project/models/qa_ft_overlap.h5')

# we import the datasets in pandas dataframes
train = pd.read_csv('data/WikiQA-train.tsv', sep='\t')
dev = pd.read_csv('data/WikiQA-dev.tsv', sep='\t')
test = pd.read_csv('data/WikiQA-test.tsv', sep='\t')
train

#sentences are tokenized and lowercased
def preprocess(sent):
    # return word_tokenize(sent.decode('utf-8').encode("ascii","ignore").lower())
    return word_tokenize(sent.lower())

# apply preprocessing
train['Question_tok'] = train['Question'].map(preprocess)
train['Sentence_tok'] = train['Sentence'].map(preprocess)
dev['Question_tok'] = dev['Question'].map(preprocess)
dev['Sentence_tok'] = dev['Sentence'].map(preprocess)
test['Question_tok'] = test['Question'].map(preprocess)
test['Sentence_tok'] = test['Sentence'].map(preprocess)
train[0:5]

#map word to IDs
def word2id(sent):
    return list(map(lambda x: dictionary.get(x, 1), sent))

# Pad sentence size to a fixed size lenght
max_len_q = 40
max_len_a = 40

dictionary = {'PAD':0, 'UNK':1}
#dictionary
dictionaryW2V = {'PAD':0, 'UNK':1}

# ft = KeyedVectors.load_word2vec_format('data/wiki-news-300d-1M.vec', binary=False)
dictionaryFT = {'PAD':0, 'UNK':1}

#functions and 
stop = set(stopwords.words('english'))
def count_feat(que, ans):
    return len((set(que)&set(ans))-stop)
def overlap(que, ans):
    return (set(que)&set(ans))
def overlap_feats(st, overlapping):
    return [1 if word not in overlapping else 2 for word in st]


# path of embeddings,
# binary = true if word2vec, = false if fastText
def import_data_with_embeddings_into_dict(path, binary):
    # load the word embedding and prepare the dictionary for our dataset
    print("STARTING TO load embeddings {}".format(path))
    embed = KeyedVectors.load_word2vec_format(path, binary=binary)
    print("DONE WITH loading embeddings {}".format(path))
    toks = (set(chain.from_iterable(train['Question_tok'])) | set(chain.from_iterable(train['Sentence_tok'])) | \
        set(chain.from_iterable(dev['Question_tok'])) | set(chain.from_iterable(dev['Sentence_tok']))     | \
        set(chain.from_iterable(test['Question_tok'])) | set(chain.from_iterable(test['Sentence_tok'])))

    i = 2
    for _, tok in enumerate(toks):
        if tok in embed:
            dictionary[tok] = i
            i+=1
    len(dictionary)

    # add mapping to ids
    train['Question_'] = train['Question_tok'].map(word2id)
    train['Sentence_'] = train['Sentence_tok'].map(word2id)
    dev['Question_'] = dev['Question_tok'].map(word2id)
    dev['Sentence_'] = dev['Sentence_tok'].map(word2id)
    test['Question_'] = test['Question_tok'].map(word2id)
    test['Sentence_'] = test['Sentence_tok'].map(word2id)
    train[0:5]

    max(len(list(sent)) for sent in train['Question_'])
    max(len(list(sent)) for sent in train['Sentence_'])

    train['Question_'] = train['Question_'].apply(lambda s: pad_sequences([s], max_len_q)[0])
    train['Sentence_'] = train['Sentence_'].apply(lambda s: pad_sequences([s], max_len_a)[0])
    dev['Question_'] = dev['Question_'].apply(lambda s: pad_sequences([s], max_len_q)[0])
    dev['Sentence_'] = dev['Sentence_'].apply(lambda s: pad_sequences([s], max_len_a)[0])
    test['Question_'] = test['Question_'].apply(lambda s: pad_sequences([s], max_len_q)[0])
    test['Sentence_'] = test['Sentence_'].apply(lambda s: pad_sequences([s], max_len_a)[0])
    train[0:5]
    
    return embed


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



# prepare the embedding matrix for model initialization
def emb_matrix(dictionar, emb_, dim):
    embedding_matrix = np.zeros((len(dictionar), dim))
    for word in dictionar:
        if word in emb_:
            embedding_matrix[dictionar[word]] = emb_[word]
    return embedding_matrix


############################ SIMPLEST MODEL ############################
# This is the simplest version of the model where the two sentence embedding 
# (Created via Convolution+Pooling) are concatenated and classified using a MLP.
def basic_model(emb, dictionar, dim):        
    np.random.seed(42)

    que = Input(shape=(max_len_q,)) # question
    ans = Input(shape=(max_len_a,)) # answer

    # sequential model for questions with embedding, convulutional and pooling layer
    que_model = Sequential() # sequential model for question
    que_model.add(Embedding(len(dictionar), dim,input_length=max_len_q, weights=[emb_matrix(dictionar, emb, dim)], trainable=True)) # add question embedding
    que_model.add(Convolution1D(100, 5, activation='tanh')) # add convolutional layer
    que_model.add(GlobalAveragePooling1D()) # add pooling layer

    # sequential model for answers with embedding, convulutional and pooling layer
    ans_model = Sequential() # sequential model for answer
    ans_model.add(Embedding(len(dictionar), dim,input_length=max_len_a, weights=[emb_matrix(dictionar, emb, dim)], trainable=True)) # add answer embedding
    ans_model.add(Convolution1D(100, 5, activation='tanh')) # add convolutional layer
    ans_model.add(GlobalAveragePooling1D()) # add pooling layer

    que_emb = que_model(que) #var for question model
    ans_emb = ans_model(ans) #var for answer model

    join = concatenate([que_emb, ans_emb]) #concatenate embedding of question and answers

    # create classifier 
    classify = Sequential() 
    classify.add(Dense(100, activation='tanh', input_dim=200))
    classify.add(Dense(1, activation='sigmoid'))
    out = classify(join) #output of classifier

    ## model creation
    model = Model(inputs=[que, ans], outputs=[out]) # model of question and answers
    model.summary() # model of question and answers

    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    return model

############ SECOND MODEL ############
## The second version of the network will use an additional features (i.e. word overlap count) 
## this is a very informative feature that alone provides ~49 MAP
def model_with_overlaps(emb2, dictionar, dim, dim2):
    train['count'] = pd.Series(count_feat(que, ans) for que, ans in zip(train['Question_tok'], train['Sentence_tok']))
    test['count'] = pd.Series(count_feat(que, ans) for que, ans in zip(test['Question_tok'], test['Sentence_tok']))
    dev['count'] = pd.Series(count_feat(que, ans) for que, ans in zip(dev['Question_tok'], dev['Sentence_tok']))
    train[0:5]

    ############################## WORD OVERLAP ##############################

    train['overlap'] = pd.Series(overlap(que, ans) for que, ans in zip(train['Question_tok'], train['Sentence_tok']))
    test['overlap'] = pd.Series(overlap(que, ans) for que, ans in zip(test['Question_tok'], test['Sentence_tok']))
    dev['overlap'] = pd.Series(overlap(que, ans) for que, ans in zip(dev['Question_tok'], dev['Sentence_tok']))
    train[0:5]

    ## Now we use word overlaps as binary feature at the embedding level
    train['Question_ov'] = pd.Series(overlap_feats(que, ov) for que, ov in zip(train['Question_tok'], train['overlap']))
    train['Sentence_ov'] = pd.Series(overlap_feats(que, ov) for que, ov in zip(train['Sentence_tok'], train['overlap']))
    dev['Question_ov'] = pd.Series(overlap_feats(que, ov) for que, ov in zip(dev['Question_tok'], dev['overlap']))
    dev['Sentence_ov'] = pd.Series(overlap_feats(que, ov) for que, ov in zip(dev['Sentence_tok'], dev['overlap']))
    test['Question_ov'] = pd.Series(overlap_feats(que, ov) for que, ov in zip(test['Question_tok'], test['overlap']))
    test['Sentence_ov'] = pd.Series(overlap_feats(que, ov) for que, ov in zip(test['Sentence_tok'], test['overlap']))

    train['Question_ov'] = train['Question_ov'].map(lambda s: pad_sequences([s], max_len_q)[0])
    train['Sentence_ov'] = train['Sentence_ov'].map(lambda s: pad_sequences([s], max_len_a)[0])
    dev['Question_ov'] = dev['Question_ov'].map(lambda s: pad_sequences([s], max_len_q)[0])
    dev['Sentence_ov'] = dev['Sentence_ov'].map(lambda s: pad_sequences([s], max_len_a)[0])
    test['Question_ov'] = test['Question_ov'].map(lambda s: pad_sequences([s], max_len_q)[0])
    test['Sentence_ov'] = test['Sentence_ov'].map(lambda s: pad_sequences([s], max_len_a)[0])
    train[0:5]

    ### MODEL
    np.random.seed(42)

    que = Input(shape=(max_len_q,))
    ans = Input(shape=(max_len_a,))
    que_ov = Input(shape=(max_len_q,))
    ans_ov = Input(shape=(max_len_a,))
    cnt = Input(shape=(1,))

    ## create embedding with word overlaps
    que_ov_emb = Embedding(3, 5,input_length=max_len_q)(que_ov)
    que_word_emb = Embedding(len(dictionar), dim , weights=[emb_matrix(dictionar, emb2, dim)],input_length=max_len_q, trainable=True)(que)

    que_emb = concatenate([que_ov_emb, que_word_emb])

    ans_ov_emb = Embedding(3, 5,input_length=max_len_a)(ans_ov)
    ans_word_emb = Embedding(len(dictionar), dim , weights=[emb_matrix(dictionar, emb2, dim)],input_length=max_len_a, trainable=True)(ans)

    ans_emb = concatenate([ans_ov_emb, ans_word_emb])

    que_model = Sequential()
    #que_model.add(Embedding(len(dictionary), 50 , weights=[emb_matrix(dictionary, w2v)],input_length=max_len_q, trainable=True))
    que_model.add(Convolution1D(100, 5, activation='tanh', kernel_initializer='lecun_uniform', input_shape=(max_len_a, dim2)))
    que_model.add(GlobalMaxPooling1D())


    ans_model = Sequential()
    #ans_model.add(Embedding(len(dictionary), 50 , weights=[emb_matrix(dictionary, w2v)],input_length=max_len_a, trainable=True))
    ans_model.add(Convolution1D(100, 5, activation='tanh', kernel_initializer='lecun_uniform', input_shape=(max_len_a, dim2)))
    ans_model.add(GlobalMaxPooling1D())

    que_emb = que_model(que_emb)
    ans_emb = ans_model(ans_emb)

    join = concatenate([que_emb, ans_emb, cnt])

    classify = Sequential()
    classify.add(Dense(100, activation='tanh', input_dim=201))
    classify.add(Dense(1, activation='sigmoid'))
    out = classify(join)

    model = Model(inputs=[que, ans,que_ov, ans_ov, cnt], outputs=[out])
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    return model

# data model prediction for basic model
def data(dataset):
    return (dataset['QuestionID'],dataset['SentenceID']), [np.vstack(dataset['Question_'].tolist()), np.vstack(dataset['Sentence_'].tolist())], np.vstack(dataset['Label'].tolist())

# data model prediction with words overlaps
def dataTre(dataset):
    return (dataset['QuestionID'],dataset['SentenceID']), [np.vstack(dataset['Question_'].tolist()), np.vstack(dataset['Sentence_'].tolist()),np.vstack(dataset['Question_ov'].tolist()), np.vstack(dataset['Sentence_ov'].tolist()), dataset['count'].as_matrix()], np.vstack(dataset['Label'].tolist())


def basic_model_pred(emb, dictionar, dim, newname): 
    # model prediction for basic model
    
    model = basic_model(emb, dictionar, dim)
    print("\n\n basic model fit\n")
    model.fit([np.vstack(train['Question_'].tolist()), np.vstack(train['Sentence_'].tolist())],
            np.vstack(train['Label'].tolist()), batch_size=100, epochs=100000, shuffle=True, verbose=2,
            callbacks=[EpochEval(data(dev), map_score_filtered, patience=5)])

    nameb ='./project/models' + newname + '_basic.h5'
    os.rename('qa.h5', nameb)

    print("\n\n basic model - over test set \n")
    (qid,_ ), X, lab = data(test)
    model = load_model(nameb)
    pred = model.predict(X)
    map_score_filtered(qid, lab, pred)
    map_score(qid, lab, pred)
    test['pred'] = pd.Series(y for y in pred)
    test[0:5]

    print("\n\n basic model - over training set \n")
    (qid,_ ), X, lab = data(train)
    pred = model.predict(X)
    print(map_score_filtered(qid, lab, pred))
    print(map_score(qid, lab, pred))
    train['pred'] = pd.Series(y for y in pred)
    train

def overlap_model_pred(emb, dictionar, dim, dim2, newname): 
    # model prediction with words overlaps
    model_over = model_with_overlaps(emb, dictionar, dim, dim2)
    print("\n\nmodel with overlaps fit\n")
    model_over.fit([np.vstack(train['Question_'].tolist()), np.vstack(train['Sentence_'].tolist()), np.vstack(train['Question_ov'].tolist()), np.vstack(train['Sentence_ov'].tolist()), train['count'].as_matrix()],
          np.vstack(train['Label'].tolist()), batch_size=100, epochs=100000, shuffle=True, verbose=2,
          callbacks=[EpochEval(dataTre(dev), map_score_filtered, patience=5)])

    nameo ='./project/models' + newname + '_overlap.h5'
    os.rename('qa.h5', nameo)

    print("\n\nmodel with overlaps - over test set \n")
    model_over = load_model(nameo)
    (qid,_ ), X, lab = dataTre(test)
    pred_over = model_over.predict(X)
    print(map_score_filtered(qid, lab, pred_over))
    print(map_score(qid, lab, pred_over))
    test['pred_ov'] = pd.Series(y for y in pred_over)
    test[0:5]

    print("\n\nmodel with overlaps - over training set \n")
    (qid,_ ), X, lab = dataTre(train)
    pred_over = model_over.predict(X)
    print(map_score_filtered(qid, lab, pred_over))
    print(map_score(qid, lab, pred_over))
    train['pred_ov'] = pd.Series(y for y in pred_over)
    train[0:5]

# create embeddings
emb2 = import_data_with_embeddings_into_dict('data/aquaint+wiki.txt.gz.ndim=50.bin', binary=True)
dictionaryW2V = dictionary
dictionary = {'PAD':0, 'UNK':1}
print("\n\ncreating basic model with W2V embedding\n")
basic_model_pred(emb2, dictionaryW2V, 50, '/qa_W2V')
print("\n\ncreating overlap model with W2V embedding\n")
overlap_model_pred(emb2, dictionaryW2V, 50, 55, '/qa_W2V')


emb_over = import_data_with_embeddings_into_dict('data/wiki-news-300d-1M.vec', binary=False)
dictionaryFT = dictionary
print("\n\ncreating basic model with FastText embedding\n")
basic_model_pred(emb_over, dictionaryFT, 300, '/qa_ft')
print("\n\ncreating models with FastText embedding\n")
overlap_model_pred(emb_over, dictionaryFT, 300, 305, '/qa_ft')
