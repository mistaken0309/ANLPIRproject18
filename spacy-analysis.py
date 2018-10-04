import spacy
import io
import numpy as np
import time

nlp = spacy.load('en', disable=['parser', 'ner'])
# print(nlp.pipe_names)

# doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

ff = io.open('../data/WikiQA-dev.txt', 'r', encoding='utf-8')
# doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
doc = nlp(ff.read())
dictionary = {'PAD':0, 'UNK':1}
tags = set()
tags_num = {}
embeddings = {}
embeds = {}

i=1
for token in doc:
    # print(token.text + "\t" + token.tag_)
    if token.tag_ not in list(tags_num.keys()):
        tags_num[token.tag_] = i
        i+=1
        # print()
    #print(token.tag_ + "\t\t" + str(tags_num[token.tag_]))

# print(tags_num)

i = 2
for token in doc:
    dictionary[token.text] = i
    # print(str(dictionary[token.text]) + "\t" + token.text)
    tag = token.tag_
    tag = tags_num[tag]
    # print(tag_in + "\t" +str(tag))
    if embeddings.get(token.text) is not None:
        if tag not in embeddings[token.text]:
            embeddings[token.text].append(tag)
    else:
        embeddings[token.text] = [tag]
    embeds[token.text] = tag

    i+=1
print(str(len(dictionary)) + "\t" + str(len(embeddings)) + "\t" + str(len(tags_num)))

for word in dictionary.keys():
    print(str(dictionary[word]) + "\t\t" + str(embeddings[word]) + "\t\t" + str(embeds[word]))



def emb_matrix(dictionar, emb_):
    embedding_matrix = np.zeros((len(dictionar)), int)
    for word in dictionar:
        embedding_matrix[dictionar[word]] = emb_[word]
    return embedding_matrix

emb_mat = emb_matrix(dictionary, embeds)
print(emb_mat)
