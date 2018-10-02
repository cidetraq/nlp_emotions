# coding: utf8
import sys
from flask import Flask, jsonify
import nltk
import boto3
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import wordnet
from keras.models import model_from_json
from keras.models import load_model
import os
import lmdb
from lmdb_embeddings.reader import LmdbEmbeddingsReader
import lmdb_embeddings.exceptions as exceptions
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import copy
import spacy
from spacy.tokens import Doc
from spacy.attrs import HEAD, DEP, POS
from sklearn.preprocessing import LabelEncoder

#Note: Cwd is /opt/python/bundle/Xnum/app where Xnum is a digit
orig_cwd=os.getcwd()
orig_cwd=copy.deepcopy(orig_cwd)

global graph
graph = tf.get_default_graph()

path="D:\Google Drive\Programming\python\\facebook market analyzer"
os.chdir(path)
embeddings=LmdbEmbeddingsReader('data\lmdb_databases')
os.chdir(orig_cwd)

nlp=spacy.load('en_core_web_sm')
encoder=LabelEncoder()

regressor=load_model('four_emotions.h5')

pos_IDS = [
    "NO_TAG",
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CONJ",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    "EOL",
    "SPACE"]

dep_labels=['ACL',
 'ACOMP',
 'ADVCL',
 'ADVMOD',
 'AGENT',
 'AMOD',
 'APPOS',
 'ATTR',
 'AUX',
 'AUXPASS',
 'CASE',
 'CC',
 'CCOMP',
 'COMPOUND',
 'CONJ',
 'CSUBJ',
 'CSUBJPASS',
 'DATIVE',
 'DEP',
 'DET',
 'DOBJ',
 'EXPL',
 'INTJ',
 'MARK',
 'META',
 'NEG',
 'NOUNMOD',
 'NPMOD',
 'NSUBJ',
 'NSUBJPASS',
 'NUMMOD',
 'OPRD',
 'PARATAXIS',
 'PCOMP',
 'POBJ',
 'POSS',
 'PRECONJ',
 'PREDET',
 'PREP',
 'PRT',
 'PUNCT',
 'QUANTMOD',
 'RELCL',
 'ROOT',
 'XCOMP']

dep_mappings={'COMPLM': 'MARK',
 'HMOD': 'COMPOUND',
 'HYPH': 'PUNCT',
 'INFMOD': 'ACL',
 'IOBJ': 'DATIVE',
 'NMOD': 'NOUNMOD',
 'NN': 'COMPOUND',
 'NPADVMOD': 'NPMOD',
 'NUM': 'NUMMOD',
 'NUMBER': 'COMPOUND',
 'PARTMOD': 'ACL',
 'POSSESSIVE': 'CASE',
 'RCMOD': 'RELCL'}

dep_deprecated_labels=['COMPLM',
 'INFMOD',
 'PARTMOD',
 'HMOD',
 'HYPH',
 'IOBJ',
 'NUM',
 'NUMBER',
 'NMOD',
 'NN',
 'NPADVMOD',
 'POSSESSIVE',
 'RCMOD']
# print a nice greeting.
def say_hello(username = "World"):
    return '<p>Hello %s!</p>\n' % clean(username)

def lmdb_vec_word(word):
    try:
        vector = embeddings.get_word_vector(word)
    except exceptions.MissingWordError:
        # 'google' is not in the database.
        return np.zeros(300, dtype='float32')
    return vector

def transform_y(y):
    encoder.fit(y)
    y=encoder.transform(y)
    y_1=np_utils.to_categorical(y)
    #y_1=np.reshape(y_1, (-1, 4, 1))
    return y_1

def vectorize(X, y=None,length=1):
    X_1=[]
    for index, row in enumerate(X[:length]):
        doc=nlp(row)
        doc=input_duplicator(doc)
        row_arr=[]
        for token in doc:
            text=lmdb_vec_word(token.text)
            pos=pos_IDS.index(token.pos_)
            zeros=np.zeros(300, dtype='float32')
            zeros[pos]=1
            pos=zeros
            dep=' '+token.dep_
            dep=dep.strip().upper()
            if dep=='':
                dep=len(dep_labels)+1
            elif dep in dep_labels:
                dep=dep_labels.index(dep)
            else:
                dep=dep_labels.index(dep_mappings[dep])
            zeros=np.zeros(300, dtype='float32')
            zeros[dep]=1
            dep=zeros
            head=lmdb_vec_word(token.head.text)
            token_arr=np.array([text,pos,dep,head])
            row_arr.append(token_arr)
        X_1.append(np.array(row_arr))
    X_1=np.array(X_1)
    if y is not None:
        y_1=y[:length]
        return X_1, y_1
    else:
        return X_1

def input_duplicator(row):
    text_only=[token.text for token in row]
    doc=row
    #attrs=[POS, DEP, HEAD]
    while len(text_only)<7:
        orig_row=text_only.copy()
        for token in orig_row:
            text_only.append(token)
        doc=' '.join(text_only)
        doc=nlp(doc)
    return doc

def transform_X_y(X, y=None, length=1):
    X_1 = []
    y_1= []
    if y is not None:
        X, y=vectorize(X, y, length)
    else:
        X=vectorize(X)
    for index in range(0, len(X)):
        doc=X[index]
        for i in range(6, len(doc)):
            X_1.append(doc[i-6:i])
            if y is not None:
                y_1.append(y.iloc[index])
    if y is not None:
        y_1=transform_y(y_1)
        return np.array(X_1), y_1
    else:
        return np.array(X_1)

def prediction(text):
    vec_text = transform_X_y(text)
    with graph.as_default():
        pred = regressor.predict(vec_text)
        pred = np.mean(pred, axis=0)
        anger=pred[0]
        fear=pred[1]
        joy=pred[2]
        sadness=pred[3]
        response_dict = {'anger': anger, 'fear': fear, 'joy': joy, 'sadness': sadness}
    return str(pred)

# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>Emotion Detector API</title> </head>\n<body>'''
instructions = '''
    <p><em>Description</em>: To use, append the text you want to get emotions for to the end of the URL.</p>\n'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__)

# add a rule for the index page.
application.add_url_rule('/', 'index', (lambda: header_text +
      str(orig_cwd)+instructions + footer_text))

# add a rule when the page is accessed with a name appended to the site
# URL.
application.add_url_rule('/<text>', 'prediction', (lambda text:
    prediction(text)))

# run the app.
application.debug = True
application.run()