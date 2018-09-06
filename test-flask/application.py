from flask import Flask, jsonify
import nltk
import boto3
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import wordnet
from keras.models import model_from_json
from gensim.models import KeyedVectors
from keras.models import load_model
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import json

orig_cwd=os.getcwd()
BUCKET_NAME = 'flasklstm'

global graph
graph = tf.get_default_graph()

path="/opt/data"
os.chdir(path)
en_model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
os.chdir(orig_cwd)
negative = ['not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except',
                         'even though', 'yet']
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)

regressor=load_model('four_emotions.h5')

def clean(doc):
    lemma=WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop if i not in negative])
    punc_free = "".join([ch for ch in stop_free if ch not in exclude])
    normalized = " ".join([lemma.lemmatize(word) for word in punc_free.split()])
    return normalized

# print a nice greeting.
def say_hello(username = "World"):
    return '<p>Hello %s!</p>\n' % clean(username)

def word_splits(series):
    word_splits=series.str.split(' ')
    return word_splits

def vec_word(li):
    total_vecs=[]
    for word in li:
        if word in en_model.vocab:
            vector = en_model[word]
            total_vecs.append(vector)
    return np.array(total_vecs)

def prediction(text):
    vec_text = transform_t6_input(text)
    with graph.as_default():
        pred = regressor.predict(vec_text)
        pred = np.mean(pred, axis=0)
        anger=pred[0]
        fear=pred[1]
        joy=pred[2]
        sadness=pred[3]
        response_dict = {'anger': anger, 'fear': fear, 'joy': joy, 'sadness': sadness, 'surprise': 0}
    #return jsonify(response_dict)
    return str(response_dict)

def transform_t6_input(text):
    X=pd.Series(text).apply(clean)
    splits=word_splits(X)
    numbers_series=splits.apply(vec_word)
    num_docs=len(numbers_series)
    for index, doc in enumerate(numbers_series):
        print(len(doc))
        while len(doc)<7:
            orig_doc=doc.copy()
            orig_doc=list(orig_doc)
            doc=list(doc)
            for word in orig_doc:
                doc.append(word)
            modified=True
        numbers_series.iloc[index]=np.array(doc)
    X_1 = []
    if num_docs>1:
        for index in range(0, num_docs):
            doc=numbers_series.iloc[index]
            for i in range(6, len(doc)):
                X_1.append(doc[i-6:i])
    else:
        doc=numbers_series.iloc[0]
        print(doc.shape)
        for i in range(6, len(doc)):
                X_1.append(doc[i-6:i])
    return np.array(X_1)

# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>EB Flask Test</title> </head>\n<body>'''
instructions = '''
    <p><em>Hint</em>: This is a RESTful web service! Append a username
    to the URL (for example: <code>/Thelonious</code>) to say hello to
    someone specific.</p>\n'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__)

# add a rule for the index page.
application.add_url_rule('/', 'index', (lambda: header_text +
    say_hello() + instructions + footer_text))

# add a rule when the page is accessed with a name appended to the site
# URL.
application.add_url_rule('/<text>', 'prediction', (lambda text:
    prediction(text)))

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()