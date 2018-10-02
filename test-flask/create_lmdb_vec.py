from lmdb_embeddings.reader import LmdbEmbeddingsReader
from lmdb_embeddings.exceptions import MissingWordError
from gensim.models.keyedvectors import KeyedVectors
from lmdb_embeddings.writer import LmdbEmbeddingsWriter
import os
import lmdb

path="/opt/data"
os.chdir(path)
OUTPUT_DATABASE_FOLDER='lmdb_databases'
gensim_model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')

def iter_embeddings():
    for word in gensim_model.vocab.keys():
        yield word, gensim_model[word]

print('Writing vectors to a LMDB database...')
writer = LmdbEmbeddingsWriter(
    iter_embeddings()
).write(OUTPUT_DATABASE_FOLDER)
print('All vectors written successfully.')