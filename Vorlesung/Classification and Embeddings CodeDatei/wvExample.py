from gensim import corpora
import gensim
import gensim.downloader as api
from gensim import models

# word2vec_model300 = api.load('word2vec-google-news-300')

# word2vec_model300.save('myw2v')

word2vec_model300 = models.KeyedVectors.load('myw2v')

dist1 = word2vec_model300.similarity('spoon', 'tea')
dist2 = word2vec_model300.similarity('moon', 'tea')
dist3 = word2vec_model300.similarity('coffee', 'tea')

print(dist1, dist2, dist3)

print(word2vec_model300.most_similar(positive=['coffee'], topn=5))

print(word2vec_model300.doesnt_match(["cow", "king", "queen", "man", "woman"]))
