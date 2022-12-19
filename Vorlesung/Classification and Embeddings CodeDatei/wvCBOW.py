# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import nltk

nltk.download('punkt')
warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec

#  Reads ‘alice.txt’ file
sample = open("alice.txt", "r")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count=1,
                                vector_size=100, window=5)

print(model1.wv.most_similar("alice"))
