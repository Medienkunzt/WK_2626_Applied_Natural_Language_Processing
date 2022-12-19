# nltk
# spacy
from sklearn.feature_extraction.text import CountVectorizer

txt = "Alice was beginning to get very tired of sitting by her sister on the bank and of having " \
      "nothing to do once or twice she had peeped into the book her sister was reading but it had no " \
      "pictures or conversations in it and what is the use of a book thought Alice without pictures or conversation"

txt = txt.lower()
words = txt.split()

print(words)

wordDict = {}

# DIY
for word in words:
    cnt = wordDict.get(word)
    if cnt == None:
        wordDict[word] = 1
    else:
        wordDict[word] = cnt + 1
    wordDict.update()

wordDict_items = wordDict.items()

sorted_items = sorted(wordDict_items)
print(sorted_items)
bow = []
for i in sorted_items:
    bow.append(i[1])

print(bow)

# Alternative... use scikit learn
vectorizer = CountVectorizer()

X = vectorizer.fit_transform([txt])
print(vectorizer.get_feature_names_out())
print("Dictionary:", vectorizer.vocabulary_)
print("Mapping:", X)
# Note: Vectorizer ignores single character words
print(X.toarray()[0])
