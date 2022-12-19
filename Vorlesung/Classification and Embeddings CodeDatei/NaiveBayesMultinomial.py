from sklearn.naive_bayes import MultinomialNB
import numpy as np
import math

# generate random BOW
# size of vacabulary
V = 10
rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, V))
print(X)
# assume classification
y = np.array([1, 2, 3, 4, 5, 6])

print(y)

# now set target
target = X[0:1]

# generate model
phi = []
# uniform distribution for labels
m = 1 / 6

alpha = 1
for i in X:
    s = np.sum(i)
    phi.append([np.log((j + alpha) / (s + V * alpha)) for j in i])
phi = [[np.log((j + alpha) / (np.sum(i) + V)) for j in i] for i in X]

print(phi)

proba = []
for i in phi:
    res = i @ target.T
    res = res + np.log(m)
    proba.append(np.exp(res))

# calculate probabilities and make argmax prediction
print(proba / np.sum(proba))
print(np.argmax(proba) + 1)

# easy using scikit learn
clf = MultinomialNB(alpha=1)
clf.fit(X, y)

# print(clf.coef_)
print(clf.predict_proba(target))
print(clf.predict(target))
