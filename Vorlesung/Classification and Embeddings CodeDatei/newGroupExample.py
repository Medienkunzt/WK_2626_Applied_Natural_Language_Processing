import pandas as pd
import numpy as np

# http://qwone.com/~jason/20Newsgroups/
train_label = open('20news-bydate/matlab/train.label')

# read labels
labels = train_label.readlines()
nOfDoc = len(labels)
print("number of documents:", nOfDoc)

# print(labels)

# count labels
nOfLabels = {}
for line in labels:
    val = int(line.split()[0])
    if val in nOfLabels.keys():
        nOfLabels[val] = nOfLabels[val] + 1
    else:
        nOfLabels[val] = 1
    nOfLabels.update()

# calculate label probabilities
pOfLabels = {}
for item in nOfLabels:
    pOfLabels[item] = nOfLabels[item] / nOfDoc
print("Probabilities of labels:", pOfLabels)

# read word statistics
train_data = open('20news-bydate/matlab/train.data')
df_words = pd.read_csv(train_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])

# add labels to word statistics
lOfLabels = []
listOfLabels = []
for ind in range(len(labels)):
    l = int(labels[ind].split()[0])
    lOfLabels.append([ind + 1, l])
df_labels = pd.DataFrame(lOfLabels)
df_labels.columns = ["docIdx", "classIdx"]
# print(df_labels)

mad = pd.merge(df_words, df_labels, how="inner", on=["docIdx"])

# vocabulary is available
vocab = open('vocabulary.txt')
vocab_df = pd.read_csv(vocab, names=['word'])
vocab_df = vocab_df.reset_index()
vocab_df['index'] = vocab_df['index'].apply(lambda x: x + 1)

# we remove some words that are too generic...
stop_words = [
    "a", "about", "above", "across", "after", "afterwards",
    "again", "all", "almost", "alone", "along", "already", "also",
    "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow",
    "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become", "becomes",
    "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by",
    "can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else",
    "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find", "for",
    "found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her",
    "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however",
    "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may",
    "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name",
    "namely", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing",
    "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise",
    "our", "ours", "ourselves", "out", "over", "own", "part", "perhaps", "please", "put", "rather", "re", "same", "see",
    "seem", "seemed", "seeming", "seems", "she", "should", "since", "sincere", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such", "take", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these",
    "they",
    "this", "those", "though", "through", "throughout",
    "thru", "thus", "to", "together", "too", "toward", "towards",
    "under", "until", "up", "upon", "us",
    "very", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while",
    "who", "whoever", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
]
# remove stop words
vocab_df = vocab_df[~vocab_df['word'].isin(stop_words)]
# print(vocab_df)

# print(mad)
mad = mad[mad['wordIdx'].isin(vocab_df['index'])]
# print(mad)
#
vSize = vocab_df.shape[0]
print("Size of vocabulary:", vSize)
# calculate probabilitities
alpha = 0.01
pb_ij = mad.groupby(['classIdx', 'wordIdx'])
# print(pb_ij['count'].sum())
pb_j = mad.groupby(['classIdx'])
# print(pb_j['count'].sum())

# print(">>>",alpha*vSize)
Pr = np.log((pb_ij['count'].sum() + alpha) / (pb_j['count'].sum() + alpha * vSize))

# Unstack series
Pr = Pr.unstack()
for c in range(1, 21):
    Pr.loc[c, :] = Pr.loc[c, :].fillna(np.log(alpha / (pb_j['count'].sum()[c] + alpha * vSize)))

print(Pr)

# Get test data
test_data = open('20news-bydate/matlab/test.data')
testWords = pd.read_csv(test_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])
target = testWords[testWords['docIdx'] == 400]

print("target", target)

# Get list of labels
test_label = pd.read_csv('20news-bydate/matlab/test.label', names=['t'])
testLabel = test_label['t'].tolist()
targetLabel = testLabel[400]
print("targetLabel", targetLabel)
# Use PR and target for prediction
res = []

for ind in range(1, 21):
    sum = np.log(pOfLabels[ind])
    val = 0
    for w, c in zip(target['wordIdx'].values, target['count'].values):
        try:
            val = val + c * Pr[w].loc[ind]
        except:
            val = val + 0
    sum = sum + val

    print(sum)
    res.append(np.exp(sum))

print(res / np.sum(res))
print("Predicted Label", np.argmax(res) + 1)
