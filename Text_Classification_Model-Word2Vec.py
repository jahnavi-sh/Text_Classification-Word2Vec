import gensim
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import string
import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

gensim.__version__

print(list(gensim.downloader.info()['models'].keys()))

wv = api.load('glove-twitter-50')

type(wv)

wv['apple']

len(wv['apple'])

wv = KeyedVectors.load('vectors.kv')

wv.similarity("apple", 'mango')

wv.similarity("apple", "car")

pairs = [
    ('car', 'minivan'),   # a minivan is a kind of car
    ('car', 'bicycle'),   # still a wheeled vehicle
    ('car', 'airplane'),  # ok, no wheels, but still a vehicle
    ('car', 'cereal'),    # ... and so on
    ('car', 'communism'),
]
for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))

print(wv.most_similar(positive=['car', 'minivan'], topn=5))

print(wv.doesnt_match(['fire', 'water', 'land', 'sea', 'air', 'car']))

#Semantic regularities captured in word embeddings 

wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=3)

wv.most_similar(positive=['woman', 'king'], topn=3)

words = ["one",'two','man','woman','table']

sample_vectors = np.array([wv[word] for word in words])
pca = PCA(n_components=2)
result = pca.fit_transform(sample_vectors)
result

#Visualizing the word vectors 

plt.figure(figsize=(12,12))
plt.scatter(result[:,0], result[:,1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()

data = pd.read_csv("toxic_commnets_500.csv",error_bad_lines=False, engine="python")
data.head()

def sent_vec(sent):
    vector_size = wv.vector_size
    wv_res = np.zeros(vector_size)
    # print(wv_res)
    ctr = 1
    for w in sent:
        if w in wv:
            ctr += 1
            wv_res += wv[w]
     wv_res = wv_res/ctr
    return wv_res

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    doc = nlp(sentence)

 # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in doc ]

    # print(mytokens)

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

sent_vec("I am happy")

nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
print(stop_words)

punctuations = string.punctuation
print(punctuations)

data['tokens'] = data['comment_text'].apply(spacy_tokenizer)

data.head()

data['vec'] = data['tokens'].apply(sent_vec)

data.head()

X = data['vec'].to_list()
y = data['toxic'].to_list()

X[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)

classifier = LogisticRegression()

classifier.fit(X_train,y_train)

predicted = classifier.predict(X_test)
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))