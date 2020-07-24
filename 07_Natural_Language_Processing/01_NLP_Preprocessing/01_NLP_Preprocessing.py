import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import gutenberg

import string

import gensim
from gensim.models.phrases import Phraser, Phrases
from gensim.models.word2vec import Word2Vec

from sklearn.manifold import TSNE

import pandas as pd
from bokeh.io import output_notebook, output_file
from bokeh.plotting import show, figure

import matplotlib.pyplot as plt

print(len(gutenberg.words()))

# Tokenize the corpus in a list of sentences and then the sentences.
"""gberg_sent_tokens = sent_tokenize(gutenberg.raw())
word_tokenize(gberg_sent_tokens[1])"""

# Same in one step
gberg_sents = gutenberg.sents()

#==================================================================================================
# Preprocessing, example on one sentence (Lower, remove stopwords, stemming)
"""print(gberg_sents[4])
testlist = [w.lower() for w in gberg_sents[4]]
print(testlist)
stpwrds = stopwords.words('english') + list(string.punctuation)
print(stpwrds)
testlist = [w.lower() for w in gberg_sents[4] if w.lower() not in stpwrds]
print(testlist)
stemmer = PorterStemmer()
testlist = [stemmer.stem(w.lower()) for w in gberg_sents[4] if w.lower() not in stpwrds]
print(testlist)

phrases = Phrases(gberg_sents, min_count=10, threshold=500)
bigram = Phraser(phrases)
print(bigram.phrasegrams)"""

#==================================================================================================
# Preprocess the entire corpus (without stop-word removal and stemming)

"""lower_sents, clean_sents =[], []
for s in gberg_sents:
    lower_sents.append([w.lower() for w in s if w.lower() not in list(string.punctuation)])

lower_bigram = Phraser(Phrases(lower_sents, min_count=32, threshold=64))

for s in lower_sents:
    clean_sents.append(lower_bigram[s])

print(lower_sents[6], clean_sents[6])"""

#==================================================================================================
# Building the vectorspace

"""model = Word2Vec(sentences=clean_sents, size=64,
                 sg=1, window=10, iter=5,
                 min_count=10, workers=4)

model.save('clean_gutenberg_model.w2v')"""

# Load after running
model = gensim.models.Word2Vec.load('clean_gutenberg_model.w2v')

# Evaluate the vector space in different ways
print(len(model.wv.vocab))
print(model.wv['dog'])
print(model.wv.most_similar('arm', topn=3))
print(model.wv.doesnt_match(['mother', 'father', 'dog']))
print(model.wv.similarity('father', 'dog'))
print(model.wv.most_similar(positive=['father', 'woman'], negative=['man']))
print(model.wv.most_similar(positive=['husband', 'woman'], negative=['man']))

#==================================================================================================
# Visualizing the vector space
"""tsne = TSNE(n_components=2, n_iter=1000)
X_2d = tsne.fit_transform(model.wv[model.wv.vocab])
coords_df = pd.DataFrame(X_2d, columns=['x', 'y'])
coords_df['token'] = model.wv.vocab.keys()
coords_df.to_csv('clean_gutenberg_tsne.csv', index=False)"""

coords_df = pd.read_csv('clean_gutenberg_tsne.csv')
print(coords_df.head())

"""coords_df.plot.scatter('x', 'y', figsize=(12, 12), marker='.', s=10, alpha=0.2)
plt.show()"""

subset_df = coords_df.sample(n=5000)
p = figure(plot_width=800, plot_height=800)
p.text(x=subset_df.x, y=subset_df.y, text=subset_df.token)
show(p)


























