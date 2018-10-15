# parsing a file
import json
from pprint import pprint
import nltk
from gensim.models import Word2Vec

with open('OrderedResult.txt') as contents:
    papers = json.load(contents)


wholeContents = ""
for paper in papers:
    title = paper['title']
    abstract = paper['abstract']
    
    wholeContents += title + ' '
    wholeContents += abstract + ' '

#print(wholeContents)
tokens = nltk.word_tokenize(wholeContents)
tagged = nltk.pos_tag(tokens)
#print(tagged)

NounsAndNounP = []
for eachTagged in tagged:
    if(eachTagged[1].find("NN") == 0 ):
        NounsAndNounP.append(eachTagged[0])
    
#print(NounsAndNounP)

embedding_model = Word2Vec([NounsAndNounP], size=30, window=5, min_count=10, workers=4, iter=100, sg=1)

df = embedding_model.most_similar(positive=['reinforcement'], topn=50)
print(df)

import matplotlib.pyplot as plt

fig = plt.figure()
fig.set_size_inches(40,20)
ax = fig.add_subplot(1,1,1)

ax.scatter(df[:0], df[:1])

for wwod, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.show()
