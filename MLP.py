import nltk
from nltk.corpus import treebank
import numpy as np
import tensorflow as tf

#Download Corpus
nltk.download('treebank')
nltk.download('universal_tagset')

#Tagging the corpus and identifying all the tags
sentences = treebank.tagged_sents(tagset='universal')
tags = set([tag for sentence in treebank.tagged_sents() for _, tag in sentence])

#Training and testing data separation
training_data = sentences[:int(.80 * len(sentences))]
testing_data = sentences[int(.80 * len(sentences)):]

c = len(training_data) - 1
features = []
labels = []

while c > 0:
    for string in training_data[c]:
         features.append(string[0])
         labels.append(string[1])
         print(string[1])
         c -= 1





