from llist import dllist
import nltk
from nltk.corpus import brown, treebank
from collections import defaultdict, Counter
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
import gensim.downloader as api
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# /////////////////////////////////Clases////////////////////////////////////////////////////////////////
class node:
    yo = 0

    def __init__(self, element):
        self.element = element
        self.next = None
        self.prev = None


class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, element):
        if self.head is None:
            newnode = node(element)
            newnode.prev = None
            self.head = newnode
        else:
            newnode = node(element)
            temp = self.head
            while temp.next:
                temp = temp.next
            temp.next = newnode
            newnode.prev = temp
            newnode.next = None

    def prepend(self, element):
        if self.head is None:
            newnode = node(element)
            newnode.prev = None
            self.head = newnode
        else:
            newnode = node(element)
            self.head.prev = newnode
            newnode.next = self.head
            self.head = newnode
            newnode.prev = None

    def print_list(self):
        temp = self.head
        while temp:
            print(temp.data)
            temp = temp.next


# ///////////////////////////////////////////////////////////////////////////////////////////////////////
class Pre:

    @staticmethod
    def parse_Questions(corpus):
        count = 0
        for sentence in corpus.sents():
            vocab_v0[count] = sentence
            count = count + 1

    # Method that receives list of tagged sentences and iterates through them to pass them into a dictionary.
    # Brown_mapping and tree_mapping equals the map of sentences:tags
    @staticmethod
    def dataSegmentation(tagged_corpus):
        keyWords = []
        xMatrix = []
        yMatrix = []
        for key in wordVocab.keys():
            keyWords.append(key)

        for sent in tagged_corpus:
            sentence = []
            tags = []
            for word in sent:
                sentence.append(int(wordVocab[word[0]]))
                tags.append(tag_mapping[word[1]])

            yMatrix.append(tags)
            xMatrix.append(sentence)

        return np.asarray(xMatrix), np.asarray(yMatrix)

    # Function that takes the labels variables with integers and iterates through  to
    # hot encode them and return the transformed data
    @staticmethod
    def labelWordEncoder():
        all_tags = set([tag for sentence in brown.tagged_sents(tagset='universal') for _, tag in sentence])
        one_hot_labels = to_categorical([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], num_classes=12)
        # Iterate through all the tags to do an int mapping
        c = 0
        for tag in all_tags:
            tag_mapping[tag] = c
            c += 1

        keys = list(tag_mapping.keys())
        c = 0
        for value in tag_mapping.values():
            tag_mapping[keys[c]] = one_hot_labels[value]
            c += 1

        count = 0
        for sentence in vocab_v0.values():
            for word in sentence:
                if word not in wordVocab.keys():
                    wordVocab[word] = count
                    count = count + 1


# /////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////MAIN//////////////////////////////////////////////

nltk.download('brown')
nltk.download('punkt')
nltk.download('universal_tagset')

vocab_v0 = {}
wordVocab = {}
tag_mapping = {}
Pre.parse_Questions(brown)
Pre.labelWordEncoder()
brown_tagsents = dllist(brown.tagged_sents(tagset='universal'))
xTrain, yTrain = Pre.dataSegmentation(brown_tagsents)


############################################              MODEL                    ########################################################################
# Data Partition
x_reduced = xTrain[:15000]
y_reduced = xTrain[:15000]
print(xTrain.shape, yTrain.shape)

# Padding Vectorized Data Set
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_reduced, padding='post')
y_train = tf.keras.preprocessing.sequence.pad_sequences(y_reduced, padding='post')
print(y_train.shape, x_train.shape)

model = models.Sequential()
model.add(layers.Dense(100, activation='relu', input_shape=(48851, 9000,)))

lstm_x = LSTM(100)(x_train)
# model.add(layers.LSTM(16, activation='relu'))
model.add(layers.Dense(11, activation='relu'))
model.add(layers.Dense(11, activation='softmax', ))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', matrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test))

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
