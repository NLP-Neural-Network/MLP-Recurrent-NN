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
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Reshape
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


    # /////////////////////////////////Clases////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////////////////////////////////////////
class Pre:
        sentences = []
        sentence_tags = []

        @staticmethod
        def parse_Questions(corpus, c):
            count = c
            for sentence in corpus.sents():
                vocab_v0[count] = sentence
                count = count + 1
            return count

        # Method that receives list of tagged sentences and iterates through them to build word set and tag set.
        @staticmethod
        def vocabularyBuilder(tagged_corpus):

            for tagged_sentence in tagged_corpus:
                sentence, tags = zip(*tagged_sentence)
                Pre.sentences.append(np.array(sentence))
                Pre.sentence_tags.append(np.array(tags))

            words, tags = set([]), set([])

            for s in Pre.sentences:
                for w in s:
                    words.add(w.lower())

            for ts in Pre.sentence_tags:
                for t in ts:
                    tags.add(t)

            word2index = {w: i + 2 for i, w in enumerate(list(words))}
            word2index['-PAD-'] = 0  # The special value used for padding
            word2index['-OOV-'] = 0  # The special value used for OOVs
            tag2index = {t: i for i, t in enumerate(list(tags))}

            return word2index,tag2index

        #Method that segments the data into 2 matrices each with a value mapping to an individual word in the x matrix and its corresponding universal tag in y
        @staticmethod
        def dataSegmentation(word2index,tag2index):
            train_sentences_X, train_tags_y = [], []

            for s in Pre.sentences:
                s_int = []
                for w in s:
                    try:
                        s_int.append(word2index[w.lower()])
                    except KeyError:
                        s_int.append(word2index['-OOV-'])

                train_sentences_X.append(s_int)

            for s in Pre.sentence_tags:
                train_tags_y.append([tag2index[t] for t in s])

            return np.asarray(train_sentences_X), np.asarray(train_tags_y)


        # Function that takes the labels variables with integers and iterates through  to
        # hot encode them and return the transformed data
        @staticmethod
        def labelWordEncoder(corpus):
            all_tags = set([tag for sentence in corpus.tagged_sents(tagset='universal') for _, tag in sentence])
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


one_hot_labels = None
vocab_v0, tag_mapping  = Pre.vocabularyBuilder(brown.tagged_sents(tagset='universal'))
wordVocab = {}
Pre.parse_Questions(brown,0)

brown_tagsents = brown.tagged_sents(tagset='universal')
xTrain, yTrain = Pre.dataSegmentation(vocab_v0,tag_mapping)

    ############################################              MODEL                    ########################################################################
    # Data Partition
    # Padding Vectorized Data Set
x_data = tf.keras.preprocessing.sequence.pad_sequences(xTrain, padding='post')
y_data = tf.keras.preprocessing.sequence.pad_sequences(yTrain, padding='post')

x_training = x_data[:50000]
y_training = y_data[:50000]

x_validation = x_data[50000:]
y_validation = y_data[50000:]


vocab_size = len(vocab_v0)


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(180, )))
    model.add(layers.Embedding(input_dim=vocab_size, input_length=180, output_dim=200, mask_zero=True))
    model.add(layers.Bidirectional(LSTM(256, return_sequences=True)))
    model.add(layers.TimeDistributed(Dense(12)))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


    history =  model.fit(x_training,to_categorical(y_training), batch_size=512, epochs=40)




loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.show()
