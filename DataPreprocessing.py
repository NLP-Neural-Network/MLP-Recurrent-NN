
from llist import dllist
import nltk
from nltk.corpus import brown, treebank


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
    def parse_Questions(corpus, c):
        count = c
        for sentence in corpus.sents():
            vocab_v0[count] = sentence
            count = count + 1
        return count

    # Method that receives list of tagged sentences and iterates through them to pass them into a dictionary.
    # Brown_mapping and tree_mapping equals the map of sentences:tags
    @staticmethod
    def dataSegmentation(tagged_corpus):
        keyWords = []
        xMatrix = []
        yMatrix = []
        for key in wordVocab.keys():
            keyWords.append(key)
        sentences, sentence_tags =[], []
        for tagged_sentence in tagged_corpus:
            sentence, tags = zip(*tagged_sentence)
            sentences.append(np.array(sentence))
            sentence_tags.append(np.array(tags))

        words, tags = set([]), set([])

        for s in sentences:
            for w in s:
                words.add(w.lower())

        for ts in sentence_tags:
            for t in ts:
                tags.add(t)

        word2index = {w: i  for i, w in enumerate(list(words))}
        #word2index['-PAD-'] = 0  # The special value used for padding
        #word2index['-OOV-'] = 0  # The special value used for OOVs
        tag2index = {t: i for i, t in enumerate(list(tags))}
        #tag2index['-PAD-'] = 0  # The special value used to padding

        for sent in tagged_corpus:
            sentence = []
            tags = []
            for word in sent:
                sentence.append(int(wordVocab[word[0]]))
                tags.append(tag_mapping[word[1]])
        tag2index
        train_sentences_X, train_tags_y = [], []

            yMatrix.append(tags)
            xMatrix.append(sentence)
        for s in sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(word2index[w.lower()])
                except KeyError:
                    s_int.append(word2index['-OOV-'])

            train_sentences_X.append(s_int)

        for s in sentence_tags:
            train_tags_y.append([tag2index[t] for t in s])

        return np.asarray(train_sentences_X), np.asarray(train_tags_y)

        return np.asarray(xMatrix), np.asarray(yMatrix)

    # Function that takes the labels variables with integers and iterates through  to
    # hot encode them and return the transformed data
    @staticmethod
    def labelWordEncoder():
        all_tags = set([tag for sentence in brown.tagged_sents(tagset='universal') for _, tag in sentence])
    def labelWordEncoder(corpus):
        all_tags = set([tag for sentence in corpus.tagged_sents(tagset='universal') for _, tag in sentence])
        one_hot_labels = to_categorical([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], num_classes=12)
        # Iterate through all the tags to do an int mapping
        c = 0
# /////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////MAIN//////////////////////////////////////////////

nltk.download('brown')
nltk.download('punkt')
nltk.download('universal_tagset')

one_hot_labels = None
vocab_v0 = {}
wordVocab = {}
tag_mapping = {}
Pre.parse_Questions(brown)
Pre.labelWordEncoder()
brown_tagsents = dllist(brown.tagged_sents(tagset='universal'))
count_until = Pre.parse_Questions(brown,0)

Pre.labelWordEncoder(brown)

brown_tagsents = brown.tagged_sents(tagset='universal')
xTrain, yTrain = Pre.dataSegmentation(brown_tagsents)

############################################              MODEL                    ########################################################################
# Data Partition
# Padding Vectorized Data Set
x_train = tf.keras.preprocessing.sequence.pad_sequences(xTrain[:40000], padding='post')
y_train = tf.keras.preprocessing.sequence.pad_sequences(yTrain[:40000], padding='post')
x_data = tf.keras.preprocessing.sequence.pad_sequences(xTrain, padding='post')
y_data = tf.keras.preprocessing.sequence.pad_sequences(yTrain, padding='post')

x_validation = x_train[30000:]
y_validation = y_train[30000:]
x_train = x_train[:30000]
y_train = y_train[:30000]
x_training = x_data[:50000]
y_training = y_data[:50000]

x_validation = x_data[50000:]
y_validation = y_data[50000:]
x_training.shape
lol = to_categorical(y_training,)
lol[0]

print(y_train.shape, x_train.shape)
vocab_size = len(wordVocab.keys())




model = models.Sequential()
model.add(layers.Embedding(input_dim=vocab_size+1, input_length=180, output_dim=100, mask_zero=True))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(12, activation='softmax', ))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.add(layers.InputLayer(input_shape=(180, )))
model.add(layers.Embedding(input_dim=vocab_size, input_length=180, output_dim=200, mask_zero=True))
model.add(layers.Bidirectional(LSTM(256, return_sequences=True)))
model.add(layers.TimeDistributed(Dense(12)))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


history =  model.fit(x_training,to_categorical(y_training), batch_size=128, epochs=40, validation_split=.2)



history = model.fit(x_train, y_train, epochs=20, batch_size=512,
                    validation_data=(x_validation, y_validation))

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
plt.legend()

plt.show()

plt.show()
