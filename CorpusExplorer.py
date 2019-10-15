import nltk
from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import brown
from collections import defaultdict, Counter
# RECOMIENDO QUE USEN UN JUPYTER NOTEBOOK PARA PODER CORRER PEDAZO A PEDAZO Y EXPLORAR LOS CORPUS, EL BUILD DATA SET METHOD ES EL QUE ESTABA EN EL LIBRO QUE ADEL NOS MANDO


# for fileid in gutenberg.fileids():
#      num_chars = len(gutenberg.raw(fileid))
#      num_words = len(gutenberg.words(fileid))
#      num_sents = len(gutenberg.sents(fileid))
#      num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
#      print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)
#
# for fileid in nltk.corpus.webtext.fileids():
#          print(webtext.raw(fileid)[:25], '...')
#
counter = 0
sentance_num = 0
len(brown.words(categories=['news']))
for x in brown.words(categories=['news']):
    if x == '?':
        counter = counter + 1
    if x == '.' or x =='?':
        sentance_num = sentance_num + 1
counter


# Keeps words and pos into a dictionary
# where the key is a word and
# the value is a counter of POS and counts
word_tags = defaultdict(Counter)


def build_dataset(words, n_words):
 count = [['UNK', -1]]
 count.extend(collections.Counter(words).most_common(n_words - 1))
 dictionary = dict()
 for word, _ in count:
     dictionary[word] = len(dictionary)
     data = list()
     unk_count = 0
 for word in words:

     if word in dictionary:
         index = dictionary[word]
     else:
        index = 0 # dictionary['UNK']
        unk_count += 1
        data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
 return data, count, dictionary, reversed_dictionary
