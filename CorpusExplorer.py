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


#preprocessing method that wont add questions, imperfect tho
#returns dictionary with number of sentances as the key and the sentance as a value
def parse_Questions(corpus):
    counter = 0
    sentance_num = 0

    Vocab_v0 = dict()
    count = 0
    for sentance in corpus.sents(categories=['news']):
        for items in sentance :
            if items == '.':
                Vocab_v0[count] = sentance
                count = count + 1
    return Vocab_v0

lol = parse_Questions(brown)



counter = 0
sentance_num = 0
len(brown.words(categories=['news']))
for x in brown.words(categories=['news']):
    if x == '?':
        counter = counter + 1
    if x == '.' or x =='?':
        sentance_num = sentance_num + 1



# Keeps words and pos into a dictionary
# where the key is a word and
# the value is a counter of POS and counts
word_tags = defaultdict(Counter)
