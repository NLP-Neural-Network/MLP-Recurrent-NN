import nltk
from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import brown
from nltk.corpus import reuters
from collections import defaultdict, Counter
from nltk import corpus
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


# RECOMIENDO QUE USEN UN JUPYTER NOTEBOOK PARA PODER CORRER PEDAZO A PEDAZO Y EXPLORAR LOS CORPUS, EL BUILD DATA SET METHOD ES EL QUE ESTABA EN EL LIBRO QUE ADEL NOS MANDO

nltk.download('brown')
nltk.download('gutenberg')
nltk.download('reuters')

nlp = spacy.load('brown')
print(nlp)


#preprocessing method that wont add questions, imperfect tho
#returns dictionary with number of sentances as the key and the sentance as a value
def parse_Questions(corpus):
    counter = 0
    sentance_num = 0

    #nlp.tokenizer.pipe(brown, gutenberg, reuters)


    doc = nlp.tokenizer.pipe(brown, gutenberg, reuters)
    Vocab_v0 = dict()
    count = 0


    for sent in doc.sents:
        if '?' not in sent & '!' not in sent:
            Vocab_v0[count] = sent
            count = count + 1


    print(count)
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
