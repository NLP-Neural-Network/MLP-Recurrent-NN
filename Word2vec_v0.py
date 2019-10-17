import tensorflow as tf
from CorpusExplorer import parse_Questions
from nltk.corpus import brown


vocab_v0 = parse_Questions(brown)
