import tensorflow as tf
from CorpusExplorer import parse_Questions
from nltk.corpus import brown


vocab_v0 = parse_Questions(brown)

<<<<<<< HEAD

=======
train_data = vocab_v0.fromkeys
>>>>>>> 42f28da964f4ff3f745103f9cbbd50968191c445

embedding_dim=16

model = keras.Sequential([
  layers.Embedding(95000, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(1, activation='sigmoid')
])

model.summary()
