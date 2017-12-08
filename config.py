'''
Created on 21 nov. 2017

@author: lvanni
'''

# Model Hyperparameters

# TOKENIZER : KEEP ONLY MOST FREQUENT WORD (NONE == ALL)
MAX_NB_WORDS = None

# ------------- EMBEDDING -------------
EMBEDDING_DIM = 128

# float >= 0. 0 for no negative (=random) samples. 1 for same number as positive samples. etc.
NEGATIVE_SAMPLES = 5.

# Window size for finding colocation (cooccurrence calcul)
WINDOW_SIZE = 5

# ------------- CNN -------------
# Size of the validation dataset (0.1 => 10%)
VALIDATION_SPLIT = 0.1

# SIZE OF SENTENCE (NUMBER OF WORDS)
MAX_SEQUENCE_LENGTH = 20

DENSE_LAYER_SIZE = 100
FILTER_POOL_LENGTHS = [2,2,2]
FILTER_SIZES = [3,4,5]
DROPOUT_VAL = 0.5

NUM_EPOCHS = 10
BACH_SIZE = 50

# label delimiter
LABEL_MARK = "__"