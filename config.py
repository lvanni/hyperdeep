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
VALIDATION_SPLIT = 0.2

# SIZE OF SENTENCE (NUMBER OF WORDS)
MAX_SEQUENCE_LENGTH = 50

NB_FILTERS = 512

# 3 filtres:
# 1) taille 3
# 2) taille 4
# 3) taille 5
FILTER_SIZES = [3,3,3]

# 3 filtres de taille 2 a chaque fois pour le maxpooling
FILTER_POOL_LENGTHS = [3,3,3]

DROPOUT_VAL = 0.2

DENSE_LAYER_SIZE = 300

NUM_EPOCHS = 5
BACH_SIZE = 50

# label delimiter
LABEL_MARK = "__"