'''
Created on 21 nov. 2017

@author: lvanni
'''
# Model Hyperparameters
embedding_dim = 128
filter_sizes = (3,4,5)
num_filters = 100
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 50
num_epochs = 20

# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

# label delimiter
label_mark = "__"