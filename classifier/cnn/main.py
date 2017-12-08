import os
import json

import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import load_model

from classifier.cnn import models
from skipgram.skipgram_with_NS import create_vectors

from config import LABEL_MARK, DENSE_LAYER_SIZE, FILTER_POOL_LENGTHS, FILTER_SIZES, DROPOUT_VAL, NUM_EPOCHS, BACH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, VALIDATION_SPLIT, MAX_NB_WORDS

####################################
class Params:

	# Initalize defaut parameters
	dense_layer_size = DENSE_LAYER_SIZE
	filter_pool_lengths = FILTER_POOL_LENGTHS
	filter_sizes = FILTER_SIZES
	dropout_val = DROPOUT_VAL
	num_epochs = NUM_EPOCHS
	batch_size = BACH_SIZE
	inp_length = MAX_SEQUENCE_LENGTH
	embeddings_dim = EMBEDDING_DIM

class PreProcessing:
	
	def loadData(self, corpus_file):   
		
		print("loading data...")
		
		self.corpus_file = corpus_file
		
		label_dic = {}
		labels = []
		texts = []
		
		# Read text and detect classes/labels
		self.num_classes = 0
		for text in open(corpus_file, "r").readlines():
			label = text.split(LABEL_MARK + " ")[0].replace(LABEL_MARK, "")
			text = text.replace(label + " ", "")
			if label not in label_dic.keys():
				label_dic[label] = self.num_classes
				self.num_classes += 1
			label_int = label_dic[label]
			labels += [label_int]
			texts += [text]
		
		tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
		tokenizer.fit_on_texts(texts)
		sequences = tokenizer.texts_to_sequences(texts)

		#print(sequences)

		word_index = tokenizer.word_index
		print('Found %s unique tokens.' % len(word_index))

		data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

		labels = np_utils.to_categorical(np.asarray(labels))
		
		print('Shape of data tensor:', data.shape)
		print('Shape of label tensor:', labels.shape)

		# split the data into a training set and a validation set
		indices = np.arange(data.shape[0])
		np.random.shuffle(indices)
		data = data[indices]
		labels = labels[indices]
		nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

		self.x_train = data[:-nb_validation_samples]
		self.y_train = labels[:-nb_validation_samples]
		self.x_val = data[-nb_validation_samples:]
		self.y_val = labels[-nb_validation_samples:]
		self.word_index = word_index

	def loadEmbeddings(self, vectors_file):
		
		#print(vectors_file)
		#print(embeddings_src)

		word_index = self.word_index
		embeddings_index = {}
		
		if not vectors_file:
			vectors = create_vectors(self.corpus_file)
		else:
			vectors = open(vectors_file, "r").readlines()
			
		i=0
		for line in vectors:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
			EMBEDDING_DIM = len(coefs)
			i+=1
			if i>10000:
				break

		print('Found %s word vectors.' % len(embeddings_index))
		embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
		for word, i in word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				embedding_matrix[i] = embedding_vector

		self.embedding_matrix = embedding_matrix

def train(corpus_file, model_file, vectors_file):

	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(corpus_file)
	preprocessing.loadEmbeddings(vectors_file)
	
	# Establish params
	params_obj = Params()
	params_obj.num_classes = preprocessing.num_classes
	params_obj.vocab_size = len(preprocessing.word_index) 
	params_obj.inp_length = MAX_SEQUENCE_LENGTH
	params_obj.embeddings_dim = EMBEDDING_DIM

	# create and get model
	cnn_model = models.CNNModel()
	model = cnn_model.getModel(params_obj=params_obj, weight=preprocessing.embedding_matrix)
		
	# train model
	x_train, y_train, x_val, y_val = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val
	model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=params_obj.num_epochs, batch_size=params_obj.batch_size)

	# save model
	model.save(model_file)
	
def predict(text_file, model_file, vectors_file):

	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(text_file)
	preprocessing.loadEmbeddings(vectors_file)
	
	# load and predict
	x_data = np.concatenate((preprocessing.x_train,preprocessing.x_val), axis=0)
	model = load_model(model_file)
	predictions = model.predict(x_data)

	# print(model.layers)
	# input = model.layers[i].get_output_at(0) => sortie d'un layer
	# model2 = CONV2D(input)
	# deconv_layer = CONV2D_TRANSPOSE
	# model2.add(deconv_layer)
	# model2.compile
	# mettre à jour les poids

	print(predictions)

	return predictions


