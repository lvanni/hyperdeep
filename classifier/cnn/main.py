import os
import json
import pickle
import random
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D

from classifier.cnn import models
from skipgram.skipgram_with_NS import create_vectors, get_w2v

from config import LABEL_MARK, DENSE_LAYER_SIZE, FILTER_SIZES, DROPOUT_VAL, NUM_EPOCHS, BACH_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, VALIDATION_SPLIT, MAX_NB_WORDS, NB_FILTERS

####################################
class Params:

	# Initalize defaut parameters
	dense_layer_size = DENSE_LAYER_SIZE
	filter_sizes = FILTER_SIZES
	dropout_val = DROPOUT_VAL
	num_epochs = NUM_EPOCHS
	batch_size = BACH_SIZE
	inp_length = MAX_SEQUENCE_LENGTH
	embeddings_dim = EMBEDDING_DIM

class PreProcessing:
	
	def tokenize(self, texts, model_file, create_dictionnary):

		if create_dictionnary:
			my_dictionary = {}
			my_dictionary["word_index"] = {}
			my_dictionary["index_word"] = {}
			my_dictionary["word_index"]["<PAD>"] = 0
			my_dictionary["index_word"][0] = "<PAD>"
			index = 0
		else:
			with open(model_file + ".index", 'rb') as handle:
				my_dictionary = pickle.load(handle)

		data = (np.zeros((len(texts), MAX_SEQUENCE_LENGTH))).astype('int32')
		
		i = 0
		for line in texts:
			words = line.split()[:MAX_SEQUENCE_LENGTH]
			sentence_length = len(words)
			sentence = []
			for word in words:
				if word not in my_dictionary["word_index"].keys():
					if create_dictionnary:
						index += 1
						my_dictionary["word_index"][word] = index
						my_dictionary["index_word"][index] = word
					else:
						my_dictionary["word_index"][word] = my_dictionary["word_index"]["<PAD>"]
				sentence.append(my_dictionary["word_index"][word])
			if sentence_length < MAX_SEQUENCE_LENGTH:
				for j in range(MAX_SEQUENCE_LENGTH - sentence_length):
					sentence.append(my_dictionary["word_index"]["<PAD>"])
			
			data[i] = sentence
			i += 1

		if create_dictionnary:
			with open(model_file + ".index", 'wb') as handle:
				pickle.dump(my_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

		return my_dictionary, data

	def loadData(self, corpus_file, model_file, create_dictionnary):   
		
		print("loading data...")
		
		self.corpus_file = corpus_file
		
		label_dic = {}
		labels = []
		texts = []
		
		# Read text and detect classes/labels
		self.num_classes = 0
		for text in open(corpus_file, "r").readlines():
			label = text.split(LABEL_MARK + " ")[0].replace(LABEL_MARK, "")
			text = text.replace(LABEL_MARK + label + LABEL_MARK + " ", "")
			if label not in label_dic.keys():
				label_dic[label] = self.num_classes
				self.num_classes += 1
			label_int = label_dic[label]
			labels += [label_int]
			texts += [text]
		
		data = list(zip(labels, texts))
		random.shuffle(data)
		labels, texts = zip(*data)
		
		"""
		tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
		tokenizer.fit_on_texts(texts)
		sequences = tokenizer.texts_to_sequences(texts)
		data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
		my_dictionary = tokenizer.my_dictionary
		"""

		my_dictionary, data = self.tokenize(texts, model_file, create_dictionnary)

		print('Found %s unique tokens.' % len(my_dictionary["word_index"]))

		labels = np_utils.to_categorical(np.asarray(labels))
		
		print('Shape of data tensor:', data.shape)
		print('Shape of label tensor:', labels.shape)

		# split the data into a training set and a validation set
		indices = np.arange(data.shape[0])
		np.random.shuffle(indices)
		data = data[indices]
		labels = labels[indices]
		nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
		
		print(data)

		self.x_train = data[:-nb_validation_samples]
		self.y_train = labels[:-nb_validation_samples]
		self.x_val = data[-nb_validation_samples:]
		self.y_val = labels[-nb_validation_samples:]
		self.my_dictionary = my_dictionary

	def loadEmbeddings(self, vectors_file):
		
		#print(vectors_file)
		#print(embeddings_src)

		my_dictionary = self.my_dictionary["word_index"]
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
		embedding_matrix = np.zeros((len(my_dictionary) + 1, EMBEDDING_DIM))
		for word, i in my_dictionary.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				embedding_matrix[i] = embedding_vector

		self.embedding_matrix = embedding_matrix

def train(corpus_file, model_file, vectors_file):

	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(corpus_file, model_file, create_dictionnary = True)
	preprocessing.loadEmbeddings(vectors_file)
	
	# Establish params
	params_obj = Params()
	params_obj.num_classes = preprocessing.num_classes
	params_obj.vocab_size = len(preprocessing.my_dictionary["word_index"]) 
	params_obj.inp_length = MAX_SEQUENCE_LENGTH
	params_obj.embeddings_dim = EMBEDDING_DIM

	# create and get model
	cnn_model = models.CNNModel()
	model, deconv_model = cnn_model.getModel(params_obj=params_obj, weight=preprocessing.embedding_matrix)
		
	# train model
	x_train, y_train, x_val, y_val = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val
	checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=params_obj.num_epochs, batch_size=params_obj.batch_size, callbacks=callbacks_list)

	# save deconv model
	i = 0
	for layer in model.layers:	
		weights = layer.get_weights()
		deconv_model.layers[i].set_weights(weights)
		i += 1
		if type(layer) is Conv2D:
			break
	deconv_model.save("bin/deconv_model.test")
	
def predict(text_file, model_file, vectors_file):

	result = []

	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(text_file, model_file, create_dictionnary = False)
	preprocessing.loadEmbeddings(vectors_file)
	
	# load and predict
	x_data = np.concatenate((preprocessing.x_train,preprocessing.x_val), axis=0)
	model = load_model(model_file)
	predictions = model.predict(x_data)
	print(predictions)	

	print("----------------------------")
	print("DECONVOLUTION")
	print("----------------------------")

	deconv_model = load_model("bin/deconv_model.test")
	deconv = deconv_model.predict(x_data)
	print("DECONVOLUTION SHAPE : ", deconv.shape)

	my_dictionary = preprocessing.my_dictionary
	#w2v = get_w2v(vectors_file)

	for sentence_nb in range(len(x_data)):
		sentence = {}
		sentence["sentence"] = ""
		sentence["prediction"] = predictions[sentence_nb].tolist()
		for i in range(len(x_data[sentence_nb])):
			word = ""
			index = x_data[sentence_nb][i]
			try:
				word = my_dictionary["index_word"][index]
			except:
				word = "PAD"

			# READ DECONVOLUTION 
			deconv_value = float(np.sum(deconv[sentence_nb][i]))
			sentence["sentence"] += word + ":" + str(deconv_value) + " "
		result.append(sentence)

	print("----------------------------")
	

	return result


