import random
import numpy as np

from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D

from classifier.cnn import models
from skipgram.skipgram_with_NS import create_vectors, create_tg_vectors
from data_helpers import tokenize

class PreProcessing:

	def loadData(self, corpus_file, model_file, config, create_dictionnary):   
		
		print("loading data...")
		
		self.corpus_file = corpus_file
		
		label_dic = {}
		labels = []
		texts = []
		
		# Read text and detect classes/labels
		self.num_classes = 0
		f = open(corpus_file, "r")
		for text in f.readlines():
			label = text.split("__ ")[0].replace("__", "")
			text = text.replace("__" + label + "__ ", "")
			if label not in label_dic.keys():
				label_dic[label] = self.num_classes
				self.num_classes += 1
			label_int = label_dic[label]
			labels += [label_int]
			texts += [text]
		f.close()
		
		print("DETECTED LABELS :")
		#print(label_dic)

		#data = list(zip(labels, texts))
		#random.shuffle(data)
		#labels, texts = zip(*data)

		my_dictionary, data = tokenize(texts, model_file, create_dictionnary, config)

		print('Found %s unique tokens.' % len(my_dictionary["word_index"]))

		labels = np_utils.to_categorical(np.asarray(labels))
		
		print('Shape of data tensor:', data.shape)
		print('Shape of label tensor:', labels.shape)

		# split the data into a training set and a validation set
		indices = np.arange(data.shape[0])
		np.random.shuffle(indices)
		data = data[indices]
		labels = labels[indices]
		nb_validation_samples = int(config["VALIDATION_SPLIT"] * data.shape[0])

		self.x_train = data[:-nb_validation_samples]
		self.y_train = labels[:-nb_validation_samples]
		self.x_val = data[-nb_validation_samples:]
		self.y_val = labels[-nb_validation_samples:]
		self.my_dictionary = my_dictionary

	def loadEmbeddings(self, model_file, config, vectors_file = False):
		
		my_dictionary = self.my_dictionary["word_index"]
		embeddings_index = {}

		if not vectors_file:
			if config["TG"]:
				vectors = create_tg_vectors(self.corpus_file, model_file + ".vec", config)
			else:
				vectors = create_vectors(self.corpus_file, model_file + ".vec", config)
		else:
			f = open(vectors_file, "r")
			vectors = f.readlines()
			f.close()
			
		i=0
		for line in vectors:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
			i+=1
			if i>10000:
				break

		print('Found %s word vectors.' % len(embeddings_index))
		embedding_matrix = np.zeros((len(my_dictionary) + 1, config["EMBEDDING_DIM"]))
		for word, i in my_dictionary.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				embedding_matrix[i] = embedding_vector

		self.embedding_matrix = embedding_matrix

def train(corpus_file, model_file, config):

	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(corpus_file, model_file, config, create_dictionnary = True)
	preprocessing.loadEmbeddings(model_file, config)
	
	# Establish params
	config["num_classes"] = preprocessing.num_classes 
	config["vocab_size"] = len(preprocessing.my_dictionary["word_index"]) 

	# create and get model
	cnn_model = models.CNNModel()
	model, deconv_model, attention_model = cnn_model.getModel(config=config, weight=preprocessing.embedding_matrix)

	# train model
	x_train, y_train, x_val, y_val = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val
	checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config["NUM_EPOCHS"], batch_size=config["BACH_SIZE"], callbacks=callbacks_list)

	# save deconv model
	try:
		i = 0
		for layer in model.layers:	
			weights = layer.get_weights()
			deconv_model.layers[i].set_weights(weights)
			i += 1
			if type(layer) is Conv2D:
				break
	except:
		print("WARNING: not convolution in this model!")
	deconv_model.save(model_file + ".deconv")

	# save attention model
	attention_model.save(model_file + ".attention")
	
def predict(text_file, model_file, config, vectors_file):

	result = []

	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(text_file, model_file, config, create_dictionnary = False)
	preprocessing.loadEmbeddings(model_file, config, vectors_file)
	
	# load and predict
	x_data = np.concatenate((preprocessing.x_train,preprocessing.x_val), axis=0)
	model = load_model(model_file)
	predictions = model.predict(x_data)
	
	print(predictions)	

	print("----------------------------")
	print("DECONVOLUTION")
	print("----------------------------")

	# load deconv_model
	deconv_model = load_model(model_file + ".deconv")	
	
	# update weights (TODO: should be after the train)
	for layer in deconv_model.layers:	
		if type(layer) is Conv2D:
			deconv_weights = layer.get_weights()[0]
	deconv_bias = deconv_model.layers[-1].get_weights()[1]
	deconv_model.layers[-1].set_weights([deconv_weights, deconv_bias])
	
	# apply deconvolution
	deconv = deconv_model.predict(x_data)
	print("deconvolution", 	deconv.shape)

	print("----------------------------")
	print("ATTENTION")
	print("----------------------------")

	# load deconv_model
	attention_model = load_model(model_file + ".attention")	

	# apply deconvolution
	attentions = attention_model.predict(x_data)

	print("attentions", attentions.shape)	
	#print(attentions)

	# Format result (prediction + deconvolution)
	my_dictionary = preprocessing.my_dictionary
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
			deconv_value = deconv[sentence_nb][i]
			
			if i == 0 or i == len(x_data[sentence_nb])-1: # because shape (?,48,1)
				attention_value = 0
			else:
				attention_value = attentions[sentence_nb][i-1]
			if "**" in word:
				j = int(config["EMBEDDING_DIM"]/3)
				word_args = word.split("**")
				# deconvolution word
				word = word_args[0] + "*" + str(float(np.sum(deconv_value[:j])))
				# deconvolution code
				word += "**" + word_args[1] + "*" + str(float(np.sum(deconv_value[j:j+j])))
				# deconvolution lemme
				word += "**" + word_args[2] + "*" + str(float(np.sum(deconv_value[-j:])))
				# attention
				word += "*" + str(float(attention_value))
			else:
				# deconvolution
				word = word + "*" + str(float(np.sum(deconv_value)))
				# attention
				word += "*" + str(float(attention_value))
			sentence["sentence"] += word + " "
		result.append(sentence)

	return result


