from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

import models
import numpy as np

data_src = ["../../data/test/rt-polarity.pos","../../data/test/rt-polarity.neg"] 
embeddings_src = "../../bin/rt-polarity.vec"

MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
VALIDATION_SPLIT = 0.1

####################################
class Params:
	inp_len = None
	vocab_size = None
	num_classes= None
	
	use_pretrained_embeddings = True
	embeddings_dim = 300
	train_embedding = True
	filter_sizes = [3,4,5]
	filter_numbers = [100,100,100]
	filter_pool_lengths = [2,2,2]
	num_epochs = 2
	batch_size = 50
	dropout_val = 0.5
	dense_layer_size = 100
	num_output_classes = 2
	use_two_channels = True
	
	def setParams(self, dct):
		if 'vocab_size' in dct:
			self.vocab_size=dct['vocab_size']
		if 'embeddings_size' in dct:
			self.embeddings_size=dct['embeddings_size']
		if 'inp_len' in dct:
			self.inp_len=dct['inp_len']
		if 'train_embedding' in dct:
			self.train_embedding=dct['train_embedding']
		if 'filter_sizes' in dct:
			self.filter_sizes=dct['filter_sizes']
		if 'filter_numbers' in dct:
			self.filter_numbers=dct['filter_numbers']			
		if 'filter_pool_lengths' in dct:
			self.filter_pool_lengths=dct['filter_pool_lengths']

class PreProcessing:
		
	def loadData(self):   
		print("loading data...")
		text_pos = open(data_src[0],"r").readlines()
		text_neg = open(data_src[1],"r").readlines()
		labels_pos = [1]*len(text_pos)
		labels_neg = [0] * len(text_neg)
		texts = text_pos
		texts.extend(text_neg)
		labels = labels_pos
		labels.extend(labels_neg)
		
		tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
		tokenizer.fit_on_texts(texts)
		sequences = tokenizer.texts_to_sequences(texts)

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

	def loadEmbeddings(self):
		word_index = self.word_index
		embeddings_index = {}
		f = open(embeddings_src)
		i=0
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
			EMBEDDING_DIM = len(coefs)
			i+=1
			if i>10000:
				break
		f.close()

		print('Found %s word vectors.' % len(embeddings_index))
		embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
		for word, i in word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				embedding_matrix[i] = embedding_vector
		self.embedding_matrix = embedding_matrix
		self.EMBEDDING_DIM = EMBEDDING_DIM
		self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH

def main():
	preprocessing = PreProcessing()
	preprocessing.loadData()
	preprocessing.loadEmbeddings()
	
	cnn_model = models.CNNModel()
	params_obj = Params()
	
	# Establish params
	params_obj.num_classes=2
	params_obj.vocab_size = len(preprocessing.word_index)
	params_obj.inp_length = preprocessing.MAX_SEQUENCE_LENGTH
	params_obj.embeddings_dim = preprocessing.EMBEDDING_DIM
	
	# get model
	model = cnn_model.getModel(params_obj=params_obj, weight=preprocessing.embedding_matrix)
	
	x_train, y_train, x_val, y_val = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val
	
	# train
	model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=params_obj.num_epochs, batch_size=params_obj.batch_size)
	
	#evaluate

if __name__ == "__main__":
	main()
