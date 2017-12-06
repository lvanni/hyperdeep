import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.legacy.layers import Merge

class CNNModel:
	
	def getModel(self, params_obj, weight=None  ):
		
		print("Use Two channels - static and non-static")
		inp = Input(shape=(params_obj.inp_length,), dtype='int32')

		embeddings_layer = Embedding(
			params_obj.vocab_size+1, # due to mask_zero
			params_obj.embeddings_dim,
			input_length=params_obj.inp_length,
			weights=[weight],
			trainable=False
		)(inp)
		
		embeddings_layer_t = Embedding(
			params_obj.vocab_size+1, # due to mask_zero
			params_obj.embeddings_dim,
			input_length=params_obj.inp_length,
			weights=[weight],
			trainable=True
		)(inp)
		
		#Convolution
		convolution_features_list = []
		for filter_size,pool_length,num_filters in zip(params_obj.filter_sizes, params_obj.filter_pool_lengths, params_obj.filter_sizes):
			conv_layer = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(embeddings_layer)
			pool_layer = MaxPooling1D(pool_size=pool_length)(conv_layer)
			flatten = Flatten()(pool_layer)
			convolution_features_list.append(flatten)
			
		for filter_size,pool_length,num_filters in zip(params_obj.filter_sizes, params_obj.filter_pool_lengths, params_obj.filter_sizes):
			conv_layer = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(embeddings_layer_t)
			pool_layer = MaxPooling1D(pool_size=pool_length)(conv_layer)
			flatten = Flatten()(pool_layer)
			convolution_features_list.append(flatten)
		out1 = Merge(mode='concat')(convolution_features_list)
		network = Model(input=inp, output=out1)
	
		# Model	
		model = Sequential()
		model.add(network)

		#Add dense layer to complete the model
		model.add(Dense(params_obj.dense_layer_size,kernel_initializer='uniform',activation='relu'))
		model.add(Dropout(params_obj.dropout_val))
		model.add( Dense(params_obj.num_classes, kernel_initializer='uniform', activation='softmax')  )
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])	
		
		return model
	
		
