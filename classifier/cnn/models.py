import numpy as np

from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.utils import np_utils
from keras.layers import Conv1D, Conv2D, Conv2DTranspose, MaxPooling1D, MaxPooling2D, GlobalMaxPooling1D,  Embedding, Reshape
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.legacy.layers import Merge

from config import EMBEDDING_DIM, NB_FILTERS, FILTER_SIZES, DROPOUT_VAL

class CNNModel:
	
	def getModel(self, params_obj, weight=None):
		
		inputs = Input(shape=(params_obj.inp_length,), dtype='int32')
		#embedding = Embedding(output_dim=params_obj.embeddings_dim, input_dim=params_obj.vocab_size, input_length=params_obj.inp_length)(inputs)
		
		embedding = Embedding(
			params_obj.vocab_size+1, # due to mask_zero
			params_obj.embeddings_dim,
			input_length=params_obj.inp_length,
			weights=[weight],
			trainable=True
		)(inputs)
		
		reshape = Reshape((params_obj.inp_length,params_obj.embeddings_dim,1))(embedding)

		# CONVOLUTION
		conv_array = []
		maxpool_array = []
		for filter in FILTER_SIZES:
			conv = Conv2D(filters=NB_FILTERS, kernel_size=(filter, EMBEDDING_DIM), strides=(1, EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)	

			maxpool = MaxPooling2D(pool_size=(params_obj.inp_length - filter + 1, 1), padding='valid')(conv)
			conv_array.append(conv)
			maxpool_array.append(maxpool)			
						
		deconv = Conv2DTranspose(filters=1, kernel_size=(FILTER_SIZES[0], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(conv_array[0])
		deconv_model = Model(inputs=[inputs], outputs=[deconv])

		if len(FILTER_SIZES) >= 2:
			merged_tensor = merge(maxpool_array, mode='concat', concat_axis=1)
			flatten = Flatten()(merged_tensor)
		else:
			flatten = Flatten()(maxpool_array[0])
		
		dropout = Dropout(DROPOUT_VAL)(flatten)
		
		hidden_dense = Dense(params_obj.dense_layer_size,kernel_initializer='uniform',activation='relu')(dropout)
		output = Dense(params_obj.num_classes, activation='softmax')(hidden_dense)

		# this creates a model that includes
		model = Model(inputs=[inputs], outputs=[output])

		op = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		#op = optimizers.Adam(lr=1e-3)
		model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])

		return model, deconv_model
	
		
