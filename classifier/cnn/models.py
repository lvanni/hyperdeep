import numpy as np

from keras import optimizers
from keras import backend as K
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.layers import RepeatVector
from keras.layers import Permute
from keras.layers import Lambda
from keras.layers import Conv1D, Conv2D, Conv2DTranspose, MaxPooling1D, MaxPooling2D, GlobalMaxPooling1D,  Embedding, Reshape
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.legacy.layers import Merge
from keras.utils import np_utils

from config import EMBEDDING_DIM, NB_FILTERS, FILTER_SIZES, DROPOUT_VAL

class CNNModel:
	
	def getModel(self, params_obj, weight=None):

		print("-"*20)
		
		inputs = Input(shape=(params_obj.inp_length,), dtype='int32')
		#embedding = Embedding(output_dim=params_obj.embeddings_dim, input_dim=params_obj.vocab_size, input_length=params_obj.inp_length)(inputs)
		
		embedding = Embedding(
			params_obj.vocab_size+1, # due to mask_zero
			params_obj.embeddings_dim,
			input_length=params_obj.inp_length,
			weights=[weight],
			trainable=True
		)(inputs)

		print("embedding : ", embedding.shape)
		
		reshape = Reshape((params_obj.inp_length,params_obj.embeddings_dim,1))(embedding)

		print("reshape : ", reshape.shape)

		# CONVOLUTION
		"""
		conv_array = []
		maxpool_array = []
		for filter in FILTER_SIZES:
			conv = Conv2D(NB_FILTERS, filter, EMBEDDING_DIM, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)	
			maxpool = MaxPooling2D(pool_size=(params_obj.inp_length - filter + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv)
			conv_array.append(conv)
			maxpool_array.append(maxpool)			
						
		deconv = Conv2DTranspose(1, FILTER_SIZES[0], EMBEDDING_DIM, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(conv_array[0])
		deconv_model = Model(input=inputs, output=deconv)

		if len(FILTER_SIZES) >= 2:
			merged_tensor = merge(maxpool_array, mode='concat', concat_axis=1)
			flatten = Flatten()(merged_tensor)
		else:
			flatten = Flatten()(maxpool_array[0])
		"""

		filter = 3
		conv = Conv2D(NB_FILTERS, filter, EMBEDDING_DIM, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)	
		print("convolution :", conv.shape)

		#maxpool = MaxPooling2D(pool_size=(params_obj.inp_length - filter + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv)
		#print("maxpool :", maxpool.shape)

		deconv = Conv2DTranspose(1, filter, EMBEDDING_DIM, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(conv)
		print("deconvolution :", deconv.shape)
		deconv_model = Model(input=inputs, output=deconv)

		#flatten = Flatten()(maxpool)

		conv_shape = conv.shape[1:]
		reshape = Reshape((int(conv_shape[0]),int(np.prod(conv_shape[1:]))))(conv)

		print("reshape :", reshape.shape)

		# LSTM
		lstm = LSTM(100, return_sequences=True)(reshape)

		print("lstm :", lstm.shape)
		#print("-"*20)

		# Attention layer
		attention = TimeDistributed(Dense(1, activation='tanh'))(lstm) 
		print("TimeDistributed :", attention.shape)

		# reshape Attention
		attention = Flatten()(attention)
		print("Flatten :", attention.shape)
		attention = Activation('softmax')(attention)
		print("Activation :", attention.shape)

		# Observe attention here
		attention_model = Model(input=inputs, output=attention)

		attention = RepeatVector(100)(attention)
		print("RepeatVector :", attention.shape)
		attention = Permute([2, 1])(attention)
		print("Permute :", attention.shape)

		# apply the attention
		sent_representation = merge([lstm, attention], mode='mul')
		print("merge :", sent_representation.shape)
		sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
		print("Lambda :", sent_representation.shape)

		dropout = Dropout(DROPOUT_VAL)(sent_representation)
		
		hidden_dense = Dense(params_obj.dense_layer_size,kernel_initializer='uniform',activation='relu')(dropout)
		output = Dense(params_obj.num_classes, activation='softmax')(hidden_dense)

		# this creates a model that includes
		model = Model(input=inputs, output=output)

		op = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		#op = optimizers.Adam(lr=1e-3)
		model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])

		return model, deconv_model, attention_model
	