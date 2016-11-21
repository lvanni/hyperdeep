import theano.tensor as T
from blocks.bricks import (Initializable, Feedforward, Sequence, Activation,
                            MLP, Identity, Rectifier, Tanh)
from blocks.initialization import IsotropicGaussian, Constant

import numpy
from six import add_metaclass
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from toolz import interleave
from picklable_itertools.extras import equizip

from blocks.config import config
from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.bricks.wrappers import WithExtraDims
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.utils import pack, shared_floatx_nans
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.bricks.conv import Convolutional

from unsupervised import LookUpTrain
from unsupervised import LookUpTable
from util import build_dictionnary, get_input_from_files
from contextlib import closing
import os
import pickle
import theano

def flat_submatrix(M,col,dwin):
	"""
		this is the operator define by Collobert : concatenate win cols of M centered around col. see <M>_col^win in COllobert
	"""
	return T.flatten(M[col:col+dwin, :])	

def build_labelled_data(dwin, repo, filenames, labels, embeddings_filename):
	# build the network
	dico, _ = build_dictionnary(repo, filenames)
	dwin = 9
	paddings = [ [], [], [], []]
	values = []
	data=[]
	index=0
	for filename, label in zip(filenames, labels):
		input_sentences = get_input_from_files(repo, [filename], dico, paddings)
		for line in input_sentences:
			np_line = numpy.zeros((4, len(line[0])))
			np_line[0] = line[0]
			np_line[1] = line[1]
			np_line[2] = line[2]
			np_line[3] = line[3]
			np_line = np_line.astype(int)
			values.append(line)
			data.append(label)
	"""
	with closing(open(os.path.join(repo, embeddings_filename), 'rb')) as f:
		values = pickle.load(f)
	"""
	print 'kikou'
	with closing(open(os.path.join(repo, embeddings_filename+"_labelled"), 'wb')) as f:
		pickle.dump([values, data], f, protocol=pickle.HIGHEST_PROTOCOL)
		

class ConvPoolNlp(Initializable, Feedforward):
    """
        This is layer make a convolution and a subsampling on an input sentence
    """
    @lazy(allocation=['n_out', 'dwin', 'vector_size', 'n_hidden_layer'])
    def __init__(self, n_out, dwin, vector_size, n_hidden_layer, **kwargs):
        super(ConvPoolNlp, self).__init__(**kwargs)
	self.vector_size = vector_size
	self.n_hidden_layer=n_hidden_layer
	self.dwin = dwin
	self.n_out=n_out

	self.rectifier = Rectifier()
	"""
	self.convolution = Convolutional(filter_size=(1,self.filter_size),num_filters=self.num_filter,num_channels=1,
					weights_init=IsotropicGaussian(0.01), use_bias=False)
	"""
	# second dimension is of fixed size sum(vect_size) less the fiter_size borders
	self.mlp = MLP(activations=[Rectifier()]*len(self.n_hidden_layer) +[Identity()],
    		dims=[self.n_out]+self.n_hidden_layer+[2],
    		weights_init=IsotropicGaussian(0.01), biases_init=Constant(0.))

        self.parameters = []
	self.children=[]
	#self.children.append(self.lookup)
	#self.children.append(self.convolution)
	self.children.append(self.mlp)
	self.children.append(self.rectifier)

    def _allocate(self):
	W = shared_floatx_nans((self.n_out, self.dwin*self.vector_size), name='W')
    	b = shared_floatx_nans((self.n_out,), name='b')
    	add_role(b, BIAS)
	add_role(W, WEIGHT)
	self.parameters.append(W)
    	self.parameters.append(b)
	self.mlp.allocate()

    @property
    def b(self):
        return self.parameters[0]

    @property
    def b(self):
        return self.parameters[0]

    def _initialize(self):
        #self.allocate()
	#import pdb
	#pdb.set_trace()
    	W, b = self.parameters
	self.weights_init.initialize(W, self.rng)
    	self.biases_init.initialize(b, self.rng)
	#self.convolution.initialize()
	self.mlp.initialize()
	#self.lookup.initialize()
	

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
	W,b = self.parameters
	#input_ = self.lookup.embedding(input_)
	#input_ = input_.dimshuffle(('x', 0, 1, 2))
	convolved_inputs, _ = theano.scan(fn=lambda i,A,W,b: T.dot(W, flat_submatrix(input_, i, self.dwin))+b,
				sequences=T.arange(input_.shape[0] -self.dwin),
				non_sequences=[input_, W,b])
	output = T.concatenate([convolved_inputs])
	#output = self.rectifier.apply(output)
	output = T.max(output, axis=0) 
	output = output.dimshuffle(('x',0))
	return self.mlp.apply(output)

    def _push_allocation_config(self):
        #self.convolution._push_allocation_config()
	self.mlp._push_allocation_config()

if __name__=='__main__':
	dwin=9
	repo='data/dico'
	filenames = ['HollandeDef.cnr', 'Sarkozy-Inter-DEF.cnr']
	labels = [1,0]
	embeddings_filename = 'discours_SarkozyHollande'
	build_labelled_data(dwin, repo, filenames, labels, embeddings_filename)
