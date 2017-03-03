#!/usr/bin/python
# -*-coding:Utf-8 -*

# blocks class 
import os
from contextlib import closing
import logging
import pickle

from blocks.bricks import Initializable, Feedforward, MLP, Tanh, Softmax, Identity
from blocks.bricks.base import application, lazy
from blocks.bricks.cost import MisclassificationRate
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.utils import shared_floatx_nans
import theano

import theano.tensor as T


logger = logging.getLogger(__name__)

def getParams(model, tensor):
    x = T.tensor4()
    cost = model.apply(tensor).sum()
    cg = ComputationGraph(cost)
    W = VariableFilter(roles=[WEIGHT])(cg.variables)
    B = VariableFilter(roles=[BIAS])(cg.variables)
    return W+B

class LookUpTable(Initializable, Feedforward):

    @lazy(allocation=['vect_size', 'n_mot'])
    def __init__(self, vect_size, n_mot, **kwargs):
        super(LookUpTable, self).__init__(**kwargs)
        self.vect_size = vect_size
        self.n_mot = n_mot
        self.parameters = []

    def _allocate(self): # Definition word representation / Random (ex: Word 2 vec)
        W = shared_floatx_nans((self.vect_size, self.n_mot), name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)

    @property
    def W(self):
        return self.parameters[0]

    def _initialize(self): # Initalize word representation / Random (ex: Word 2 vec) 
        self.allocate()
        W = self.parameters[0]
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_): # Appliquer le reseau/fonction à une phrase donnée => input_
        shape = input_.shape
        input_ = input_.flatten()
        W, = self.parameters
        result, _ = theano.scan(fn=lambda x_i,W : W[:,x_i], sequences=input_, non_sequences=W)
        output = T.concatenate([result,])
        output = output.reshape((shape[0], self.vect_size, shape[1])) #(batch_size, vector_size, nb_words)
        return output

    def get_dim(self, name): 
        if name == 'input_':
            return self.n_mot
        if name == 'output':
            return self.vect_size
        super(LookUpTable, self).get_dim(name)

"""
Appel LookUpTable pour chaque niveau (forme/code/lemme/???) pour contruire une matrice
+ MLP : Multi Layer Perceptron 
"""
class Window(Initializable, Feedforward):

    @lazy(allocation=['dwin', 'n_mot', 'vect_size', 'n_hidden'])
    def __init__(self, dwin, n_mot, vect_size, n_hidden, n_out=2, **kwargs):
        super(Window, self).__init__(**kwargs)
        self.dwin = dwin
        self.n_mot=n_mot
        self.vect_size = vect_size
        self.n_hidden = n_hidden
        self.n_out = n_out # Nombre de classes en sortie
        self.n_tables = len(self.vect_size)
        self.tables = [LookUpTable(self.vect_size[i], self.n_mot[i], weights_init=IsotropicGaussian(0.001), use_bias=False) for i in range(self.n_tables)]
        self.mlp = MLP(activations=[Tanh()]*len(self.n_hidden)+[Identity()], dims=[self.dwin*sum(self.vect_size)]+ self.n_hidden + [self.n_out], weights_init=IsotropicGaussian(0.001), biases_init=Constant(0.))
        self.parameters = []
        self.children = self.tables + [self.mlp]

    def _initialize(self):
        for i in range(self.n_tables):
            self.tables[i].initialize()
        self.mlp.initialize()
        W = self.parameters[0]
        self.weights_init.initialize(W, self.rng)

    def _allocate(self):
        for i in range(self.n_tables):
            self.tables[i].allocate()
        self.mlp.allocate()
        W = shared_floatx_nans((sum(self.n_mot), sum(self.vect_size)), name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)

    def update_transition_matrix(self): # Transition => des lookUpTable vers la matrice resultat
        W_tmp = self.parameters[0]
        params_lookup = [getParams(table, T.itensor3()) for table in self.tables]
        index_row=0; index_col=0;
        for i in range(len(self.tables)):
            W_tmp_value = W_tmp.get_value()
            p_value = params_lookup[i][0].get_value()
            W_tmp_value[index_row: index_row+ p_value.shape[1], index_col: index_col+p_value.shape[0]] = p_value.transpose()
            index_row+=p_value.shape[1]
            index_col+=p_value.shape[0]
            W_tmp.set_value(W_tmp_value)

    def update_lookup_weights(self):
        W_tmp = self.parameters[0]
        params_lookup = [getParams(table, T.itensor3()) for table in self.tables]
        index_row=0; index_col=0;
        for i in range(len(self.tables)):
            W_tmp_value = W_tmp.get_value().transpose()
            p_value = params_lookup[i][0].get_value()
            params_lookup[i][0].set_value(W_tmp_value[index_col: index_col+p_value.shape[0],index_row: index_row+ p_value.shape[1]])
            index_row+=p_value.shape[1]
            index_col+=p_value.shape[0]

    def get_Params(self):
        params = getParams(self.mlp, T.matrix())
        self.update_transition_matrix()
        weights=[]
        biases=[]
        for p in params:
            if p.ndim==1:
                biases.append(p)
            else:
                weights.append(p)
            if len(params[0].name) == 1:
                # words has not been renamed yet
                if weights[0].shape[-1].eval() == self.n_out:
                    weights.reverse(); biases.reverse()
                # add the lookuptables weights
                weights = [self.parameters[0]] + weights
                assert len(weights)==len(biases)+1
                for w, index in zip(weights, range(len(weights))):
                    w.name = "layer_"+str(index)+"_"+w.name
                for b, index in zip(biases, range(len(biases))):
                    b.name = "layer_"+str(index+len(weights)-len(biases))+"_"+b.name
            else:
                weights = [self.parameters[0]] + weights
        return weights, biases

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        outputs = [self.tables[i].apply(input_[:,i]) for i in xrange(self.n_tables)] # (batch_size, vector_size[i], dwin)
        outputs = [output.dimshuffle((1, 0, 2)) for output in outputs]
        output = T.concatenate(outputs, axis=0) # (sum vector_size, batch_size, dwin)
        output = output.dimshuffle((1, 0, 2))
        shape = output.shape
        output = output.reshape((shape[0], shape[1]*shape[2]))
        return self.mlp.apply(output)


    @application(inputs=['input_'], outputs=['output'])
    def embedding(self, input_):
        input_ = input_.dimshuffle(('x',0, 1))
        outputs = [self.tables[i].apply(input_[:,i]) for i in xrange(self.n_tables)] # (batch_size, vector_size[i], nb_words)
        outputs = [output.dimshuffle((1, 0, 2)) for output in outputs]
        output = T.concatenate(outputs, axis=0) # (sum vector_size, batch_size, nb_words)
        return output.dimshuffle((1, 2, 0))

    def _push_allocation_config(self):
        for i in range(self.n_tables):
            self.tables[i]._push_allocation_config()
            self.mlp._push_allocation_config()

"""
Couche au dessus de Windows (Ajout de diverses fonctions)
"""
class LookUpTrain(Initializable, Feedforward):

    @lazy(allocation=['dwin', 'n_mot', 'vect_size', 'n_hidden'])
    def __init__(self, dwin, n_mot, vect_size, n_hidden, n_out=2, **kwargs):
        self.dwin = dwin
        self.n_mot=n_mot
        self.vect_size = vect_size
        if isinstance(n_hidden, int):
            self.n_hidden = [n_hidden]
        else:
            self.n_hidden = n_hidden
        self.n_out = n_out
        self.window = Window(self.dwin, self.n_mot, self.vect_size, self.n_hidden, self.n_out, weights_init=IsotropicGaussian(0.001))
        super(LookUpTrain, self).__init__(**kwargs)
        self.softmax = Softmax()
        self.error = MisclassificationRate()
        self.children = [self.window, self.softmax, self.error]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return self.window.apply(input_)

    @application(inputs=['x', 'y'], outputs=['output'])
    def cost(self, x, y):
        return self.softmax.categorical_cross_entropy(y, self.apply(x))

    @application(inputs=['x', 'y'], outputs=['output'])
    def errors(self, x, y):
        return self.error.apply(y, self.apply(x))

    @application(inputs=['x'], outputs=['output'])
    def predict(self, x):
        return T.argmax(self.apply(x), axis=1)

    def probabilities_text(self, x):
        return T.mean(Softmax().apply(self.apply(x)), axis=0)

    @application(inputs=['x'], outputs=['output'])
    def predict_confidency(self, x):
        return T.max(self.apply(x), axis=1)

    def update_lookup_weights(self):
        self.window.update_lookup_weights()

    # PRE-Apprentissage ------
    @application(inputs=['input_', 'input_corrupt'], outputs=['output'])
    def score(self, input_, input_corrupt): # PRE-Apprentissage
        # modify the input_ with an incorrect central word ?
        return (1 - -self.apply(input_)).norm(2) + (self.apply(input_corrupt)).norm(2)
        #return T.maximum(0,1 - self.apply(input_)+self.apply(input_corrupt) )[0]
        #return T.maximum(0, 1 -self.apply(input_))[0] + 0.1*T.maximum(0, 1 +self.apply(input_corrupt))[0] + 0.1*T.maximum(0,1 - self.apply(input_)+self.apply(input_corrupt) )[0]
        # change that !!!!

    def _initialize(self):
        self.window.initialize()

    @application(inputs=['input_'], outputs=['output'])
    def embedding(self, input_):
        return self.window.embedding(input_)

    def _allocate(self):
        self.window.allocate()

    def load(self, filename):
        params = getParams(self, T.itensor3())
        with closing(open(filename, 'rb')) as f:
            params_value = pickle.load(f)
        for p, p_value in zip(params, params_value):
            p.set_value(p_value.get_value())

    def get_Params(self):
        return self.window.get_Params()

    def save(self, repo, filename):
        params = getParams(self, T.itensor3())
        with closing(open(os.path.join(repo, filename), 'wb')) as f:
            pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
