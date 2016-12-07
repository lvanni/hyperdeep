########################################
# training with Collobert architecture #
########################################
import theano.tensor as T
import theano
import os
from contextlib import closing
import pickle
import numpy as np
from unsupervised import getParams
from blocks.utils import shared_floatx
from util import get_input_from_files, add_padding

def saving_data(repo, recorded_data):
	filename = "sentence_HS_v0_"
	index=0
	while os.path.isfile(os.path.join(repo, filename+str(index))):
		index+=1
	filename = filename+str(index)
	with closing(open(os.path.join(repo, filename), 'wb')) as f:
		pickle.dump(recorded_data, f, protocol=pickle.HIGHEST_PROTOCOL) 


def Adam(cost, params, lr=0.002, b1=0.1, b2=0.001, e=1e-8):
	decay_factor = 1-e
	updates=[]
	updates_reinit = []
	grads=T.grad(cost, params)
	i = shared_floatx(0.,"adam_t")
	i_t = i+1
	updates.append((i,i_t))
	updates_reinit.append((i, 0*i))
	lr = (lr *T.sqrt((1. - (1. - b2)**i_t)) /
		(1. - (1. - b1)**i_t))
	b1_t = 1 - (1 - b1) * decay_factor ** (i_t - 1)
	for p,g in zip(params, grads):
		m = shared_floatx(p.get_value() * 0., "adam_m_"+p.name)
		v = shared_floatx(p.get_value() *0., "adam_v_"+p.name)
		m_t = b1_t*g + (1-b1_t)*g
		v_t = b2*T.sqr(g) + (1-b2)*v
		g_t = m_t/(T.sqrt(v_t)+e)
		updates.append((m,m_t)); updates_reinit.append((m, 0*m))
		updates.append((v,v_t)); updates_reinit.append((v, 0*v))
		updates.append((p, p-lr*g_t))
	
	return updates, updates_reinit

def RMSProp(cost, params, learning_rate, decay_rate):
	updates =[]
	grad_params = T.grad(cost, params)
	for param, grad_param in zip(params, grad_params):
		caches = shared_floatx(param.get_value() * 0.,"cache_"+param.name)
		update_cache = decay_rate*caches\
					+ (1 - decay_rate)*grad_param**2
		update_param = param  - learning_rate*grad_param/T.sqrt(update_cache + 1e-8)
		updates.append((caches, update_cache))
		updates.append((param, update_param))
	return updates

def build_database(repo, dico_filename, filenames, dwin):
	index = 0
	y_value = []
	x_value = []
	with closing(open(os.path.join(repo, dico_filename), 'rb')) as f:
		dico = pickle.load(f)
	for filename in filenames:
		lines,_ = get_input_from_files(repo, [filename], dico)
		for line in lines:
			x_value.append(line)
			y_value.append(index)
		if index ==0:
			index+=1
	y_value = np.asarray(y_value, dtype=int)

	x_value_0 = [ x_value[i] for i in range(np.argmax(y_value))]
	y_value_0 = [ y_value[i] for i in range(np.argmax(y_value))]
	indexes = np.random.permutation(y_value.shape[0] - np.argmax(y_value))[:np.argmax(y_value)] #TODO PUT IT BACK
	x_value_1 = [x_value[i+np.argmax(y_value)] for i in indexes]# balance the numbers
	y_value_1 = [y_value[i+np.argmax(y_value)] for i in indexes]# balance the numbers
	
	pos_percentage = (int) (len(y_value_0)*0.8)
	neg_percentage = (int) (len(y_value_1)*0.8)
	other_pos_percentage = (len(y_value_0) - pos_percentage)/2
	other_neg_percentage = (len(y_value_1) - neg_percentage)/2

	pos_permut = np.random.permutation(len(y_value_0))
	neg_permut = np.random.permutation(len(y_value_1))
	x_train = [x_value_0[i] for i in pos_permut[:pos_percentage]] + [x_value_1[i] for i in neg_permut[:neg_percentage]]
	x_valid = [x_value_0[i] for i in pos_permut[pos_percentage:pos_percentage+other_pos_percentage]] + \
		  [x_value_1[i] for i in neg_permut[neg_percentage:neg_percentage+other_neg_percentage]]
	x_test = [x_value_0[i] for i in pos_permut[pos_percentage+other_pos_percentage:]] + \
		  [x_value_1[i] for i in neg_permut[neg_percentage+other_neg_percentage:]]

	y_train = [y_value_0[i] for i in pos_permut[:pos_percentage]] + [y_value_1[i] for i in neg_permut[:neg_percentage]]
	y_valid = [y_value_0[i] for i in pos_permut[pos_percentage:pos_percentage+other_pos_percentage]] + \
		  [y_value_1[i] for i in neg_permut[neg_percentage:neg_percentage+other_neg_percentage]]
	y_test = [y_value_0[i] for i in pos_permut[pos_percentage+other_pos_percentage:]] + \
		  [y_value_1[i] for i in neg_permut[neg_percentage+other_neg_percentage:]]

	index_train = np.random.permutation(len(y_train))
	index_valid = np.random.permutation(len(y_valid))
	index_test = np.random.permutation(len(y_test))
	x_train_ = [x_train[i].astype(int) for i in index_train]
	x_valid_ = [x_valid[i].astype(int) for i in index_valid]
	x_test_ = [x_test[i].astype(int) for i in index_test]
	y_train_ = [y_train[i] for i in index_train]
	y_valid_ = [y_valid[i] for i in index_valid]
	y_test_ = [y_test[i] for i in index_test]

	paddings = [ [], [], [], []]
	for i in range(dwin/2):
		for i in xrange(4):
			paddings[i].append(dico[i]['PARSING'])
	paddings = np.asarray(paddings)
	#paddings = paddings.reshape((1, paddings.shape[0], paddings.shape[1]))
	x_train_ = [add_padding(elem, paddings).astype(int) for elem in x_train_]
	x_valid_ = [add_padding(elem, paddings) for elem in x_valid_]
	x_test_ = [add_padding(elem, paddings) for elem in x_test_]

	x_train=[]; x_valid=[]; x_test=[]
	y_train=[]; y_valid=[]; y_test=[]
	for elem, label in zip(x_train_, y_train_):
		for i in range(elem.shape[1] -dwin):
			x_train.append(elem[:,i:i+dwin])
			y_train.append(label)
	for elem, label in zip(x_valid_, y_valid_):
		for i in range(elem.shape[1] -dwin):
			x_valid.append(elem[:,i:i+dwin])
			y_valid.append(label)
	for elem, label in zip(x_test_, y_test_):
		for i in range(elem.shape[1] -dwin):
			x_test.append(elem[:,i:i+dwin])
			y_test.append(label)

	index_train = np.random.permutation(len(y_train))
	index_valid = np.random.permutation(len(y_valid))
	index_test = np.random.permutation(len(y_test))
	x_train = [x_train[i].astype(int) for i in index_train]
	x_valid = [x_valid[i].astype(int) for i in index_valid]
	x_test = [x_test[i].astype(int) for i in index_test]
	y_train = [y_train[i] for i in index_train]
	y_valid = [y_valid[i] for i in index_valid]
	y_test = [y_test[i] for i in index_test]

	return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def training_committee_member(instance, learning_rate, decay_rate, train, valid, batch_size):
	x = T.itensor3('x')
	y = T.ivector('y')
	cost = T.mean(instance.cost(x,y))
	error = T.mean(instance.errors(x,y)) # just to pik the right hyperparameters
	params = getParams(instance, x)
	Params = []
	for p in params:
		if p.shape[-1].eval()==2:
			Params.append(p)
	#updates = RMSProp(cost, params, learning_rate, decay_rate)
	if len(Params)!=2:
		print 'PROBLEM !'
	updates,_ = Adam(cost, params, learning_rate) # *0.1 ???
	train_model = theano.function(inputs=[x,y], outputs=cost, updates=updates,
					allow_input_downcast=True)

	valid_model = theano.function(inputs=[x, y], outputs=cost, allow_input_downcast=True)
	test_model = theano.function(inputs=[x, y], outputs=error, allow_input_downcast=True)
	x_train, y_train = train
	x_valid, y_valid = valid
	n_train = len(y_train)/batch_size; n_valid = len(y_valid)/batch_size
	increment_rate=0.95
	initial_increment = 3
	increment = initial_increment
	best_valid = np.inf
	valid_cost = []
	#while(increment >0):
	for i in range(2):
		for minibatch_index in range(n_train):
			sentence = x_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
			y_value = y_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
			train_value = train_model(sentence, y_value)
		valid_cost = []
		for minibatch_index in range(n_valid):
			sentence = x_valid[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
			y_value = y_valid[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
			valid_value = test_model(sentence, y_value)
			valid_cost.append(valid_value)
		"""
		if best_valid*0.95 > np.mean(valid_cost):
			best_valid = np.mean(valid_cost)
			increment = initial_increment
		else:
			increment -=1
		"""
	print np.mean(valid_cost)*100
	print '####'
	return instance
	
def training_committee(committee, learning_rate, decay_rate, train, valid, batch_size):
	committee_new = []
	for instance in committee:
		new_instance = training_committee_member(instance, learning_rate, decay_rate, train, valid, batch_size)
		committee_new.append(new_instance)
	return committee_new

"""
def training_Hollande(repo, dico_filename, filenames, dwin, learning_rate, decay_rate):

	train, valid, test = build_database(repo, dico_filename, filenames, dwin)
	x_train, y_train = train
	x_valid, y_valid = valid
	x_test, y_test = test
	#########
	# MODEL #
	#########
	with closing(open(os.path.join(repo, output_dico), 'rb')) as f:
		dico = pickle.load(f)
	n_mot = [len(dico[i]) for i in dico.keys()]
	vect_size = [100, 10, 5, 5]
	n_hidden = [100, 50]

	t_nlp = Architecture(dwin, n_mot, vect_size, n_hidden, dropout=0.5, committee_size=5)
	t_nlp.initialize()
	t_nlp.model.get_Params()
	x = T.itensor3('x')
	y = T.ivector('y')
	cost = T.mean(t_nlp.model.cost(x,y))
	error = T.mean(t_nlp.model.errors(x,y))
	params = getParams(t_nlp.model, x)
	updates, updates_reinit = Adam(cost, params, learning_rate)

	train_model = theano.function(inputs=[x,y], outputs=cost, updates=updates,
					allow_input_downcast=True)

	valid_model = theano.function(inputs=[x, y], outputs=cost, allow_input_downcast=True)
	test_model = theano.function(inputs=[x, y], outputs=error, allow_input_downcast=True)
	reinit = theano.function(inputs=[], outputs=T.zeros((1,)), updates=updates_reinit)

	x_train, y_train = train
	x_valid, y_valid = valid
	print sum(y_valid)*1./len(y_valid)
	x_test, y_test = test
	batch_size = 32
	n_train = len(y_train)/batch_size; n_valid = len(y_valid)/batch_size; n_test = len(y_test)/batch_size
	n_train_batches = 350
	print n_train
	index=0

	# stop training when the validation is not improving
	best_valid = np.inf
        best_test = np.inf
	best_train = np.inf
	improvement_rate = 0.95
        active_phase = False
	increment_init = 3
	increment = increment_init
	nb_query = 20

	for epoch in range(100):
		
		for minibatch_index in range(n_train_batches):
			sentence = x_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
			y_value = y_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
			train_value = train_model(sentence, y_value)
		
		valid_cost = []
		for minibatch_valid in range(n_valid):
			sentence = x_valid[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size]
			y_value = y_valid[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size]
			valid_value = test_model(sentence, y_value)
			valid_cost.append(valid_value)
		valid_score = np.mean(valid_cost)

		if best_valid*0.95 > valid_score:
			print 'kikou'
			increment= increment_init
			best_valid = valid_score
			# compute training cost and jadt_2016
			train_cost = []
			for minibatch_valid in range(n_train_batches):
				sentence = x_train[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size]
				y_value = y_train[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size]
				train_value = valid_model(sentence, y_value)
				train_cost.append(train_value)
			best_train = np.mean(train_cost)
			test_cost = []
			for minibatch_valid in range(n_test):
				sentence = x_test[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size]
				y_value = y_test[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size]
				test_value = test_model(sentence, y_value)
				test_cost.append(test_value)
			best_test = np.mean(test_cost)	
		else:
			increment-=1
		if increment ==0:
			active_phase = True
		#print "TRAIN : "+str(best_train); print "VALID : "+str(valid_score); print "TEST : "+str(best_test)+" ,"+str(increment)
		
		active_phase = True
		if active_phase :
			print 'COMMITTEE TRAINING'
			print "TRAIN : "+str(best_train); print "VALID : "+str(best_valid); print "TEST : "+str(best_test)
			best_train=np.inf; best_valid=np.inf; best_test=np.inf
			active_phase = False
			increment=increment_init
			# selection step
			# step1 : build the commitee
			# step2 : do the retraining
			# step3 : pick a subset of the unnatotated training set
			# step4 : make the committee predictions
			# step5 : pick the best samples
			# step6 : add them to the training set
			# step7 : increment n_train_batches
			# step9 : pickle the sentence and their score !
			# step8 : rerun

			#STEP 1
			committee = t_nlp.generate_committee()
			#STEP 2
			sub_train = (x_train[:n_train_batches*batch_size], y_train[:n_train_batches*batch_size])
			training_committee(committee, learning_rate, decay_rate, sub_train , valid, batch_size)
			temp_decision = [ instance.predict(x) for instance in committee] # changer la decision !!!
            		function = theano.function(inputs=[x], outputs= temp_decision,
								allow_input_downcast=True)
			#STEP 3
			minibatches_comitee = (np.random.permutation(n_train - n_train_batches) + n_train_batches)[:200]

			#evaluation
			print 'EVALUATION TO DO !!!'
            		minibatches_errors = []
            		temp = np.zeros((2, batch_size*len(minibatches_comitee))) # 2 classes !
            		for j, minibatch in zip(xrange(len(minibatches_comitee)), minibatches_comitee):
                		x_batch = x_train[minibatch*batch_size:(minibatch+1)*batch_size]
				if len(x_batch)==0:
					import pdb
					pdb.set_trace()
					print 'merde'
                		value = np.asarray(function(x_batch))
                		for index in xrange(batch_size):
                    			for i in xrange(len(committee)):
                        			temp[ value[i,index] , batch_size*j + index]+= 1;
            		for q in xrange(batch_size*len(minibatches_comitee)):
                		score = ( len(committee) - np.max(temp[:,q]))/ (1.*len(committee))
                		minibatches_errors.append(score)
            		index_to_add_ = np.argsort(minibatches_errors)[::-1][:batch_size]
			print np.sort(minibatches_errors)[::-1][:20]
            		index_to_add = []
            		for q in xrange(batch_size):
                		first_coord = index_to_add_[q]/batch_size # indice du minibatch qu'on va selectionner
                		second_coord = index_to_add_[q] - batch_size*first_coord # indice du sample dans le minibatch
                		index_to_add.append([minibatches_comitee[first_coord], second_coord]) # FAUX ???
			recorded_data = []
	            	for j,coord, score in zip(xrange(len(index_to_add)), index_to_add, np.sort(minibatches_errors)[::-1][:batch_size]):
                		coord_0 = coord[0]
               			coord_1 = coord[1]
                		temp_x = x_train[n_train_batches*batch_size + j:(n_train_batches)*batch_size + j+1]
                		temp_y = y_train[n_train_batches*batch_size + j:(n_train_batches)*batch_size + j+1]
				if len(recorded_data) <20:
					recorded_data.append((temp_x, temp_y, score))
                		x_train[n_train_batches*batch_size + j:n_train_batches*batch_size + j+1]=\
                		x_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1]
                		y_train[n_train_batches*batch_size + j:n_train_batches*batch_size + j+1]=\
                		y_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1]
                		x_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1] = temp_x
                		y_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1] = temp_y
            		n_train_batches+=1;
			saving_data(repo, recorded_data)

            		t_nlp.initialize()
            		# init cache
			# update init
			reinit()
			
"""			
		


if __name__=='__main__':
	learning_rate = 1e-4
	decay_rate = 0.4
	dwin=20
	repo='data/dico'
	#n_out = 20
	#vect_size = 40
	#n_hidden_layer=[100, 100]
	filenames = ['HollandeDef.cnr', 'Sarkozy-Inter-DEF.cnr']
	output_dico="embedding_dico_H_S_v3"
	#filenames = ['HollandeDef.cnr', 'GaulleDef.cnr']
	#output_dico ="embedding_dico_H_G_v0"
	#training_Hollande(repo, output_dico, filenames, dwin, learning_rate, decay_rate)
