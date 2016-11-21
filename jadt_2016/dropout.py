# -*- coding: utf-8 -*-
# DROPOUT function
#@author Mélanie Ducoffe
#@date 8/09/2015

import re
import numpy as np
import theano.tensor as T
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.bricks import WEIGHT, BIAS

#analyze_param_name:
def analyze_param_name(name):
    prog = re.compile(r"(layer)_(\d+)_([^_]*)")
    m = prog.match(name)
    if m:
        layer_name = m.group(1)
        layer_number =int(m.group(2))
        role = m.group(3)
        return (layer_name, layer_number, role)
    else:
        print "Failed to get layer number from %s." %name
        return None
        
def get_single_dropout_indices(shape, dropout_pair):

    assert len(dropout_pair) ==2
    
    if dropout_pair[0] == 0.0:
        index_row = np.arange(shape[0])
    else:
        (N,p) =(shape[0], dropout_pair[0])
        N_kept = N -int(N*p)
        index_row = np.sort(np.random.permutation(N)[:N_kept])
        assert len(index_row) == N_kept
        
    if dropout_pair[1] == 0.0:
        index_col = np.arange(shape[1])
    else:
        (N,p) = (shape[1], dropout_pair[1])
        N_kept = N -int(N*p)
        index_col = np.sort(np.random.permutation(N)[:N_kept])
        assert len(index_col) == N_kept

    index_row = index_row.astype(np.intc)
    index_col = index_col.astype(np.intc)
    
    return [index_row, index_col]
    

def dropout_indices(L_W, L_B, dropout):
    # WARNING : we assume that the lists are sorted by layer_number
    splits_for_W={}
    rough_kinds = {}
    for w in L_W:
        w_shape = w.shape.eval()
        layer_name, layer_number, _ = analyze_param_name(w.name)
        dropout_pair = [dropout, dropout]
        if not layer_number:
            dropout_pair[0] = 0 # no dropout on the input
        if layer_number == len(L_W) - 1:
            dropout_pair[1] = 0 # no dropout on the output
        if w.ndim ==4:
            #CONV_FILTER
            [index_col, index_row] = get_single_dropout_indices((w_shape[1], w_shape[0]), dropout_pair)
            # retrive the name of the layer
            splits_for_W[layer_name+str(layer_number)] = [index_col, index_row]
            rough_kinds[layer_name+str(layer_number)] = "CONV_FILTER"
        elif w.ndim ==2:
            #FULLY CONNECTED
            [index_col, index_row] = get_single_dropout_indices((w_shape[0], w_shape[1]), dropout_pair)
            splits_for_W[layer_name+str(layer_number)] = [index_col, index_row]
            rough_kinds[layer_name+str(layer_number)] = "FULLY_CONNECTED"
        else:
            raise Exception('unknow size for a parameter %d', w.ndim)
            
    #making two consecutive layers agree on the splits for "W"
    for w in L_W[:-1]:
        
        layer_name, layer_number, _ = analyze_param_name(w.name)
        layer_name_i = layer_name+str(layer_number)
        layer_name_o = layer_name+str(layer_number+1)
        # 1 FULLY_CONNECTED -> FULLY_CONNECTED
        if (rough_kinds[layer_name_o] == "FULLY_CONNECTED" and
            rough_kinds[layer_name_i] == "FULLY_CONNECTED"):
	    if layer_number==0:
		# we are in a case of an embedding where we repeat dwin times the weights
		# be VERY CAREFUL about this section
		shape_input = w.shape[1].eval()
		# find the next layer
            	for w_next in L_W:
                    layer_name_next, layer_next_number, _ = analyze_param_name(w_next.name)
                    if layer_next_number == layer_number +1:
                        shape_output = w_next.shape.eval()[0]
                    	break
		c=shape_output/shape_input
		if c*shape_input != shape_output:
		    print "You have a problem with your configuration of dropout at the junction of the convolution and fully-connected layers."
                    raise Exception("Setup for split indices at LOOKUP -> FULLY_CONNECTED cannot proceed.")
		indices = []
		for k in range(c):
			for elem in splits_for_W[layer_name_i][1]:
				indices.append(elem +k)
		splits_for_W[layer_name_o][0] = np.asarray(indices, dtype=np.intc)

                continue
	    else:
            	splits_for_W[layer_name_o][0] = splits_for_W[layer_name_i][1]
            continue
        if (rough_kinds[layer_name_next] == "CONV_FILTER" and 
            rough_kinds[layer_name] == "FULLY_CONNECTED"):

            raise Exception("FULLY_CONNECTED -> CONV_FILTER not implemented")  
            
    splits_for_b = {}
    just_the_zero_index = np.array([0], dtype=np.intc)
    for b in L_B:
        (layer_name, layer_number, _) = analyze_param_name(b.name)
        layer_name = layer_name+str(layer_number)
        if rough_kinds[layer_name] == "FULLY_CONNECTED":
            splits_for_b[layer_name] = splits_for_W[layer_name][1]
        elif rough_kinds[layer_name] == "CONV_FILTER":
            splits_for_b[layer_name] = splits_for_W[layer_name][1]
        else:
            raise Exception("bug !")

    return splits_for_W, splits_for_b, rough_kinds
    
    
def retrieve_params(mlp_full, mlp_sub, dropout, exo_dropout=0.):
    # get the list of parameters
    assert exo_dropout>=0 and exo_dropout<1
    assert dropout>=0 and dropout<1
    L_W_full, L_B_full = mlp_full.get_Params()
    L_W_sub, L_B_sub = mlp_sub.get_Params()
    # sort the lists here
    L_W_full.sort(key=lambda e: analyze_param_name(e.name)[1])
    L_W_sub.sort(key=lambda e: analyze_param_name(e.name)[1])
    L_B_full.sort(key=lambda e: analyze_param_name(e.name)[1])
    L_B_sub.sort(key=lambda e: analyze_param_name(e.name)[1])

    # get the splits
    splits_W, splits_B, rough_kinds = dropout_indices(L_W_full, L_B_full, dropout)
    for w_full, w_sub in zip(L_W_full, L_W_sub):
        #get the indices :
        (layer_name, layer_number, _) = analyze_param_name(w_full.name)
        layer_name = layer_name+str(layer_number)
        w_split = splits_W[layer_name]
        rough_kind = rough_kinds[layer_name]
        if rough_kind == "FULLY_CONNECTED":
	    #import pdb
	    #pdb.set_trace()
            w_sub.set_value(w_full.get_value()[w_split[0]][:,w_split[1]])
            if w_sub.name != w_full.name:
                raise Exception( "names %s and %s do not match", w_sub.name, w_full.name)
	else:
	    raise Exception("unknown type :%s", rough_kind)
    for  b_full, \
        b_sub in zip(L_B_full, L_B_sub):
        #get the indices :
        (layer_name, layer_number, _) = analyze_param_name(b_full.name)
        layer_name = layer_name+str(layer_number)
        b_split = splits_B[layer_name]
        rough_kind = rough_kinds[layer_name]
        if rough_kind == "FULLY_CONNECTED":
	    b_sub.set_value(b_full.get_value()[b_split])
	    if b_sub.name != b_full.name:
                raise Exception("names %s and %s do not match", b_sub.name, b_full.name)
	else:
	    raise Exception("unknown type :%s", rough_kind)
