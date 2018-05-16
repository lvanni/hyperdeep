import pickle
import numpy as np

from config import MAX_SEQUENCE_LENGTH

def tokenize(texts, model_file, create_dictionnary):

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
				#print(word,  list(my_dictionary["word_index"].keys())[:20])
				if create_dictionnary:
					index += 1
					my_dictionary["word_index"][word] = index
					my_dictionary["index_word"][index] = word
					"""
					if "**" in word:
						args = word.split("**")
						for arg in args:
							if arg not in my_dictionary["word_index"].keys():
								index += 1
								my_dictionary["word_index"][arg] = index
								my_dictionary["index_word"][index] = arg
					"""
				else:        
					my_dictionary["word_index"][word] = my_dictionary["word_index"]["<PAD>"]
					"""     
					if "**" in word:
						args = word.split("**")
						for arg in args:        
							if arg in my_dictionary["word_index"].keys():
								my_dictionary["word_index"][word] = my_dictionary["word_index"][arg]
					"""        
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