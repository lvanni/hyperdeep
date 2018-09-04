import pickle
import numpy as np

def tokenize(texts, model_file, create_dictionnary, config):

	if create_dictionnary:
		my_dictionary = {}
		my_dictionary["word_index"] = {}
		my_dictionary["index_word"] = {}
		my_dictionary["word_index"]["PAD"] = 0
		my_dictionary["index_word"][0] = "PAD"
		index = 0
	else:
		with open(model_file + ".index", 'rb') as handle:
			my_dictionary = pickle.load(handle)

	data = (np.zeros((len(texts), config["SEQUENCE_SIZE"]))).astype('int32')

	i = 0
	for line in texts:
		words = line.split()[:config["SEQUENCE_SIZE"]]
		sentence_length = len(words)
		sentence = []
		for word in words:
			if word not in my_dictionary["word_index"].keys():
				if create_dictionnary:
					index += 1
					my_dictionary["word_index"][word] = index
					my_dictionary["index_word"][index] = word
					star = False
					if "Star" in word:
						star = True
						print("\t", word, my_dictionary["word_index"][word])
					
					# FOR UNKNOWN WORDS     
					if "**" in word:
						args = word.split("**")
						word = args[1] + "**" + args[1] + "**" + args[1]
						if word not in my_dictionary["word_index"].keys():
							index += 1
							my_dictionary["word_index"][word] = index
							my_dictionary["index_word"][index] = word
							if star:
								print("\t\t", word, my_dictionary["word_index"][word])

				else:        
					# FOR UNKNOWN WORDS
					if "**" in word:
						args = word.split("**")    
						word = args[1] + "**" + args[1] + "**" + args[1]
						if word in my_dictionary["word_index"].keys():
							my_dictionary["word_index"][word] = my_dictionary["word_index"][word]
						else:
							my_dictionary["word_index"][word] = my_dictionary["word_index"]["PAD"]
					else:
						my_dictionary["word_index"][word] = my_dictionary["word_index"]["PAD"]


			sentence.append(my_dictionary["word_index"][word])

		# COMPLETE WITH PAD IF LENGTH IS < SEQUENCE_SIZE
		if sentence_length < config["SEQUENCE_SIZE"]:
			for j in range(config["SEQUENCE_SIZE"] - sentence_length):
				sentence.append(my_dictionary["word_index"]["PAD"])
		
		data[i] = sentence
		i += 1

	if create_dictionnary:
		with open(model_file + ".index", 'wb') as handle:
			pickle.dump(my_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return my_dictionary, data