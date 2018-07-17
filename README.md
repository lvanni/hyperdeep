# HYPERDEEP
Hyperdeep is a proof of concept of the deconvolution algorithm applied on text.
It use a standard CNN for text classification, and include one layer of deconvolution for the automatic detection of linguistic marks. This software is integrated on the Hyperbase web platform (http://hyperbase.unice.fr) - a web toolbox for textual data analysis.

# requirements:
- python3
- keras + tensorflow

# HOW TO USE IT
# data
Datas are stored in the data/ folder. The training set should be splited into phrases of fixed length (50 words by default). And each phrase should have a label name at the begining of the line. A label is written : __LABELNAME__
Hyperdeep is distributed with an example of corpus named Campagne2017 (in data/Campagne2017). This is a french corpus of the 2017 presidential election in france. There are 5 main candidates encoded with the labels (Melenchon, Hamon, Macron, Fillon, LePen).

# train skipgram
You can train a skipgram model (word2vec) by using the command :
	$ python hyperdeep.py skipgram -input data/Campagne2017 -output bin/Campagne2017.vec
This command will create a file named Campagne2017.vec in the folder bin (create the folder if needed). This txt is the vectors file of the training data.

# test skipgram
the skipgram model can be tested by using the command (example with the word France) :
	$ python hyperdeep.py nn bin/Campagne2017.vec France

# train classifier
To train the classifier:
	$ python3 hyperdeep.py train -input data/Campagne2017 -output bin/Campagne2017
The command will create bin/Campagne2017.index for the vocabulary and bin/Campagne2017 for the model

# test the classifier
Then you can make predictions on new text. There is an example in bin/Campagne2017.test. It's a discourse of E. Macron as french president on 31th december 2017. Hyperdeep will split the discourse in fixed length phrases and should predict most of the phrase a E. Macron
	$ python hyperdeep.py predict bin/Campagne2017 bin/Campagne2017.vec data/Campagne2017.test

# Observe the deconvolution
The predict command line will create a result file in the folder result/ (create the folder if needed). This file is a json format file where you can find the activation score given by the deconvolution for each word. An example of results is given in result/Campagne2017.test.	res