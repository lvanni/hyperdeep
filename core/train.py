'''
Created on 7 déc. 2016

@author: lvanni
'''

if __name__ == '__main__':
    # Preprocessing => découpage des textes
    corpus = []
    for filename in os.listdir(CORPUS_PATH):
        if ".txt" in filename:
            corpus.append(CORPUS_PATH + filename)
    x_train, x_valid, x_test, y_train, y_valid, y_test = pre_process(corpus)
    
    # Training, Validation, Test
    training(x_train, x_valid, x_test, y_train, y_valid, y_test)