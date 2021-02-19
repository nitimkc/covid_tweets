# create iterable of train and test sets for CV

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split as tts

class CorpusLoader(object):

    def __init__(self, reader, folds=12, shuffle=True, label=None, size=None):
        self.reader = reader
        self.folds  = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=78)
        self.files  = np.asarray(list(self.reader.fields('id')))
        self.label = label
        self.idx = range(0,size)


    def documents(self, idx=None):
        tweets = self.reader.process_tweet()
        if idx is None:
            return tweets
        return  list(tweets[i] for i in idx)

    def labels(self, idx=None):
        labels = list(self.reader.fields(self.label))
        if idx is None:
            return labels
        return list(labels[i] for i in idx)

    def __iter__(self):
        i=1
        # for train_index, test_index in self.folds.split(self.idx):
        for train_index, test_index in self.folds.split(self.documents(), self.labels()):
            print('iter', i)
            print(len(train_index), len(test_index))
            
            i = i+1
            X_train = self.documents(train_index)
            y_train = self.labels(train_index)
            X_test = self.documents(test_index)
            y_test = self.labels(test_index)

            yield X_train, X_test, y_train, y_test

