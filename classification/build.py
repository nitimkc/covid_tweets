# 1. create pipeline of normalizer and vectorizer
# 2. apply pipeline to each classification model 
# 3. function to store model scores

import nltk
nltk.download('stopwords')
import unicodedata
import numpy as np
import time
import json

from reader import TweetsCorpusReader
from loader import CorpusLoader

from transformers import TextNormalizer, GensimVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def identity(words):
    return words

def create_pipeline(estimator, reduction=False):

    steps = [
        ('normalize', TextNormalizer(lemma=True)),
        # ('to_dense', DenseTransformer()),
        # ('vectorize', CountVectorizer(binary=True, lowercase=False)) #ohe
        # ('vectorize', CountVectorizer(ngram_range=(1, 4), analyzer='char', lowercase=False)) #freq
        ('vectorize', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1,2), max_features=13000, max_df=0.85, min_df=2))        
    ]
    
    if reduction:
        steps.append(('reduction', TruncatedSVD(n_components=600)))
    
    steps.append(('classifier', estimator))

    return Pipeline(steps)


binary_models = []
for form in (LogisticRegression, SGDClassifier):
    binary_models.append(create_pipeline(form(), True))
    binary_models.append(create_pipeline(form(), False))
binary_models.append(create_pipeline(SVC(kernel='linear'), False))
# binary_models.append(create_pipeline(MultinomialNB(), False))
# binary_models.append(create_pipeline(GaussianNB(), True))

multiclass_models = []
multiclass_models.append(create_pipeline(LogisticRegression(solver='newton-cg',multi_class="multinomial"), False))
multiclass_models.append(create_pipeline(SVC(kernel='linear'), False))
multiclass_models.append(create_pipeline(KNeighborsClassifier(n_neighbors = 8), False))
# multiclass_models.append(create_pipeline(GaussianNB(), False))
# multiclass_models.append(create_pipeline(RandomForestRegressor(n_estimators = 1000, random_state = 42), False)) # requires to_dense in pipeline

def score_models(models, loader):

    for model in models:

        name = model.named_steps['classifier'].__class__.__name__
        if 'reduction' in model.named_steps:
            name += " (TruncatedSVD)"

        scores = {
            'model': str(model),
            'name': name,
            'size': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_valid': [],
            'f1_train': [],
            'time': [],
        }

        for X_train, X_test, y_train, y_test in loader:
            from collections import Counter
            print(len(X_train))
            print(len(X_test))
            print('y_train', Counter(y_train))
            print('y_test', Counter(y_test))
            
            start = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            scores['time'].append(time.time() - start)
            scores['size'].append([len(X_train), len(X_test)])
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            scores['f1_valid'].append(f1_score(y_test, y_pred, average='weighted'))
            scores['f1_train'].append(f1_score(y_train, y_train_pred, average='weighted'))
            print('model: ', scores['name'])
            print('accuracy: ', scores['accuracy'])
            print('precision: ', scores['precision'])
            print('recall: ', scores['recall'])
            print('f1_valid: ', scores['f1_valid'])
            print('f1_train: ', scores['f1_train'])
            print('time: ', scores['time'])

        yield scores

# if __name__ == '__main__':
#     for scores in score_models(binary_models, loader):
#         with open('results.json', 'a') as f:
#             f.write(json.dumps(scores) + "\n")

# for X_train, X_test, y_train, y_test in loader:
#     x = X_train
#     print(len(x))