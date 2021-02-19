# 1. read corpus
# 2. create loader object (iterable of train and test sets)
# 3. 

from reader import TweetsCorpusReader
import numpy as np
import pickle
import nltk
import os
import json
import logging
import re

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

#windows
ROOT = 'C:\\Users\\niti.mishra\\Documents\\Personal\\cyberbullying\\'
CORPUS = os.path.join(ROOT, 'data\\labelled_tweets\\ab')
RESULTS = os.path.join(ROOT, 'results')
# file = CORPUS+'\\random_tweets_2b.json'

#mac
# ROOT = '/Users/peaceforlives/Documents/Projects/cyberbullying/'
# CORPUS = os.path.join(ROOT, 'data/labelled_tweets')
# RESULTS = os.path.join(ROOT, 'results')

DOC_PATTERN = r'.*\.json'  
# file = CORPUS+'/tweets.json'


if __name__ == "__main__":
    PATH = "model.pickle"

    # if not os.path.exists(PATH):
    # Time to build the model
    from reader import TweetsCorpusReader
    from transformers import TextNormalizer_lemmatize, GensimVectorizer

    corpus = TweetsCorpusReader(CORPUS, DOC_PATTERN, bullying_trace = 'bullying_trace')        
    processed_tweets = corpus.process_tweet()
    normalize  = TextNormalizer_lemmatize()
    X = list(normalize.fit_transform(processed_tweets)) # X = [' '.join(doc) for doc in normalized_tweets]
    y = list(corpus.fields('bullying_trace'))
    
    from build_evaluate import build_and_evaluate   
    model = build_and_evaluate(X, y, outpath=PATH, n=0.2)

    # else:
    #     with open(PATH, 'rb') as f:
    #         model = pickle.load(f)

    from build_evaluate import show_most_informative_features
    print(show_most_informative_features(model))
