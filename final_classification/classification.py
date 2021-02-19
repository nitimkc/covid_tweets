from reader import TweetsCorpusReader
from transformers import TextNormalizer, GensimVectorizer
from build_evaluate import build_and_evaluate   
from build_evaluate import show_most_informative_features
from build_evaluate_svm  import build_and_evaluateSVM

from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import nltk
import os
import json
import logging
import re

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

ROOT = Path(r'/Users/peaceforlives/Documents/Projects/covid_tweets/')         # mac
CORPUS = Path.joinpath(ROOT, 'data', 'labelled_tweets', 'ab')
# FULL_CORPUS = Path.joinpath(ROOT, 'data', 'located_tweets')
RESULTS = Path.joinpath(ROOT, 'results', 'bullyphysicalverbal_combined')

DOC_PATTERN = r'.*\.json'

if __name__ == "__main__":
    labels = {'bullying_trace':['no','yes'],
                'bullying_role':['accuser', 'defender','other','reporter','victim'],
                'form_of_bullying':['cyber','general', 'xenophobia'],
                'bullying_post_type':['accusation','cyberbullying','denial','report','self-disclosure']}
    targets = list(labels.keys())
    svm=True

    for target in targets:
        print(target)
        categories = labels[target]
        label = {i:categories[i] for i in range(len(categories))}

        corpus = TweetsCorpusReader(CORPUS.__str__(), DOC_PATTERN, bullying_trace=target)     
        processed_tweets = corpus.process_tweet()
        print(len(processed_tweets), 'no. of tweets labelled')

        normalize = TextNormalizer()
        X = list(normalize.fit_transform(processed_tweets)) 
        y = list(corpus.fields(target))

        PATH = Path.joinpath(RESULTS, target+"_spacy.pickle")
        if len(categories)<2:
            print('at least two categories needed')
        elif len(categories)==2:
            multiclass=False
            n=0.3
        else:
            multiclass=True
            n=0.2
        if (svm==True) and (target!='bullying_trace'):
            print('SVC')
            model, y_pred, idx_test = build_and_evaluateSVM(X, y, outpath=PATH, n=n)
        else:
            print('LogisticRegression')
            # model, y_pred, idx_test = build_and_evaluate(X, y, n=n, outpath=PATH, verbose=True, multiclass=multiclass, spacy=True) 
            model, y_pred, idx_test = build_and_evaluate(X, y, n=n, outpath=PATH, verbose=True, multiclass=multiclass) 
            print(show_most_informative_features(model))

        y_pred_new = model.predict(X[i] for i in idx_test)

        target_data = pd.DataFrame(corpus.docs())
        target_data = target_data[['id','full_tweet', target]]
        test_data = target_data.iloc[idx_test,]
        test_data['pred'] = list(map(label.get, y_pred))
        test_data.to_csv(Path.joinpath(RESULTS,target+'.csv'), encoding='utf-8') # save test set with actual and predicted labels
