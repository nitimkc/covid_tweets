from reader import TweetsCorpusReader
from transformers import TextNormalizer, GensimVectorizer
from build_evaluate import build_and_evaluate   
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
FULL_CORPUS = Path.joinpath(ROOT, 'data', 'located_tweets')
RESULTS = Path.joinpath(ROOT, 'results', 'finalselection')

DOC_PATTERN = r'.*\.json'

#################################
# predictions for entire dataset
# 25467
#################################

# full_corpus = TweetsCorpusReader(FULL_CORPUS.__str__(), DOC_PATTERN, bullying_trace=None)
# docs = full_corpus.docs()
# print(len(docs))
# data_docs = pd.DataFrame(docs)
# print(data_docs.shape)
# data_docs.to_pickle(Path.joinpath(RESULTS,"full_data.pkl"))
# data_docs.to_csv(Path.joinpath(RESULTS,"full_data.csv"), encoding='utf-8-sig')
data_docs = pd.read_pickle(Path.joinpath(RESULTS,'full_data.pkl'))

# full_processedtweets = full_corpus.process_tweet()
# normalize  = TextNormalizer()
# full_X = list(normalize.fit_transform(full_processedtweets)) # X = [' '.join(doc) for doc in normalized_tweets]
# with open(Path.joinpath(RESULTS,'full_X.txt'), 'wb') as f:
#     pickle.dump(full_X, f)
with open(Path.joinpath(RESULTS, 'full_X.txt'), 'rb') as f:
    full_X = pickle.load( f)
print(len(full_X))

labels = {'bullying_trace':['no','yes'],
            'bullying_role':['accuser', 'defender','other','reporter','victim'],
            'form_of_bullying':['cyber','general', 'physical', 'verbal', 'xenophobia'],
            'bullying_post_type':['accusation','cyberbullying','denial','report','self-disclosure']}
targets = list(labels.keys())

#################################
# predict bullying trace
# 6974
#################################

PATH = Path.joinpath(RESULTS,targets[0]+"_spacy.pickle")
with open(PATH, 'rb') as f:
    model = pickle.load(f)

full_pred = model.predict(full_X)
from collections import Counter
Counter(full_pred)

label = {i:labels[targets[0]][i] for i in range(len(labels[targets[0]]))}
print(label)
data_docs[targets[0]] = list(map(label.get, full_pred))

bullying_traces_idx = data_docs[data_docs['bullying_trace']=='yes'].index
bullying_traces = data_docs.loc[bullying_traces_idx]
bullying_traces_X = [full_X[i] for i in bullying_traces_idx]

#################################
# predict rest
#################################
for target in targets[1:]:
    print(target)
    PATH = Path.joinpath(RESULTS,target+"_spacy.pickle")
    with open(PATH, 'rb') as f:
        model = pickle.load(f)
    full_pred = model.predict(bullying_traces_X)
    from collections import Counter
    Counter(full_pred)
    label = {i:labels[target][i] for i in range(len(labels[target]))}
    print(label)
    bullying_traces[target]= list(map(label.get, full_pred))

cols = ['id']
cols.extend(targets)
all_pred = pd.merge(data_docs['id', 'full_tweet'], bullying_traces[cols], how='outer')
print(all_pred.columns)

for target in targets:
    print(target)
    print(all_pred[target].value_counts())

all_pred.to_pickle(Path.joinpath(RESULTS,"data_predicted.pkl"))


################################################################################
# 2500 sample of bullying and non bullying trace tweets as predicted by machine
################################################################################

# yes = data_docs[data_docs['bullying_trace']=='yes']
# yes_sample = yes.sample(n=2500)

# no = data_docs.loc[data_docs['bullying_trace']=='no']
# no_sample = no.sample(n=2500)

# sample_tweets = pd.concat([yes_sample,no_sample])
# sample_tweets = sample_tweets.sample(frac=1)
# sample_tweets.to_csv('results/sample_tweets.csv', encoding='utf-8-sig', index=False)


#####################
# read saved files
####################
# sample_tweets = pd.read_csv('results/sample_tweets.csv', encoding='utf-8-sig')
all_pred = pd.read_pickle(Path.joinpath(RESULTS,"data_predicted.pkl"))

table = pd.pivot_table(all_pred, index=['bullying_role'], columns=['form_of_bullying'], aggfunc=np.sum)
all_pred.groupby(['form_of_bullying', 'bullying_role']).size()
groups = all_pred.groupby(['form_of_bullying', 'bullying_post_type', 'bullying_role']).size()
groups['xenophobia']
