##########################################################################
## transformers.py
##########################################################################

# !/usr/bin/env python3

import os
import nltk
import gensim
import unicodedata
import string 

# from loader import CorpusLoader
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.stem import SnowballStemmer

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.matutils import sparse2full

class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english', lemma=True):
        self.lemmate = lemma 
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer(language)

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def normalize_lemm(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for (token, tag) in document
            if not self.is_punct(token) #and not self.is_stopword(token)
        ]
    
    def stemmize(self, token, pos_tag):
        return self.stemmer.stem(token.lower())

    def normalize_stem(self, document):
        return [
            self.stemmize(token, tag).lower()
            for (token, tag) in document
            if not self.is_punct(token) #and not self.is_stopword(token)
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            if self.lemmate:
                yield self.normalize_lemm(document)
            else:
                yield self.normalize_stem(document)

    # def transform(self, documents):
    #     norm_tweet = []
    #     for document in documents:
    #         if self.lemmate:
    #             norm_tweet.append(self.normalize_lemm(document))
    #         else:
    #             norm_tweet.append(self.normalize_stem(document))  
    #     return norm_tweet



class GensimVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, path=None):
        self.path = path
        self.id2word = None

        self.load()

    def load(self):
        if os.path.exists(self.path):
            self.id2word = gensim.corpora.Dictionary.load(self.path)

    def save(self):
        self.id2word.save(self.path)

    def fit(self, documents, labels=None):
        self.id2word = gensim.corpora.Dictionary(documents)
        self.save()

    def transform(self, documents):
        for document in documents:
            for tweet in document:
                tweetvec = self.id2word.doc2bow(tweet)
                yield sparse2full(tweetvec, len(self.id2word))