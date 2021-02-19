import os
import time
import string
import pickle
import numpy as np
from operator import itemgetter

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import classification_report as clsr
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split as tts

import spacy

def timeit(func):
    """
    Simple timing decorator
    """
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        delta  = time.time() - start
        return result, delta
    return wrapper

def identity(words):
    return words

def spacy_tokenizer(document):
    tokens = nlp(' '.join(document))
    tokens = [token.lemma_ for token in tokens if (
        token.is_stop == False and \
        token.is_punct == False and \
        token.lemma_.strip()!= '')]
    return tokens

nlp = spacy.load("en_core_web_md")

def build_and_evaluate(X, y, n=None, method=LogisticRegression, outpath=None, verbose=True, multiclass=False, spacy=False):
    """
    Builds a classifer for the given list of documents and targets in two
    stages: the first does a train/test split and prints a classifier report,
    the second rebuilds the model on the entire corpus and returns it for
    operationalization.
    X: a list or iterable of raw strings, each representing a document.
    y: a list or iterable of labels, which will be label encoded.
    Can specify the classifier to build with: if a class is specified then
    this will build the model with the Scikit-Learn defaults, if an instance
    is given, then it will be used directly in the build pipeline.
    If outpath is given, this function will write the model as a pickle.
    If verbose, this function will print out information to the command line.
    """

    @timeit
    def build(classifier, X, y=None):
        """
        Inner build function that builds a single model.
        """
        if isinstance(classifier, type):
            if multiclass:
                classifier = classifier(random_state=0,  max_iter=1000, multi_class="multinomial", solver='lbfgs') 
            else:
                classifier = classifier(random_state=0,  max_iter=1000, solver='liblinear')
        
        gridsearch_pipe = Pipeline([
            # ('preprocessor', TextNormalizer_lemmatize()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1,2)) ),
            ('classifier', classifier)
        ])

        maxdf = [0.85, .90]
        mindf = (4, 3, 2)
        max_features = [ 13500, 13600, 13700]
        ngrams = [(1, 1), (1, 2), (1,3)]
        
        C = np.array([.001, .01, .1, 1, 10, 100, 1000])
        
        param_grid = {'vectorizer__max_df':maxdf, 'vectorizer__min_df':mindf, 'vectorizer__ngram_range':ngrams, 'vectorizer__max_features':max_features, 'classifier__C':C}
        grid_search = GridSearchCV(gridsearch_pipe, param_grid, cv=6)
        grid_search.fit(X, y)
        best_param = grid_search.best_params_
        print(best_param)
        # {'classifier__C': 1.0, 'vectorizer__max_df': 0.85, 'vectorizer__max_features': 13500, 'vectorizer__min_df': 3, 'vectorizer__ngram_range': (1, 2)}
        # {'classifier__C': 1.0, 'vectorizer__max_df': 0.85, 'vectorizer__max_features': 13500, 'vectorizer__min_df': 2, 'vectorizer__ngram_range': (1, 2)}
        vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, 
                        max_df=best_param['vectorizer__max_df'], min_df=best_param['vectorizer__min_df'], 
                        ngram_range=best_param['vectorizer__ngram_range'], max_features=best_param['vectorizer__max_features'])
        classifier = LogisticRegression( random_state=0, max_iter=1000, C=best_param['classifier__C'])
        
        if spacy:
            vectorizer = TfidfVectorizer(input = 'content', tokenizer=spacy_tokenizer, preprocessor=None, lowercase=False, 
                            max_df=best_param['vectorizer__max_df'], min_df=best_param['vectorizer__min_df'], 
                            ngram_range=best_param['vectorizer__ngram_range'], max_features=best_param['vectorizer__max_features'])
        
        model = Pipeline([
            # ('preprocessor', TextNormalizer_lemmatize()),
            ('vectorizer', vectorizer),
            ('classifier', classifier),
        ])

        model.fit(X, y)
        
        return model

    # Label encode the targets
    labels = LabelEncoder()
    y = labels.fit_transform(y)

    # Begin evaluation
    if n:
        if verbose: print("splitting test and test set by: "+str(n))
        n_samples = len(y)
        indicies = np.arange(n_samples)  
        X_train, X_test, y_train, y_test, idx_train, idx_test = tts(X, y, indicies, test_size=n, stratify=y)
        print(len(X_train), len(X_test))
        from collections import Counter
        print('y_train', Counter(y_train))

        model, secs = build(method, X_train, y_train)
        model.labels_ = labels

        if verbose: print("Evaluation model fit in {:0.3f} seconds".format(secs))
        y_pred = model.predict(X_test)
        # z = 0.70
        # print(z)
        # y_prob = model.predict_proba(X_test)[:,1]
        # print(y_prob)
        # y_pred = np.where(y_prob > z, 1,0)

        if verbose: print("Classification Report:\n")
        print(clsr(y_test, y_pred, target_names=labels.classes_))
        print(cm(y_test, y_pred))
        print('acc', accuracy_score(y_test, y_pred))
        print('f1', f1_score(y_test, y_pred, average='weighted'))

        if verbose: print("Evaluation of naive prediction ...")
        y_naive = [0]*len(y_test)
        print('acc naive', accuracy_score(y_test, y_naive))
        print('f1 naive', f1_score(y_test, y_naive, average='weighted'))

    else:
        if verbose: print("Building for evaluation with full set")    
        model, secs = build(classifier, X, y)
        model.labels_ = labels

        if verbose: print("Evaluation model fit in {:0.3f} seconds".format(secs))
        y_pred = model.predict(X)

        if verbose: print("Classification Report:\n")
        print(clsr(y, y_pred, target_names=labels.classes_))
        print(cm(y, y_pred))
        print(accuracy_score(y, y_pred))

        if verbose: print("Evaluation of naive prediction ...")
        y_naive = [0]*len(y)
        print(type(y))
        print('acc naive', accuracy_score(y, y_naive))

    if verbose: print("Complete model fit in {:0.3f} seconds".format(secs))

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model, y_pred, idx_test


def show_most_informative_features(model, text=None, n=10):
    """
    Accepts a Pipeline with a classifer and a TfidfVectorizer and computes
    the n most informative features of the model. If text is given, then will
    compute the most informative features for classifying that text.
    Note that this function will only work on linear models with coefs_
    """
    # Extract the vectorizer and the classifier from the pipeline
    vectorizer = model.named_steps['vectorizer']
    classifier = model.named_steps['classifier']

    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {} model.".format(
                classifier.__class__.__name__
            )
        )

    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append("Classified as: {}".format(model.predict([text])))
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn)
        )

    return "\n".join(output)


# ROOT = '/Users/peaceforlives/Documents/Projects/cyberbullying/'
# CORPUS = os.path.join(ROOT, 'data/labelled_tweets')
# RESULTS = os.path.join(ROOT, 'results')

# DOC_PATTERN = r'.*\.json' 

# if __name__ == "__main__":
#     PATH = "model.pickle"

#     if not os.path.exists(PATH):
#         # Time to build the model
#         from reader import TweetsCorpusReader
        
#         corpus = TweetsCorpusReader(CORPUS, DOC_PATTERN)

#         X = corpus.docs
#         y = list(corpus.fields('bullying_trace'))

#         model = build_and_evaluate(X,y, outpath=PATH)

#     else:
#         with open(PATH, 'rb') as f:
#             model = pickle.load(f)

#     print(show_most_informative_features(model))