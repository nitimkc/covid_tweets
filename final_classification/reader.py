import json
import os
import re
import nltk
import time
import pickle
from nltk import pos_tag
from six import string_types

from nltk.tokenize import TweetTokenizer

from nltk.corpus.reader.util import StreamBackedCorpusView, concat, ZipFilePathPointer
from nltk.corpus.reader.api import CorpusReader

DOC_PATTERN = r'.*\.json' 
# PKL_PATTERN = r'.*\.pickle'

##########################################################################
## Tweets Corpus Reader
##########################################################################

class TweetsCorpusReader(CorpusReader):

    """
    Reader for corpora that consist of Tweets represented as a list of line-delimited JSON.
    Individual Tweets can be tokenized using the default tokenizer, or by a
    custom tokenizer specified as a parameter to the constructor.
    Construct a new Tweet corpus reader for a set of documents
    located at the given root directory.
    If you want to work directly with the raw Tweets, the `json` library can
    be used::
       import json
       for tweet in reader.docs():
           print(json.dumps(tweet, indent=1, sort_keys=True))
    """

    CorpusView = StreamBackedCorpusView  # The corpus view class used by this reader.

    def __init__(self, root, fileids=None, word_tokenizer=TweetTokenizer(), encoding='utf-8-sig', bullying_trace='bullying_trace'):

        CorpusReader.__init__(self, root, fileids, encoding, bullying_trace)

        for path in self.abspaths(self._fileids):
            if isinstance(path, ZipFilePathPointer):
                pass
            elif os.path.getsize(path) == 0:
                raise ValueError("File {} is empty".format(path))
        """Check that all user-created corpus files are non-empty."""

        self._word_tokenizer = word_tokenizer
        self._bullying_trace = bullying_trace
    
    def _read_tweets(self, stream):
        """
        Assumes that each line in ``stream`` is a JSON-serialised object.
        """
        tweets = []
        for i in range(100):
            line = stream.readline()
            if not line:
                return tweets
            tweet = json.loads(line)
            tweets.append(tweet)
        return tweets

    def docs(self, fileids=None, bullying_trace=None):
        """
        Returns the full Tweet objects, 
        :return: the given file(s) as a list of dictionaries deserialised from JSON.
        :rtype: list(dict)
        """
        tweets = concat(
            [
                self.CorpusView(path, self._read_tweets, encoding=enc)
                for (path, enc, fileid) in self.abspaths(fileids, True, True)
            ])
        
        if bullying_trace:
            remove = [None,'remove', float('nan'), 'nan', 'None'] #'nan
            tweets = [tweet for tweet in tweets if str(tweet[self._bullying_trace]) not in remove]

        return tweets 

           
    def sizes(self, fileids=None):
        """
        Returns a list of tuples, the fileid and size on disk of the file.
        This function is used to detect oddly large files in the corpus.
        """
        # Create a generator, getting every path and computing filesize
        for path in self.abspaths(fileids):
            yield os.path.getsize(path)

    def fields(self, fields, fileids=None):
        """
        extract particular fields from the json doc. Can be string or an 
        iterable of fields. If just one fields in passed in, then the values 
        are returned, otherwise dictionaries of the requested fields returned
        """
        if isinstance(fields, string_types):
            fields = [fields,]

        if len(fields) == 1:
            for doc in self.docs(fileids):
                if fields[0] in doc:
                    yield doc[fields[0]]

        else:
            for doc in self.docs(fileids):
                yield {
                    key : doc.get(key, None)
                    for key in fields
                }
    
    def strings(self, fileids=None):
        """
        Returns only the text content of Tweets in the file(s)
        :return: the given file(s) as a list of Tweets.
        :rtype: list(str)
        """
        fulltweets = self.docs(fileids)
        tweets = []
        for jsono in fulltweets:
            try:
                text = jsono['full_tweet']
                if isinstance(text, bytes):
                    text = text.encode('latin-1').decode('utf-8') #.decode(self.encoding) #
                tweets.append(text)
            except KeyError:
                pass
        return tweets

    def tokenized(self, fileids=None):
        """
        :return: the given file(s) as a list of the text content of Tweets as
        as a list of words, screenanames, hashtags, URLs and punctuation symbols.
        :rtype: list(list(str))
        """
        tweets = self.strings(fileids)
        tokenizer = self._word_tokenizer
        tokenized = [tokenizer.tokenize(t) for t in tweets]
        return [ pos_tag(token) for token in tokenized ]
    
    def process_tweet(self, fileids=None):

		# emoji_pattern = re.compile("["
        # u"\U0001F600-\U0001F64F"  # emoticons
        # u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        # u"\U0001F680-\U0001F6FF"  # transport & map symbols
        # u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        # u"\U00002702-\U000027B0"
        # u"\U000024C2-\U0001F251"
        # "]+", flags=re.UNICODE)
        tagged_tweets = self.tokenized(fileids)
        mod_tweets = []
        for tweet in tagged_tweets:
            mod_tweet=[]
            for (token, tag) in tweet:    
                if '@' in token:
                    token = '@user'
                elif '#' in token:
                    token = token[1:]
                elif ('http' or 'https') in token:
                    token = 'UR'
                elif token == 'luv':
                    token = 'love'
                else:
                    pass
                mod_tweet.append((token,tag))
            mod_tweets.append(mod_tweet)
        return mod_tweets

    def raw(self, fileids=None):
        """
        Return the corpora in their raw form.
        """
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]
        return concat([self.open(f).read() for f in fileids])

    def describe(self, fileids=None):
        """
        Performs a single pass of the corpus and
        returns a dictionary with a variety of metrics
        concerning the state of the corpus.
        """
        started = time.time()
        # Structures to perform counting.
        counts  = nltk.FreqDist()
        tokens  = nltk.FreqDist()

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.fileids())

        # Perform single pass over paragraphs, tokenize and count
        for tweet in self.tokenized(fileids):
            counts['tweets'] += 1

            for word in tweet:
                counts['words'] += 1
                tokens[word] += 1
            # print(counts, tokens)

        # Return data structure with information
        return {
            'files':  n_fileids,
            'tweets':  counts['tweets'],
            'words':  counts['words'],
            'vocab':  len(tokens),
            'lexdiv': float(counts['words']) / float(len(tokens)),
            'twdoc':  float(counts['tweets']) / float(n_fileids),
            'mins':   (time.time() - started)/60,
        }
    
    def describes(self, fileids=None, categories=None):
        """
        Returns a string representation of the describe command.
        """
        return (
            "This twitter corpus contains {files} files.\n"
            "Structured as:\n"
            "    {tweets} tweets ({twdoc:0.3f} mean tweets per file)\n"
            "    Word count of {words} with a vocabulary of {vocab}\n"
            "    ({lexdiv:0.3f} lexical diversity).\n"
            "Corpus scan took {mins:0.3f} minutes."
        ).format(**self.describe(fileids))
