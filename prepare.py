
import logging
import os
import sys
import re
import tarfile
import itertools

import nltk
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

from textblob import TextBlob

import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models, similarities
from gensim.parsing import PorterStemmer
from collections import defaultdict
from pprint import pprint

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

my_corpus = "D:/data/20_newsgroups/20news-19997.tar.gz"

def process_message(message):
    message = gensim.utils.to_unicode(message, 'latin1').strip()
    blocks = message.split(u'\n\n')
    content = u'\n\n'.join(blocks[1:-1])
    return content

def iter_20newsgroups(fname, log_every=None):
    extracted = 0
    with tarfile.open(fname, 'r:gz') as tf:
        for file_number, file_info in enumerate(tf):
            if file_info.isfile():
                if log_every and extracted % log_every == 0:
                    logging.info("extracting 20newsgroups file #%i: %s" % (extracted, file_info.name))
                content = tf.extractfile(file_info).read()
                yield process_message(content)
                extracted += 1

def best_ngrams(words, top_n=1000, min_freq=100):
    tcf = TrigramCollocationFinder.from_words(words)
    tcf.apply_freq_filter(min_freq)
    trigrams = [' '.join(w) for w in tcf.nbest(TrigramAssocMeasures.chi_sq, top_n)]
    logging.info("%i trigrams found: %s..." % (len(trigrams), trigrams[:20]))
    bcf = tcf.bigram_finder()
    bcf.apply_freq_filter(min_freq)
    bigrams = [' '.join(w) for w in bcf.nbest(BigramAssocMeasures.pmi, top_n)]
    logging.info("%i bigrams found: %s..." % (len(bigrams), bigrams[:20]))
    pat_gram2 = re.compile('(%s)' % '|'.join(bigrams), re.UNICODE)
    pat_gram3 = re.compile('(%s)' % '|'.join(trigrams), re.UNICODE)
    return pat_gram2, pat_gram3

class Corpus20News(object):
    def __init__(self, fname):
        self.fname = fname
        logging.info("collecting ngrams from %s" % self.fname)
        documents = (self.split_words(text) for text in iter_20newsgroups(self.fname, log_every=1000))
        words = itertools.chain.from_iterable(documents)
        self.bigrams, self.trigrams = best_ngrams(words)

    def split_words(self, text, stopwords=STOPWORDS):
        #PorterStemmer().stem(word)
        return [word
                for word in gensim.utils.tokenize(text, lower=True)
                if word not in STOPWORDS and len(word) > 3]

    def tokenize(self, message):
        text = u' '.join(self.split_words(message))
        text = re.sub(self.trigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        text = re.sub(self.bigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        return text.split()

    def __iter__(self):
        for message in iter_20newsgroups(self.fname):
            yield self.tokenize(message)

collocations_corpus = Corpus20News(my_corpus)

dictionary = corpora.Dictionary(collocations_corpus)
dictionary.save('D:/sources/PycharmProjects/TopicModelling/tmp/newsgroups.dict')

corpus = [dictionary.doc2bow(text) for text in collocations_corpus]
corpora.MmCorpus.serialize('D:/sources/PycharmProjects/TopicModelling/tmp/newsgroups.mm', corpus)