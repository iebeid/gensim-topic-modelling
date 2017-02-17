try:
    from gensim import corpora, models, similarities
    from collections import defaultdict
    from pprint import pprint
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
except ImportError:
    print "not installed"

dictionary = corpora.Dictionary.load('D:/sources/PycharmProjects/TopicModelling/tmp/newsgroups.dict')
corpus = corpora.MmCorpus('D:/sources/PycharmProjects/TopicModelling/tmp/newsgroups.mm')
print(corpus)

ldamodel = models.LdaModel(corpus, id2word=dictionary, num_topics=200, passes=20)
corpus_lda = ldamodel[corpus]
ldamodel.save('D:/sources/PycharmProjects/TopicModelling/tmp/lda.model')
ldamodel.print_topics(num_topics=200, num_words=20)

