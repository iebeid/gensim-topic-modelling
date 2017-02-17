try:
    from gensim import corpora, models, similarities
    from collections import defaultdict
    from pprint import pprint
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
except ImportError:
    print "not installed"

#general

documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]

stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
        for text in texts]
pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('D:/sources/PycharmProjects/TopicModelling/tmp/deerwester.dict')
print(dictionary)
print(dictionary.token2id)

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('D:/sources/PycharmProjects/TopicModelling/tmp/deerwester.mm', corpus)
print(corpus)

tfidf = models.TfidfModel(corpus)
vec = [(0, 1), (4, 1)]
print(tfidf[vec])
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)
sims = index[tfidf[vec]]
print(list(enumerate(sims)))

#streaming

# class MyCorpus(object):
#     def __iter__(self):
#         for line in open('D:/sources/PycharmProjects/TopicModelling/data/mycorpus.txt'):
#             yield dictionary.doc2bow(line.lower().split())
#
# corpus_memory_friendly = MyCorpus()
# print(corpus_memory_friendly)
#
# for vector in corpus_memory_friendly:
#     print(vector)
#
# dictionary = corpora.Dictionary(line.lower().split() for line in open('D:/sources/PycharmProjects/TopicModelling/data/mycorpus.txt'))
# stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
#             if stopword in dictionary.token2id]
# once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
# dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
# dictionary.compactify() # remove gaps in id sequence after words that were removed
# print(dictionary)


#similiartiy

dictionary = corpora.Dictionary.load('D:/sources/PycharmProjects/TopicModelling/tmp/deerwester.dict')
corpus = corpora.MmCorpus('D:/sources/PycharmProjects/TopicModelling/tmp/deerwester.mm')
print(corpus)

ldamodel = models.LdaModel(corpus, id2word=dictionary, num_topics=2)

doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lda = ldamodel[vec_bow]
print(vec_lda)

index = similarities.MatrixSimilarity(ldamodel[corpus],num_features=2)

sims = index[vec_lda]
print(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims)

# transformation

dictionary = corpora.Dictionary.load('D:/sources/PycharmProjects/TopicModelling/tmp/deerwester.dict')
corpus = corpora.MmCorpus('D:/sources/PycharmProjects/TopicModelling/tmp/deerwester.mm')
print(corpus)

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lsi = lsi[corpus_tfidf]
lsi.print_topics(2)

for doc in corpus_lsi:
    print(doc)

lsi.save('D:/sources/PycharmProjects/TopicModelling/tmp/model.lsi')
lsi = models.LsiModel.load('D:/sources/PycharmProjects/TopicModelling/tmp/model.lsi')


ldamodel = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=200)
corpus_lda = ldamodel[corpus_tfidf]
ldamodel.print_topics(2)

for doc in corpus_lda:
    print(doc)