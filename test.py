try:
    from gensim import corpora, models, similarities
    from collections import defaultdict
    from pprint import pprint
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
except ImportError:
    print "not installed"

#testing
dictionary = corpora.Dictionary.load('D:/sources/PycharmProjects/TopicModelling/tmp/newsgroups.dict')
corpus = corpora.MmCorpus('D:/sources/PycharmProjects/TopicModelling/tmp/newsgroups.mm')
print(corpus)
ldamodel = models.LdaModel.load('D:/sources/PycharmProjects/TopicModelling/tmp/lda.model')
doc = "A program in the archive keymap00.zip on simtel and mirror sites in the msdos/keyboard directory will do this.  It is written in assembler and it best if you have a compiler to create a new keyboard map.  It is possible, however, to use a binary editor to edit the provided compiled keyboard driver if you do not have a compiler.  I used hexed100.zip, also available on simtel.  Simply serach for the codes 00 01 02 03 to locate the biginning of the 'normal' keyboard map.  Then swap the codes for the keys that you wish to swap.  See the keyboard directory of simtel for programs that report the scancode for each key to you (some bios programs also have this info)"
vec_lda = ldamodel[dictionary.doc2bow(doc.lower().split())]
print(vec_lda)
