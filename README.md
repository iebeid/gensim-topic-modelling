# Topic modelling algorithm evaluation over differentpackages

On this resport we are going to work with Mallet[1] and GenSim[2] and compare them to see
how good or bad both work on the topic modelling task. We also want to see how data should
be prepared for each tool and see how easy is to get each app up and running.
For both reports we are going to use as an input data the 20 newsgroups data set[3]; This
data set contains data from different topics like hardware, computers, motorcycles, politics, electronics,
religion among others. The idea of this project is to train a topic model with all the
datasets together, experiment with stop words, number of topics and then analyze the extracted
topics and display the top 20 words that have the highest probabilities for each topic.
Latent Dirichlet Allocation (LDA) is a topic model that generates topics based on word
frequency from a set of documents. LDA is particularly useful for finding reasonably accurate
mixtures of topics within a given document set.
We will use Latent Dirichlet Allocation to perform topic modeling and discover semantic
structure of the provided dataset, by examining word statistical co-occurrence patterns within
the provided dataset titled 20 Newsgroups which consists of 20000 messages taken from 20 newsgroups.
The messages are provided by the School of Computing at Carnegie Mellon University
which dates back to 1999. The 20000 messages represent 1000 use net articles with approximately
4% of the articles cross posted, they are typical postings thus they have headers with
subject lines, signature files and quoted portions of other articles, also each newsgroup is stored
in a subdirectory, with each article stored as a separate file.
This dataset comes organized by date, and also comes with some noise (like the “From”,
“Subject” headers on each post as mentioned before and some typos in the dataset aswell); some
of the topics are very closely related to each other like Hardware MAC and Hardware PC, while
other topics are very dissimilar like christian topics and motorcycles.
The unsupervised approach provided with LDA allow expressing the documents in the new
semantic representation and queried for topical similarity against other documents, this approach
is based mainly on creating a term document matrix through performing singular value decomposition,
and identifying the occurrences of a certain terms and assigning weights.
Some of the advantages of LDA can be summarized by the following: simple model based
on linear algebra, term weights not binary, allows computing a continuous degree of similarity
between queries and documents, allows ranking documents according to their possible relevance,
Allows partial matching.
The limitations of LDA includes: long documents are poorly represented because they have
1
poor similarity values, search keywords must precisely match document terms; word substrings
might result in a ”false positive match”, semantic sensitivity; documents with similar context but
different term vocabulary won’t be associated, resulting in a ”false negative match”, the order in
which the terms appear in the document is lost in the vector space representation, theoretically
assumes terms are statistically independent and last weighting is intuitive but not very formal.
After getting the results we plan to compare both Mallet and GenSim against each other and
see how different or similar both topics end up being trained.

