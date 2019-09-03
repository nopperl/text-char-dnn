#!/bin/sh
DATA_DIR=data
RAW_DIR=${DATA_DIR}/raw
mkdir -p $RAW_DIR
cd $RAW_DIR

# Blog Authorship Corpus
echo Retrieving Blog Authorship Corpus...
wget http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip
unzip blogs.zip
rm blogs.zip

# Enron
echo Retrieving Enron Email Dataset...
wget https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz
tar xzf enron_mail_20150507.tar.gz
mv maildir enron
wget https://www.inf.ed.ac.uk/teaching/courses/tts/assessed/roles.txt -O enron_roles.txt

# PAN13
echo Retrieving PAN13 dataset...
wget https://www.uni-weimar.de/medien/webis/corpora/corpus-pan-labs-09-today/pan-13/pan13-data/pan13-author-profiling-training-corpus-2013-01-09.zip
unzip pan13-author-profiling-training-corpus-2013-01-09.zip
mv pan13-author-profiling-training-corpus-2013-01-09 pan13_tr
rm pan13-author-profiling-training-corpus-2013-01-09.zip
wget https://www.uni-weimar.de/medien/webis/corpora/corpus-pan-labs-09-today/pan-13/pan13-data/pan13-author-profiling-test-corpus2-2013-04-29.zip
unzip pan13-author-profiling-test-corpus2-2013-04-29.zip
mv pan13-author-profiling-test-corpus2-2013-04-29 pan13_te
rm pan13-author-profiling-test-corpus2-2013-04-29.zip

# PAN14
echo Retrieving PAN14 dataset...
wget https://www.uni-weimar.de/medien/webis/corpora/corpus-pan-labs-09-today/pan-14/pan14-data/pan14-author-profiling-training-corpus-2014-04-16.zip
unzip pan14-author-profiling-training-corpus-2014-04-16.zip
unzip ${RAW_DIR}/pan14-author-profiling-training-corpus-2014-04-16/pan14-author-profiling-training-corpus-english-blogs-2014-04-16.zip
unzip ${RAW_DIR}/pan14-author-profiling-training-corpus-2014-04-16/pan14-author-profiling-training-corpus-spanish-blogs-2014-04-16.zip
mv pan14-author-profiling-training-corpus-2014-04-16/pan14-author-profiling-training-corpus-english-blogs-2014-04-16/mnt/nfs/tira/data/pan14-training-corpora-truth/pan14-author-profiling-training-corpus-english-blogs-2014-04-16 pan14_en
mv pan14-author-profiling-training-corpus-2014-04-16/pan14-author-profiling-training-corpus-spanish-blogs-2014-04-16/mnt/nfs/tira/data/pan14-training-corpora-truth/pan14-author-profiling-training-corpus-spanish-blogs-2014-04-16 pan14_es
rm -rf pan14-author-profiling-training-corpus-2014-04-16
rm pan14-author-profiling-training-corpus-2014-04-16.zip