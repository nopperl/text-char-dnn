#!/bin/sh
DATA_DIR=data

echo Processing Blog Authorship corpus...
./blogs.py -d DATA_DIR

echo Processing Enron Email Dataset...
./enron.py -d DATA_DIR

echo Processing PAN13 dataset...
./pan13.py -d DATA_DIR -m tr -l en
./pan13.py -d DATA_DIR -m te -l en

echo Processing PAN14 dataset...
./pan14.py -d DATA_DIR -l en