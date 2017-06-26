#!/bin/bash
mkdir -p ../../data/glove
pushd ../../data/glove
#wget http://nlp.stanford.edu/data/glove.6B.zip
#tar -xvf glove.6B.zip
unzip glove.6B.zip
popd
