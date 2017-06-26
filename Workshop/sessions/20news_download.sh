#!/bin/bash
mkdir -p ../../data/20news
pushd ../../data/20news
wget http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
tar -xvf 20news-bydate.tar.gz
popd
