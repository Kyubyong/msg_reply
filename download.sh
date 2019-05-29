#!/bin/bash

echo "Download and extract OpenSubtitles 2018 en-es parallel data"
echo "to opensubtitles2018"
wget http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-es.txt.zip -O temp.zip;
unzip temp.zip -d opensubtitles2018/;
rm temp.zip

echo "Download Cornell Movie Dialogue Corpus"
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip -O temp.zip;
unzip temp.zip;
rm temp.zip

