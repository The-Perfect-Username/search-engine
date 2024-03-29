import os
import math
import itertools
from nltk.stem.porter import *
import xml.etree.ElementTree as ET

class BowDocument:

    def __init__(self, docId):
        self.stemmer = PorterStemmer()
        self.docId = docId
        self.term_freq_map = {}
        self.doclen = 0
        self.denom = 0
        self.bm25 = 0

    def get_freq_word_map(self):
        return self.term_freq_map

    def add_term(self, term):
        self.term_freq_map[term] = 1

    def term_count(self, term):
        term = self.stem_word_by_snowball(term)
        if term not in self.get_stop_words():
            if term in self.term_freq_map:
                self.doclen += 1
                self.term_freq_map[term] += 1
            else:
                self.doclen += 1
                self.add_term(term)

    def get_doc_len(self):
        return self.doclen

    def get_stop_words(self):
        stop_words_file = open("./documents/stop-words/stop-words.txt", "r")
        stop_words = stop_words_file.read().split(',')
        stop_words_file.close()
        return stop_words

    # Stem the terms
    def stem_word_by_snowball(self, word):
        return self.stemmer.stem(word)

    def get_doc_length(self):
        return self.doclen

    def get_docId(self):
        return self.docId

    def set_bm25(self, bm25):
        self.bm25 = bm25

    def get_bm25(self,):
        return self.bm25
