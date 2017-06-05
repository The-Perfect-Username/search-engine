import os
import re
import math
import string
import itertools
from nltk.stem.porter import *
import xml.etree.ElementTree as ET

class InformationFilter:

    def __init__(self, setId):
        self.setId = setId
        self.documents = {}
        self.total_len = 0
        self.total_len_avg = 0
        self.shared_terms = {}
        self.term_weighting = {}
        self.bm25 = {}
        self.bm25_scores = {}

        self.relevant = {}
        self.docs = {}

    # Set relevant document and their freq word map
    def set_relevant(self, docId, data):
        self.relevant[docId] = self.documents[docId].get_freq_word_map()

    # Add documents
    def add_document(self, docId, document):
        self.documents[docId] = document

    # Start Information Filtering process
    def start(self):
        self.bm25_term_weighting()
        self.ranking()
        self.create_results()

    # Tokenise keywords into an array
    def read_lines_to_array(self, directory):
        tmp = []
        with open(directory, 'r') as f:
            for line in f:
                tmp.append(line)
        return tmp

    # BM25 term weighting algorithm
    def bm25_term_weighting(self):
        T = set()
        N = len(self.documents)
        R = len(self.relevant)
        tmp = {}

        for doc in self.relevant:
            for term in self.relevant[doc]:
                T = T | set([term])

        for term in T:
            tmp[term] = {"n": 0, "r": 0}

        for term in T:
            for doc in self.documents:
                if term in self.documents[doc].get_freq_word_map():
                    tmp[term]['n'] += 1

        for term in T:
            for doc in self.relevant:
                if term in self.relevant[doc]:
                    tmp[term]['r'] += 1

        for term in T:
            n = tmp[term]['n']
            r = tmp[term]['r']
            self.bm25[term] = self.w5(n, r, N, R)


    # Term weighting
    def w5(self, n, r, N, R):
        one = (r + 0.5) / (R - r + 0.5)
        two = (n - r + 0.5) / ((N - n) - (R - r) + 0.5)
        return math.log(one / two)

    # Ranking algorithm
    def ranking(self):
        tmp_bm25 = {}
        for docId in self.documents:
            tmp = {}
            for term in self.documents[docId].get_freq_word_map():
                if term in self.bm25:
                    tmp[term] = self.bm25[term]
                else:
                    tmp[term] = 0

            tmp_bm25[docId] = tmp

        for docId in tmp_bm25:
            self.bm25_scores[docId] = self.count_score(tmp_bm25[docId])

        self.bm25_scores = dict(self.sort_dict(self.bm25_scores))

    # Sort dictionary in descending order
    def sort_dict(self, dictionary):
        return sorted(dictionary.items(), key=lambda x:x[1], reverse=True)

    def count_score(self, dictionary):
        score = 0
        for i in dictionary:
            score += dictionary[i]
        return score
        
    # Get the top 5 relevant documents
    def get_top_five(self):
        return dict(list(self.bm25_scores.items())[:5])

    def create_results(self):
        directory = "./documents/results/"
        filename = "result{}.dat".format(int(self.setId) - 100)
        f = open(directory + filename, "w")
        __bm25_scores = self.bm25_scores
        for scores in __bm25_scores:
            text = "{} {} \n".format(scores, __bm25_scores[scores])
            f.write(text)
        f.close()
