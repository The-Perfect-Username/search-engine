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
        self.relevant = {}
        self.nonrelevant = {}
        self.total_len = 0
        self.total_len_avg = 0
        self.shared_terms = {}
        self.term_weighting = {}
        self.doc_set = {}
        self.bm25 = {}
        self.bm25_scores = {}


    def set_relevant(self, key, value):
        self.relevant[key] = value

    def set_nonrelevant(self, key, value):
        self.nonrelevant[key] = value

    def start(self):
        self.get_document_keywords()
        self.bm25_term_weighting()
        self.ranking()

    # Passes the sets document dictionary
    def get_document_keywords(self):
        self.doc_set = {**self.relevant, **self.nonrelevant}

    def read_lines_to_array(self, directory):
        tmp = []
        with open(directory, 'r') as f:
            for line in f:
                tmp.append(line)
        return tmp

    def bm25_term_weighting(self):
        T = set()
        N = len(self.doc_set)
        R = len(self.relevant)
        tmp = {}

        for d in self.relevant:
            for t in self.relevant[d]:
                T = T | set([t])

        for t in T:
            tmp[t] = {"n": 0, "r": 0}

        for t in T:
            for d in self.doc_set:
                if t in self.doc_set[d]:
                    tmp[t]['n'] += 1
        for t in T:
            for d in self.relevant:
                if t in self.relevant[d]:
                    tmp[t]['r'] += 1

        for t in T:
            n = tmp[t]['n']
            r = tmp[t]['r']
            self.bm25[t] = self.w5(n, r, N, R)

    def w5(self, n, r, N, R):
        one = (r + 0.5) / (R - r + 0.5)
        two = (n - r + 0.5) / ((N - n) - (R - r) + 5)
        return math.log(one / two)

    def ranking(self):
        for docId in self.doc_set:
            tmp = {}
            for term in self.doc_set[docId]:
                if term in self.bm25:
                     tmp[term] = float(self.doc_set[docId][term]) * float(self.bm25[term])
            self.bm25_scores[docId] = tmp
            print (self.bm25_scores)
