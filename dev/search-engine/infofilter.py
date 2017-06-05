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
        self.doc_set = {}
        self.bm25 = {}
        self.bm25_scores = {}

        self.relevant = {}
        self.nonrelevant = {}
        self.noise = {}
        self.docs = {}

    def read_files(self):
        path = "./documents/topic-assignment/Training{}.txt".format(self.setId)
        with open(path, 'r') as f:
            for line in f:
                tokens = line.split(" ")
                self.docs[tokens[1]] = tokens[2].replace("\n", '')
        f.close()

    def set_noise(self):
        for document in self.documents:
            if document not in self.relevant and document not in self.nonrelevant:
                self.noise[document] = self.documents[document].get_freq_word_map()

    def set_relevant(self, docId, data):
        self.relevant[docId] = self.documents[docId].get_freq_word_map()


    def set_nonrelevant(self, docId, data):
        self.nonrelevant[docId] = self.documents[docId].get_freq_word_map()

    def add_document(self, docId, document):
        self.documents[docId] = document

    def start(self):
        self.read_files()
        self.get_document_keywords()
        self.bm25_term_weighting()
        self.ranking()
        self.create_results()

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
        print ("N: {} R: {}".format(N, R))
        tmp = {}

        for doc in self.relevant:
            for term in self.relevant[doc]:
                T = T | set([term])

        for term in T:
            tmp[term] = {"n": 0, "r": 0}

        for term in T:
            for doc in self.doc_set:
                if term in self.doc_set[doc]:
                    tmp[term]['n'] += 1

        for term in T:
            for doc in self.relevant:
                if term in self.relevant[doc]:
                    tmp[term]['r'] += 1

        for term in T:
            n = tmp[term]['n']
            r = tmp[term]['r']
            self.bm25[term] = self.w5(n, r, N, R)

    def w5(self, n, r, N, R):
        one = (r + 0.5) / (R - r + 0.5)
        two = (n - r + 0.5) / ((N - n) - (R - r) + 0.5)
        return math.log(one / two)

    def ranking(self):
        tmp_bm25 = {}
        for docId in self.doc_set:
            tmp = {}
            for term in self.doc_set[docId]:
                if term in self.bm25:
                    tmp[term] = self.bm25[term]
                else:
                    tmp[term] = 0

            tmp_bm25[docId] = tmp

        for docId in tmp_bm25:
            self.bm25_scores[docId] = self.count_score(tmp_bm25[docId])


    def count_score(self, dictionary):
        score = 0
        for i in dictionary:
            score += dictionary[i]
        return score


    def create_results(self):
        directory = "./documents/results/"
        filename = "result{}.dat".format(int(self.setId) - 100)
        f = open(directory + filename, "w")
        __bm25_scores = self.bm25_scores
        for scores in __bm25_scores:
            text = "{} {} \n".format(scores, __bm25_scores[scores])
            f.write(text)
        f.close()
