import os
import re
import string
import itertools
from nltk.stem.porter import *
import xml.etree.ElementTree as ET


class SearchEngine:

    def __init__(self):
        self.path = './documents'
        self.dataset_path = '/dataset/'
        self.stop_word_path = '/stop-words/'
        self.topic_assignment = '/topic-assignment/'
        self.topic_statement = '/topic-statement/'
        self.queries = {}

    def get_query_terms(self):
        path = self.path + self.topic_statement
        directory = os.listdir(path)

        for _file_ in directory:
            self.find_query_terms(path + _file_)

        for i in self.queries:
            print("{} {} \n".format(i, self.queries[i]))

    def find_query_terms(self, directory):
        with open(directory, 'r') as f:
            for line in f:
                if "<num>" in line:
                    ID = re.findall(r'\bR\w+', line)[0]
                if "<title>" in line:
                    tokens = re.findall(r'<title>(.*)', line)[0]
                    tokens = [word.strip(string.punctuation) for word in tokens.split(" ") if word]
                    self.queries[ID] = tokens
        f.close()

    def tfidf(self, number_of_tokens, number_of_documents, number_of_docs_with_term):
        tf = number_of_tokens / number_of_documents
        idf = math.log((number_of_documents / number_of_docs_with_term))
        return tf * idf

SE = SearchEngine()

SE.get_query_terms()
