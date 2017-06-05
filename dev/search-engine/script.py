import os
import re
import math
import string
import itertools
from nltk.stem.porter import *
import xml.etree.ElementTree as ET
from bowdocument import BowDocument
from training import TrainingSet
from infofilter import InformationFilter
from Evaluation import Evaluator
import time



class SearchEngine:

    def __init__(self):
        self.path = './documents'
        self.dataset_path = '/dataset/'
        self.stop_word_path = '/stop-words/'
        self.topic_assignment = '/topic-assignment/'
        self.topic_statement = '/topic-statement/'
        self.stemmer = PorterStemmer()
        self.queries = {}
        self.data_dict = {} # key => Set ID; value => list of xml files
        self.document_dict = {} # key => document ID; value => BD object
        self.search_dict = {} # key => Set ID; value => {key => document ID; value => BD document}
        self.training_sets = {}
        self.information_filters = {}


    # Query Documents Code
    def get_stop_words(self):
        return open(self.path + self.stop_word_path + "stop-words.txt").read().split(',')

    ## Gets the query terms from the documents in
    ## the topic-statement folder and stores the
    ## terms into a dictionary
    def get_query_terms(self):
        path = self.path + self.topic_statement
        directory = os.listdir(path)

        for _file_ in directory:
            self.find_query_terms(path + _file_)

    def create_training_sets(self):
        for query in self.queries:
            setId = query.replace('R', '')
            TS = TrainingSet(setId)
            IF = InformationFilter(setId)
            TS.set_query(self.queries[query])
            self.add_training_sets(setId, TS)
            self.add_info_filters(setId, IF)

    # Just for testing
    def add_training_sets(self, setId, tsobj):
        self.training_sets[setId] = tsobj

    def add_info_filters(self, setId, ifobj):
        self.information_filters[setId] = ifobj

    ## Parses the document files for the document ID
    ## and for the query terms
    def find_query_terms(self, directory):
        with open(directory, 'r') as f:
            for line in f:
                if "<num>" in line:
                    ID = re.findall(r'\bR\w+', line)[0]
                if "<title>" in line:
                    tokens = re.findall(r'<title>(.*)', line)[0]
                    tokens = [word.strip(string.punctuation) for word in tokens.split(" ") if word]
                    tokens = [re.sub(r'[^a-zA-Z]', '', word) for word in tokens]
                    tokens = [self.stemmer.stem(word) for word in tokens]
                    self.queries[ID] = tokens
        f.close()

    # Documents Code

    def get_documents(self):
        path = self.path + self.dataset_path
        directory = os.listdir(path)

        for __f__ in directory:
            files = os.listdir(path + __f__)
            number = __f__.replace("Training", '')
            self.data_dict[number] = self.trunc_files(files)

    # Removes all xml files that only work on Mac iOS
    def trunc_files(self, target):
        doclist = []
        for __d__ in target:
            if not "._" in __d__:
                doclist.append(__d__)
        return doclist

    def process(self):
        # Loop through each set of documents
        for documents in self.data_dict:
            TS = self.training_sets[documents]
            IF = self.information_filters[documents]

            for document in self.data_dict[documents]:
                self.process_document(documents, document)

            shared_terms = TS.count_shared_terms()

            docs = TS.get_documents()

            tmp_dict = {}
            for doc in docs:
                tmp_dict[doc] = TS.BM25(docs[doc], shared_terms)


            _sorted_ = self.to_dict(self.sort_dict(tmp_dict))

            filename = "BaselineResult{}.dat".format(int(documents) - 100)
            self.create_file(filename, documents, _sorted_)

            IF.start()

    def process_document(self, setId, filename):
        try:
            path = self.path + self.dataset_path + "Training"
            sw = self.get_stop_words()
            TS = self.training_sets.get(setId)
            IF = self.information_filters.get(setId)

            target_file = path + setId + "/" + filename

            itemId = filename.replace(".xml", '')
            # Create new BowDocument Object
            BD = BowDocument(itemId)
            for line in ET.parse(target_file).getroot().iter('p'):
                #tokenise text into array
                terms = line.text.split(" ")
                # Remove stopwords and puntuation
                for term in terms:
                    term = re.sub(r'[^a-zA-Z]', '', term)
                    if re.search(r'[a-z]+', term) and term.lower() not in sw and not term == '\'s' and len(term) > 2:
                         BD.term_count(term)

            self.document_dict[itemId] = BD

            TS.add_document(itemId, BD)
            IF.add_document(itemId, BD)

        except PermissionError:
            pass

    # Sort dictionary
    def sort_dict(self, dictionary):
        return sorted(dictionary.items(), key=lambda x:x[1], reverse=True)

    # Covnert array to dictionary
    def to_dict(self, array):
        return dict(array)

    # Create result files for the training set
    def create_file(self, name, setId, data):
        directory = "./documents/training-set/"
        f = open(directory + name, "w")

        # Call training set and information filter objects
        IF = self.information_filters.get(setId)
        TS = self.training_sets.get(setId)

        data = list(data.items())

        # Top 5 Relevant
        for i in data[:5]:
            text = "{} {}\n".format(i[0], i[1])
            f.write(text)
            document = TS.get_documents().get(i[0])
            IF.set_relevant(i[0], document)

        # Noise
        for i in data[5: (len(data) - 5)]:
            text = "{} {}\n".format(i[0], i[1])
            f.write(text)

        # Bottom 5 nonrelevant
        for i in data[-5:]:
            text = "{} {}\n".format(i[0], i[1])
            f.write(text)
            document = TS.get_documents().get(i[0])
        f.close()

    def start(self):
        self.get_query_terms()
        self.create_training_sets()
        self.get_documents()
        self.process()

        for documents in self.data_dict:
            E = Evaluator(documents)
            E.startProcess()


SE = SearchEngine()

SE.start()
