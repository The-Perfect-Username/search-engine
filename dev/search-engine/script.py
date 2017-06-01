import os
import re
import math
import string
import itertools
from nltk.stem.porter import *
import xml.etree.ElementTree as ET
from bowdocument import BowDocument
from training import TrainingSet



class SearchEngine:

    def __init__(self):
        self.path = './documents'
        self.dataset_path = '/dataset/'
        self.stop_word_path = '/stop-words/'
        self.topic_assignment = '/topic-assignment/'
        self.topic_statement = '/topic-statement/'
        self.stemmer = PorterStemmer()
        self.queries = {}
        self.query_freq = {}
        self.data_dict = {} # key => Set ID; value => list of xml files
        self.document_dict = {} # key => document ID; value => BD object
        self.search_dict = {} # key => Set ID; value => {key => document ID; value => BD document}
        self.bm25_dict = {}
        self.training_sets = {}
        self.total_doc_len = 0
        self.doc_len_avg = 0
        self.k1 = 1.2
        self.k2 = 100
        self.b = 0.75


    # Query Documents Code

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
            TS.set_query(self.queries[query])
            self.add_training_sets(setId, TS)

    # Just for testing
    def add_training_sets(self, setId, tsobj):
        self.training_sets[setId] = tsobj

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

    def get_document_terms(self):
            path = self.path + self.dataset_path + "Training"
            count = 0
            for a in self.data_dict:
                if a == "119" or a == "120"or a == "121":
                    for target in self.data_dict[a]:
                        try:
                            TS = self.training_sets.get(a)
                            target_file = path + a + "/" + target
                            root = ET.parse(target_file).getroot()
                            itemId = root.get('itemid')
                            # Create new BowDocument Object
                            BD = BowDocument(itemId)
                            for text in root.iter('text'):
                                for p in text.iter('p'):
                                    #tokenise text into array
                                    terms = [word.strip(string.punctuation) for word in p.text.split(" ")]
                                    for token in terms:
                                        #add terms to BowDocument List
                                        token = re.sub(r'[^a-zA-Z]', '', token)
                                        if not re.search(r'[a-zA-Z]', token) == None:
                                            BD.term_count(token)

                            self.document_dict[itemId] = BD
                            TS.add_document(itemId, BD)
                        except PermissionError:
                            pass
                    self.search_dict[a] = self.document_dict


    def sort_dict(self, dictionary):
        return sorted(dictionary.items(), key=lambda x:x[1], reverse=True)

    def to_dict(self, array):
        return dict(array)

    def sort_bow_document(self, BD):
        return sorted(BD.get_tfidf().items(), key=lambda x:x[1], reverse=True)


    def start(self):
        self.get_query_terms()
        self.create_training_sets()
        self.get_documents()
        self.get_document_terms()
        for i in self.training_sets:
            ts = self.training_sets[i]
            docs = ts.get_documents()
            print ("********************* {} ***********************".format(i))
            if i == "119" or i == "120" or i == "121":
                shared_terms = ts.count_shared_terms()
                for doc in docs:
                    print (ts.BM25(docs[doc], shared_terms))
                print ("\n")

SE = SearchEngine()

SE.start()
