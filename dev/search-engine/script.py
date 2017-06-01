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
        try:
            path = self.path + self.dataset_path + "Training"
            count = 0
            for a in self.data_dict:
                if count < 1:
                    for target in self.data_dict[a]:
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

                    self.search_dict[a] = self.document_dict
                    # self.search_dict[a] = self.document_dict
                    count += 1
                else:
                    break
        except PermissionError:
            pass


    def tfidf(self, number_of_tokens, number_of_documents, number_of_docs_with_term):
        tf = number_of_tokens / number_of_documents
        idf = math.log((number_of_documents / number_of_docs_with_term))
        return tf * idf

    ## Records the terms that are shared across several documents
    ## in the same dictionary
    def count_shared_terms(self, dictionary):
        # New dictionary to be created then sorted
        temp_dict = {}
        # Dictionary items
        dict_items = dictionary.items()
        # Document Id
        docId = next(iter(dictionary))
        # Dictionary of all frequent words
        freq_words = dictionary.get(docId).get_freq_word_map()
        for key, value in dict_items:

            if key != docId:
                doc_tokens = dictionary.get(key).get_freq_word_map()
                for token in doc_tokens:
                    # If the token exists, count by 1
                    if token in temp_dict:
                        temp_dict[token] += 1
                    else: # Add the token to the new dictionary
                        if (len(token) > 2): # Only add tokens with more than 2 characters
                            temp_dict[token] = 1
        # Sort the dictionary by the number of tokens and return the sorted dictionary
        return self.sort_dict(temp_dict)

    def sort_dict(self, dictionary):
        return sorted(dictionary.items(), key=lambda x:x[1], reverse=True)

    def to_dict(self, array):
        return dict(array)

    def sort_bow_document(self, BD):
        return sorted(BD.get_tfidf().items(), key=lambda x:x[1], reverse=True)

    def set_doc_len_avg(self, bd_dict):
        self.doc_len_avg = self.total_doc_len / len(bd_dict)

    def query_term_freq(self, terms):
        for term in terms:
            if term in self.query_freq:
                self.query_freq[term] += 1
            else:
                if len(term) > 2:
                    self.query_freq[term] = 1

    def K(self, doc_len):
        return self.k1 * ((1 - self.b) + (self.b * (doc_len / self.doc_len_avg)))

    # Calculate the BM25
    def BM25(self, query, docId, shared_terms):

        self.query_term_freq(query)

        document = self.document_dict.get(docId)

        freq_map = document.get_freq_word_map()

        sum_of_bm25 = 0.0

        __N__ = len(self.document_dict)
        __R__ = 0.0
        __r__ = 0.0
        __K__ = self.K(len(document.get_freq_word_map()))

        for term in query:

            __f__ = freq_map[term] if term in freq_map else 0.0
            __qf__ = self.query_freq[term] if term in self.query_freq else 0.0
            __n__ = shared_terms[term] if term in shared_terms else 0.0

            part1 = ( __r__ + 0.5 ) / ( __R__ - __r__ + 0.5 )
            part2 = ( __n__ - __r__ + 0.5 ) / ( __N__ - __n__ - __R__ + __r__ + 0.5 )
            part3 = (((self.k1 + 1 ) * __f__ ) / ( __K__ + __f__ ))
            part4 = (((self.k2 + 1) * __qf__ ) / ( self.k2 + __qf__ ))

            alg = (part1 / part2) * part3 * part4

            sum_of_bm25 += math.log(alg) if alg > 0 else 0.0

        return sum_of_bm25

    def start(self):
        self.get_query_terms()
        self.create_training_sets()
        self.get_documents()
        self.get_document_terms()
        for i in self.training_sets:
            b = self.training_sets[i].get_documents()
            print ("********************* {} ***********************".format(i))
            for x in b:
                print (b.get(x).get_freq_word_map())
                
        # for i in self.search_dict:
        #     key = "R" + str(i)
        #     query = self.queries[key]
        #
        #     bow_document_dict = self.search_dict[i]
        #     shared_terms = self.count_shared_terms(bow_document_dict)
        #     self.total_doc_len = self.get_total_doc_len(bow_document_dict)
        #     self.set_doc_len_avg(bow_document_dict)
        #     tmp_bm25_dict = {}
        #     print (i)
        #     print ("\n\n")
        #     for docId in bow_document_dict:
        #         print (docId)
        #         print ("\n")
        #         # print ("{} \n".format(docId))
        #         # tmp_bm25_dict[docId] = self.BM25(query, docId, shared_terms)
        #     # print ("Test {} \n".format(i))
        #     # self.bm25_dict[i] = tmp_bm25_dict
        #
        # for x in self.bm25_dict:
        #     print ("{} \n {} \n".format(x, self.bm25_dict.get(x)))



SE = SearchEngine()

SE.start()
