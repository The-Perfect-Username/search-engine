import os
import re
import math
import string
import itertools
from nltk.stem.porter import *
import xml.etree.ElementTree as ET
from bowdocument import BowDocument


class SearchEngine:

    def __init__(self):
        self.path = './documents'
        self.dataset_path = '/dataset/'
        self.stop_word_path = '/stop-words/'
        self.topic_assignment = '/topic-assignment/'
        self.topic_statement = '/topic-statement/'
        self.queries = {}
        self.data_dict = {}
        self.document_dict = {}
        self.total_doc_len = 0
        self.doc_len_avg = 0

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

    def get_docs(self):
        for __BD__ in self.document_dict:
            print(self.document_dict[__BD__].get_doc_length())

    def get_document_terms(self):
        try:
            path = self.path + self.dataset_path + "Training"
            count = 0
            for a in self.data_dict:
                if count < 3:
                    for target in self.data_dict[a]:

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
                        count+= 1
                else:
                    break
        except PermissionError:
            pass

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

    def some_shit(self, docId, number_of_shared_terms):
        BD = self.document_dict.get(docId)

        self.total_doc_len += BD.get_doc_length()

        freq_words = BD.get_freq_word_map()

        i = 0
        for word in freq_words:
            tfidf = self.tfidf(freq_words[word], len(self.document_dict), number_of_shared_terms[i][1])
            BD.store_tfidf(word, tfidf)
            i += 1

        BD.nomralise_tfidf_values()
        BD.nomralise_tfidf()

    def count_shared_terms(self):
        # New dictionary to be created then sorted
        temp_dict = {}
        # Dictionary items
        dict_items = self.document_dict.items()
        # Document Id
        docId = next(iter(self.document_dict))
        # Dictionary of all frequent words
        freq_words = self.document_dict.get(docId).get_freq_word_map()
        for key, value in dict_items:
            if key != docId:
                doc_tokens = self.document_dict.get(key).get_freq_word_map()
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

    def doc_len_avg(self):
        self.doc_len_avg = self.total_doc_len / len(self.document_dict)

    def start(self):
        self.get_documents()
        self.get_document_terms()
        shared_terms = self.count_shared_terms()
        for key, value in self.document_dict.items():
            self.some_shit(key, shared_terms)
            BD = self.document_dict.get(key)
            # sortedBD = self.sort_bow_document(BD)
            #
            # for tokens in sortedBD:
            #     print ("{} {}".format(tokens[0], tokens[1]))
        self.doc_len_avg()

SE = SearchEngine()

SE.start()
