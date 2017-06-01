class TrainingSet:

    def __init__(self, setId):
        self.document_dictionary = {}
        self.setId = setId
        self.query = []
        self.query_freq = {}
        self.documents = {}
        self.total_len = 0
        self.total_len_avg = 0
        self.bm25 = {}
        self.k1 = 1.2
        self.k2 = 100
        self.b = 0.75

    def get_set_id(self):
        return self.setId

    def set_query(self, terms):
        self.query = terms

    def get_query(self):
        return self.query

    # Adds the bow document object to the training set
    def add_document(self, docId, document):
        self.documents[docId] = document
        self.set_doc_len_avg()

    # Retrives the documents dictionary
    def get_documents(self):
        return self.documents

    def set_total_len(self, len):
        self.total_len = len

    def get_total_len(self, doclen):
        self.total_len = doclen

    def set_doc_len_avg(self):
        self.total_len_avg = self.total_len / len(document_dictionary)

    def query_term_freq(self, terms):
        for term in terms:
            if term in self.query_freq:
                self.query_freq[term] += 1
            else:
                if len(term) > 2:
                    self.query_freq[term] = 1

    def get_bm25(self):
        return self.bm25

    def K(self, doc_len):
        return self.k1 * ((1 - self.b) + (self.b * (doc_len / self.doc_len_avg)))

    # Calculate the BM25
    def BM25(self, document, shared_terms):

        self.query_term_freq(self.query)

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

    ## Records the terms that are shared across several documents
    ## in the same dictionary
    def count_shared_terms(self):
        # New dictionary to be created then sorted
        temp_dict = {}
        # Dictionary items
        dict_items = self.document_dictionary.items()
        # Document Id
        docId = next(iter(self.document_dictionary))
        # Dictionary of all frequent words
        freq_words = self.document_dictionary.get(docId).get_freq_word_map()
        for key, value in dict_items:

            if key != docId:
                doc_tokens = self.document_dictionary.get(key).get_freq_word_map()
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
