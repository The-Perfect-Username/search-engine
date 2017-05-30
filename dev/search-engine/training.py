class TrainingSet:

    def __init__(self, doc_id):
        self.document_dictionary = {}
        self.document_id = doc_id

    def get_doc_id(self):
        return self.document_id

    def add_term(self, term, relevance):
        self.document_dictionary[term] = relevance

    def get_term_map(self):
        return self.document_dictionary

def discover_training_sets(self):
    for _file_ in self.data_directory:
        read_document = self.read_document(self.training_path, _file_)
        self.create_set(read_document)
    
    # def __init__(self):
    #     self.training_path = './documents/topic-assignment/'
    #     self.data_directory = os.listdir(self.training_path)
    #     self.training_sets = {}
    # def create_set(self, array):
    #     training_set = {}
    #     set_id = array[0][0]
    #     TS = TrainingSet(set_id)
    #
    #     for i in array:
    #         document_id = i[1]
    #         relevance = i[2]
    #         TS.add_term(document_id, relevance)
    #
    #     self.training_sets[set_id] = TS
    #
    # def read_document(self, path, file):
    #     array = []
    #     with open(path + file) as d:
    #         for line in d:
    #             array.append(line.rstrip('\n').split(" "))
    #     return array
    #
    # def display(self):
    #     for sets in self.training_sets:
    #         print(self.training_sets[sets].get_term_map())
