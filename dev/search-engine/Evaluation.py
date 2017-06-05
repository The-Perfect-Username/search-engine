import os
import re
import xml.etree.ElementTree as ET

class Evaluator:

    def __init__(self, setId):
        self.setId = setId
        self.TaskOnePath = "./documents/training-set/BaselineResult{}.dat".format(int(self.setId) - 100)
        self.TaskTwoPath = "./documents/results/result{}.dat".format(int(self.setId) - 100)
        self.Training = "./documents/topic-assignment/Training{}.txt".format(self.setId)

        self.number_of_relevant_documents = 0
        self.number_of_retrieved_documents = 0
        self.number_of_retrieved_relevant_documents = 0

        self.task_one_documents = {}
        self.task_two_documents = {}
        self.training_docs = {}

    def reset(self):
        self.number_of_relevant_documents = 0
        self.number_of_retrieved_documents = 0
        self.number_of_retrieved_relevant_documents = 0

    def startProcess(self):
        self.read_training()
        self.start_task_one()
        __recall = self.recall()
        __precision = self.precision()
        __f_one = self.f_one(__recall, __precision)

        f = open("./documents/evaluations/Evaluation{}.txt".format(self.setId), "w")

        text = "Task One\n"
        text += "The number of relevant documents is {} \n".format(self.number_of_relevant_documents)
        text += "The number of retrieved documents is {} \n".format(self.number_of_retrieved_documents)
        text += "recall {} \n".format(__recall)
        text += "precision {} \n".format(__precision)
        text += "F-Measure {} \n\n\n".format(__f_one)

        self.reset()
        self.start_task_two()
        __recall = self.recall()
        __precision = self.precision()
        __f_one = self.f_one(__recall, __precision)

        text += "Task Two \n"
        text += "The number of relevant documents is {} \n".format(self.number_of_relevant_documents)
        text += "The number of retrieved documents is {} \n".format(self.number_of_retrieved_documents)
        text += "recall {} \n".format(__recall)
        text += "precision {} \n".format(__precision)
        text += "F-Measure {} \n\n".format(__f_one)

        f.write(text)
        f.close()


    def read_training(self):
        documents = self.read_documents(self.Training)
        for document in documents:
            self.training_docs[document[1]] = int(document[2])
            # self.number_of_relevant_documents += int(document[2])

    def start_task_one(self):

        arr = self.read_documents(self.TaskOnePath)

        for document in arr:
            self.number_of_retrieved_documents += 1
            self.number_of_relevant_documents += int(document[2])
            if int(document[2]) == int(self.training_docs[document[0]]):
                self.number_of_retrieved_relevant_documents += 1

    def start_task_two(self):
        arr = self.read_documents(self.TaskTwoPath)

        for document in arr:
            self.number_of_retrieved_documents += 1
            self.number_of_relevant_documents += int(document[2])
            if int(document[2]) == int(self.training_docs[document[0]]):
                self.number_of_retrieved_relevant_documents += 1

    def read_documents(self, path):
        temp = []
        with open(path) as f:
            for line in f:
                temp.append(line.rstrip('\n').split(" "))
        return temp

    def recall(self):
        return (self.number_of_retrieved_relevant_documents) / self.number_of_relevant_documents

    def precision(self):
        return (self.number_of_retrieved_relevant_documents) / self.number_of_retrieved_documents

    def f_one(self, recall, precision):
        return (2 * recall * precision) / (recall + precision)
