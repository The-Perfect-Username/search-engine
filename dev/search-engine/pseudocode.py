'''

Training Set Discovery

Q = Topic Statement.txt
S = Datasets [101...150]
D = XML documents in the folders in S

Parse Q to find the training set IDs and obtain the query string
Tokenise and stem query terms
Store the query terms in a dictionary where the key = Training Set ID

Access each S[i] folder to parse through each D in S[i]
Tokenise and stem each keyword
Store the keywords into a dictionary where the key = the Document ID of D

Go through the dictionary and create a new dictionary:
    Dictionary = {
                    Set ID: {
                        Document ID: {
                            [list of bow documents]
                        },
                        ...
                    },
                    ...
                }

Calculate TFIDF while doing so
Calculate bm25

Sort to find |D+/D-| in U


Task 2

Parse through the topic assignment poder for the documents in D.

Assign all Relevant Documents in the Relevant Document dictionary with their terms,
Do the same thing with nonrelevant

Loop through each document in the documents dictionary and use the bm25 weighting algorithm to find the bm25
weight score for each term in the document.

Then find the ranking of each document then create new files to show results








'''
