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


Part 2

DictionaryA = {}

For each D in U:
    DictionaryB = {}

    Open file D
    Tokenise D
    For each token in Tokens:
        Remove stopwords
        Stem word
        Store in Array

    Add Array to Dictionary with D Id as the key







'''
