'''

Training Set Discovery

DictionaryA = {}
U = topic assignment directory
for D in U:
    DictionaryB = {}
    Tokens = Tokenised D line by line and word by word
    for token in Tokens:
        DocumentId = token[1]
        Relevant = token[2]
        DictionaryB[DocumentId] = Relevant
    DictionaryA[Document Number] = DictionaryB






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
