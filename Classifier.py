"""
FEATURES:
- WORD FREQUENCIES (UNIGRAM)
- SENTENCE LENGTH
- AVERAGE WORD LENGTH
- SPECIAL CHARACTERS
- COMMA, SEMICOLON, COLON, QUOTE MARKS, BINDESTRECK, QUESTION MARK, EXCLAMATION MARK
- PART OF SPEECH (ORDKLASS)

CLEANING:
- EVERYTHING EXCEPT A-Z
"""
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

# TOKENIZATION
file_content = open("data/charles_dickens_1.txt").read()
tokens = nltk.word_tokenize(file_content)

# LEMMATIZATION
lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for token in tokens:
    lemmatized_words.append(lemmatizer.lemmatize(token))

print(lemmatized_words)

# STOPWORD REMOVAL