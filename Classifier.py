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
from Cleaner import Cleaner

class Classifier(Cleaner):

    def read_text(self, filename):
        with open(filename, "r", "UTF-8"):
            pass
