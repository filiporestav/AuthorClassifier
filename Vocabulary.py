"""
This class stores all the unigram probabilities for all our training data. It is used to compute the probabilities of words.
"""

class Vocabulary:
    self.unigram = {} # Stores the probabilities of each token, either word or special character.
    self.average_word_length = 0 # Store the average word length of our vocabulary
    self.part_of_speech = {} # Stores occurance of a part of speech: {part of speech: probability}