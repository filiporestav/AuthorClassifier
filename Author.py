from nltk import sent_tokenize
import numpy as np
import re

class Author:
    def __init__(self, name):
        self.name = name
        self.total_word_count = 0
        self.total_sentence_count = 0
        self.total_unique_words = set()
        self.total_function_word_count = 0
        self.total_punctuation_counts = {}
        self.total_chars = 0 # Used to calculate average word length

        self.average_word_length = 0
        self.average_sentence_length = 0
        self.unique_word_ratio = 0
        self.function_words_ratio = 0
        self.average_punctuation_counts = 0

    def update_data(self, text):
        """
        Method which updates the key metrics used for calculating
        stylometric features for the author based on new text data.
        """
        sentences = sent_tokenize(text)

        for sentence in sentences:
            words = re.findall(r"\b\w+\b", sentence)
            for word in words:
                self.total_chars += len(word)

            self.total_word_count += len(words)
            self.total_sentence_count += 1
            self.total_unique_words.update(set(words))

            function_words = ["I", "you", "he", "she", "it", "we", "they", "am", "is", "are", "was", "were", "be", "being", "been", "have", "has", "had", "do", "does", "did", "a", "an", "the", "and", "but", "or", "if", "unless", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"]
            self.total_function_word_count += sum(1 for word in words if word.lower() in function_words)

            punctuation_marks = re.findall(r"[^\w\s]", sentence)
            for mark in punctuation_marks:
                self.total_punctuation_counts[mark] = self.total_punctuation_counts.get(mark, 0) + 1

    def update_stylometric_features(self):
        """
        Method which updates the authors styleometric features for the author
        if it has been updated with text data.
        """
        self.average_sentence_length = self.total_word_count / self.total_sentence_count
        self.unique_word_ratio = len(self.total_unique_words) / self.total_word_count
        self.function_words_ratio = self.total_function_word_count / self.total_word_count
        self.average_punctuation_counts = sum(self.total_punctuation_counts.values()) / self.total_word_count
        self.average_word_length = self.total_chars / self.total_word_count

    def print_stylometric_features(self):
        print(f"Author: {self.name}")
        print(f"Average Word Length: {self.total_word_count / self.total_sentence_count}")
        print(f"Average Sentence Length: {self.total_word_count / self.total_sentence_count}")
        print(f"Unique Word Ratio: {len(self.total_unique_words) / self.total_word_count}")
        print(f"Function Words Ratio: {self.total_function_word_count / self.total_word_count}")
        print("Average Punctuation Counts:")
        for mark, count in self.total_punctuation_counts.items():
            print(f"{mark}: {count / self.total_word_count}")

    def get_name(self):
        return self.name
