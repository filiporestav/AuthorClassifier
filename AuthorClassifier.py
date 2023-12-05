import pandas as pd
import os
from nltk import sent_tokenize
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

class AuthorClassifier:

    def __init__(self, books_directory, max_features=5000):
        self.df = None
        self.books_directory = books_directory
        self.data = {"text": [], "author": []}

        # Lists holding our training and test data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Our vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)

        # Td-idf-weighted document-term matrices (sparse matrices of (n_samples, n_features))
        self.X_train_vectorized = None
        self.X_test_vectorized = None

        # Our classifier
        self.classifier = MultinomialNB()

    """
    Method which creates a Pandas dataframe with text and author column.
    """
    def read_files(self):
        for filename in os.listdir(self.books_directory):
            if filename.endswith(".txt"):
                author = filename.split("_")[0] # Filenames are in the format "author_book.txt"
                filepath = os.path.join(self.books_directory, filename)

                with open(filepath, "r", encoding="utf-8") as file:
                    content = file.read()
                    sentences = sent_tokenize(content)

                    for sentence in sentences:
                        self.data["text"].append(sentence)
                        self.data["author"].append(author)

        self.df = pd.DataFrame(self.data) # Create a dataframe for the data
        self.df["text"] = self.df["text"].apply(self.preprocess_text) # Preprocess the text data
        print(self.df.head())

    """
    Method which processes a text snippet or corpus.
    """
    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    """
    Splits our data in training and test sets.
    """
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df["text"],
            self.df["author"],
            test_size=0.2, # 80% training, 20% test
            random_state=42 # Used to make sure we always use the same training & test data
        )

    """
    Method which vectorizes the text data using a technique 
    called Term Frequency-Inverse Document Frequency (TF-IDF)
    """
    def vectorize_data(self):
        self.X_train_vectorized = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vectorized = self.vectorizer.transform(self.X_test)

    """
    Method which trains a classifier on the vectorized training data
    using multinomial Naive Bayes.
    """
    def train_classifier(self):
        self.classifier.fit(self.X_train_vectorized, self.y_train)

    """
    Method which evaluates the model on the test set.
    """
    def evaluate_model(self):
        y_pred = self.classifier.predict(self.X_test_vectorized.toarray())

        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        print('\nClassification Report:')
        print(classification_report(self.y_test, y_pred))

    """
    Method which predicts the author of a new sentence.
    """
    def predict_author(self, sentence):
        new_sentence_vectorized = self.vectorizer.transform([self.preprocess_text(sentence)])
        prediction = self.classifier.predict(new_sentence_vectorized)
        print(f'Predicted Author: {prediction[0]}')

def main():
    classifier = AuthorClassifier("books")
    classifier.read_files()
    classifier.split_data()
    classifier.vectorize_data()
    classifier.train_classifier()
    classifier.evaluate_model()
    while True:
        test_sentence = input("Enter a sentence to predict author of: ")
        if test_sentence=="": break
        classifier.predict_author(test_sentence)

if __name__ == "__main__":
    main()
