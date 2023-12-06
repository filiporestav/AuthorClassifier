import pandas as pd
import os
from nltk import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re # Regex
from nltk.corpus import stopwords
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Author import Author

"""
DESCRIPTION:
This program identifies authors based on texts. We have selected X number of authors
and the goal is to have the program be able to predict the author better than humans
do (have higher accuracy)
"""

class AuthorClassifier:
    """
    A class used to classify authors of texts.

    """

    def __init__(self, books_directory, max_features=5000, min_sentence_length=15):
        self.df = None
        self.books_directory = books_directory
        self.data = {"text": [], "author": []}
        self.min_sentence_length = min_sentence_length # Only consider sentences which have number of words greater or equal than this

        # Lists holding our training and test data
        self.X_train = None
        self.X_test = None
        self.y_train = None # Target label associated with each document in training set (i.e. author)
        self.y_test = None

        # Our vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)

        # Td-idf-weighted document-term matrices (sparse matrices of (n_samples, n_features))
        self.X_train_vectorized = None # Each row corresponds to a document (or text snippet), and each column to an unique word or term in the entire corpus
        self.X_test_vectorized = None

        # Our classifier
        self.classifier = ComplementNB() # Better than MultinomialNB if we have unbalanced data (which we have)

    """
    Method which creates a Pandas dataframe with text and author column.
    """
    def read_files(self):
        for root, dirs, files in os.walk(self.books_directory): # Iterate through each file in the specified directory
            for filename in files:
                if filename.endswith(".txt"):
                    author_name = os.path.basename(root) # Get author name from subfolder name

                    filepath = os.path.join(root, filename)

                    with open(filepath, "r", encoding="utf-8") as file:
                        content = file.read() # Returns the whole text as a string
                        sentences = sent_tokenize(content) # Save the sentences in a list

                        for sentence in sentences:
                            processed_text_list = self.preprocess_text(sentence)
                            if len(processed_text_list)>=self.min_sentence_length: 
                                self.data["text"].append(" ".join(processed_text_list)) # Convert the list of words to a string
                                self.data["author"].append(author_name)

        self.df = pd.DataFrame(self.data) # Create a dataframe for the data

    """
    Method which processes or "cleans" a text snippet or corpus.
    Input: a string of text
    Output: a list with the words
    """
    def preprocess_text(self, text):
        text = text.lower() # Convert to lowercase

        # Remove short abbreviations, e.g. "K."
        text = re.sub("\w\.", "", text)

        # Replace — with whitespace (commonly used in e.g. Kafka books)
        text = re.sub("—", " ", text)

        # Remove content within square brackets, e.g. "illustration" and "copyright"
        text = re.sub(r'\[[^\]]*\]*', '', text)

        # Remove underscores
        text = re.sub("_", "", text)

        # Remove lines starting with "chapter" followed by Roman numerals or numbers
        text = re.sub(r'\bchapter\b\s+([ivxlcdm]|[0-9])+', '', text)

        text = re.sub("[^A-za-z\s]", "", text) # Remove punctuation and quotation marks

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return filtered_words
    
    """
    Splits our data in training and test sets.
    """
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df["text"],
            self.df["author"],
            test_size=0.2, # 80% training, 20% test
            random_state=42 # Provides a seed for the random number generator, ensuring split is reproducible (same seed will always result in the same split)
        )

    """
    Method which converts the raw text data into a numerical format suitable
    for machine learning models. It is done using a technique called
    Term Frequency-Inverse Document Frequency (TF-IDF)
    """
    def vectorize_data(self):
        self.X_train_vectorized = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vectorized = self.vectorizer.transform(self.X_test)

    """
    Method which trains (fits) a classifier on the vectorized training data
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

        cm = confusion_matrix(self.y_test, y_pred) # Compute confusion matrix

        # Plot confusion matrix as heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(self.y_test), yticklabels=np.unique(self.y_test))
        plt.title("Confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    """
    Method which predicts the author of a new sentence.
    """
    def predict_author(self, sentence):
        processed_sentence_list = self.preprocess_text(sentence)
        processed_sentence = " ".join(processed_sentence_list)
        print(processed_sentence) # Print the processed sentence (for testing purpose)
        new_sentence_vectorized = self.vectorizer.transform([processed_sentence])
        prediction = self.classifier.predict(new_sentence_vectorized)
        print(f'Predicted Author: {prediction[0]}')

    def store_in_csv(self):
        self.df.to_csv("data.csv", index=False)

    """
    Method which plots and visualizes the distribution of our classes (authors)
    """
    def plot_class_distribution(self):
        sns.countplot(x="author", data=self.df)
        plt.title("Class distribution")
        plt.show()

    """
    Method which visualizes the feature importance for each class. Assumes we
    have a feature_log_prob_ attribute, which is present in Naive Bayes classifiers.
    """
    def plot_feature_importance(self, top_n=10):
        if hasattr(self.classifier, "feature_log_prob_"):
            feature_names = np.array(self.vectorizer.get_feature_names_out())
            class_labels = np.unique(self.y_train)
            num_classes = len(class_labels)

            plt.figure(figsize=(16, 6 * num_classes), layout="tight")

            for i, label in enumerate(class_labels):
                plt.subplot(num_classes, 3, i + 1)
                plt.title(f"Top {top_n} Feature Importance for {label}")
                
                # Select the top N features with the highest log probabilities
                top_indices = np.argsort(self.classifier.feature_log_prob_[i, :])[::-1][:top_n]
                top_features = feature_names[top_indices]
                top_log_probs = self.classifier.feature_log_prob_[i, :][top_indices]

                plt.barh(top_features, top_log_probs)
                plt.xlabel("Log probability")

            plt.show()

def main():
    classifier = AuthorClassifier("books")
    print("Reading files...")
    start_reading = time.process_time() # Measure time taken to read files
    classifier.read_files()
    print(f"Sentences read and cleaned in: {time.process_time() - start_reading}")

    classifier.store_in_csv()
    classifier.split_data()

    classifier.vectorize_data()

    # Plot our label distribution
    classifier.plot_class_distribution()

    print("Training our model...")
    start_training = time.process_time()
    classifier.train_classifier()
    print(f"Sentences trained in: {time.process_time() - start_training}")

    # Plot the feature importance of each author
    classifier.plot_feature_importance()

    classifier.evaluate_model()
    while True:
        print("Enter a sentence to predict author of (press Enter on an empty line to finish input): ")
        lines = []

        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        
        test_sentence = " ".join(lines)
        
        if not test_sentence: break

        classifier.predict_author(test_sentence)

if __name__ == "__main__":
    main()
