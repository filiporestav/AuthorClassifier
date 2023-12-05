import pandas as pd

class AuthorClassifier:

    def __init__(self, txt):
        self.lines = None
        self.df = None
        self.txt = txt

    def read_lines(self):
        with open(self.txt, "r", encoding="utf-8") as file:
            self.lines = file.readlines()

    def create_dataFrame(self):
        # Create dataframe with one column named "text"
        self.df = pd.DataFrame({'text' : self.lines})

    def save_to_csv(self):
        self.df.to_csv(f"{self.txt}.csv", index=False)

def main():
    classifier = AuthorClassifier("charles_dickens_1.txt")
    classifier.read_lines()
    classifier.create_dataFrame()
    classifier.save_to_csv()
