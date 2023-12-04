import string
import regex as re

class Cleaner:


    """
    Takes in a line as input and returns the line as a list with the words with only alphabets.
    """
    def clean_line(self, line):
        printable_string = "".join(char for char in line if char in string.printable) # Remove non-printable characters
        cleaned_line = re.sub("[^A-Za-z\s]", "", printable_string) # Replaces everyting except alphabets with nothing (i.e. removal)

        cleaned_line = cleaned_line.split()