#Experiment 5: with combine dataset and bug reports for Eclipse and Mozzila for Training and OpenOffice for Testing 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import collections
import random
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
from sklearn.dummy import DummyClassifier
import time
import json 
import helper  


bugs_openoffice_df = pd.read_csv("bugs_OpenOffice.csv")



# Rename Type column and Description column into Severity and Summary, so that we have uniformity across datasets.
bugs_openoffice_df.rename(columns={'Summary': 'Detail'}, inplace=True)
bugs_openoffice_df.rename(columns={'Type': 'Severity'}, inplace=True)
bugs_openoffice_df.rename(columns={'Description': 'Summary'}, inplace=True)

#Drop Rows where Severity and Summary column is Null
bugs_openoffice_df = bugs_openoffice_df.dropna(subset=['Severity'])
bugs_openoffice_df = bugs_openoffice_df.dropna(subset=['Summary'])


#Catagorise the severity level into a Severe and Non Severe to make it a binary problem

bugs_openoffice_df.loc[bugs_openoffice_df["Severity"] == "major", "Severity"] = 'Severe'
bugs_openoffice_df.loc[bugs_openoffice_df["Severity"] == "normal", "Severity"] = 'NonSevere'
bugs_openoffice_df.loc[bugs_openoffice_df["Severity"] == "minor", "Severity"] = 'NonSevere'
bugs_openoffice_df.loc[bugs_openoffice_df["Severity"] == "trivial", "Severity"] = 'NonSevere'

# Move 'Severity' column using
severity_col = bugs_openoffice_df.pop('Severity')
# Get the list of remaining columns (excluding 'Severity')
remaining_cols = list(bugs_openoffice_df.columns)
# Insert 'Severity' column at the second last position (index -2)
bugs_openoffice_df.insert(len(remaining_cols) -1, 'Severity', severity_col)


# Select 5 rows from the "Severe" category
bugs_openoffice_df_severe = bugs_openoffice_df[bugs_openoffice_df['Severity'] == 'Severe'].head(5)



print(bugs_openoffice_df_severe)
print("Total bugs:", len(bugs_openoffice_df_severe))
severity_counts = bugs_openoffice_df_severe['Severity'].value_counts()
print(severity_counts)
print()




def find_words_in_lexicon(summary, lexicon_file):
    """
    This function checks if words in a summary exist in a lexicon file.

    Args:
        summary: The text summary to search.
        lexicon_file: Path to the JSON lexicon file.

    Prints each found word and its category.
    """
    with open(lexicon_file) as f:
        lexicon_data = json.load(f)
    for word in summary.split():
        word = word.lower()  # Case insensitive search
        if word in lexicon_data["Severe Lexicons"]:
            print(f"{word} - Severe")
        elif word in lexicon_data["NonSevere Lexicon"]:
            print(f"{word} - NonSevere")
        else:
            print(f"{word} - Not found in lexicon")


summary = bugs_openoffice_df_severe["Summary"].iloc[1]  
lexicon_file = "static_Lexicon_Experiment5_Filtered0.json"

find_words_in_lexicon(summary, lexicon_file)




