#Experiment 14 Firefox dataset with Vader Lexicon approach
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

bugs_firefox= pd.read_csv("bugs_firefox.csv")
bugs_calendar= pd.read_csv("bugs_Calendar.csv")


bugs_df = pd.concat([bugs_firefox,bugs_calendar])

# Dropped rows with severity level '--'
bugs_df = bugs_df[bugs_df["Severity"].str.contains("--")==False].reset_index()

#Dropped rows with Type "Enhancement" and "Task" because they are not a bug but a new feature
indexSevere = bugs_df[ (bugs_df['Type'] == 'enhancement') & (bugs_df['Type'] == 'enhancement') ].index
bugs_df.drop(indexSevere , inplace=True)

indexSevere = bugs_df[ (bugs_df['Type'] == 'task') & (bugs_df['Type'] == 'task') ].index
bugs_df.drop(indexSevere , inplace=True)


#Catagorise the severity level into a Severe and Non Severe to make it a binary problem
bugs_df.loc[bugs_df["Severity"] == "blocker", "Severity"] = 'Severe'
bugs_df.loc[bugs_df["Severity"] == "critical", "Severity"] = 'Severe'
bugs_df.loc[bugs_df["Severity"] == "major", "Severity"] = 'Severe'
bugs_df.loc[bugs_df["Severity"] == "S1", "Severity"] = 'Severe'
bugs_df.loc[bugs_df["Severity"] == "S2", "Severity"] = 'Severe'
bugs_df.loc[bugs_df["Severity"] == "S3", "Severity"] = 'NonSevere'
bugs_df.loc[bugs_df["Severity"] == "normal", "Severity"] = 'NonSevere'
bugs_df.loc[bugs_df["Severity"] == "minor", "Severity"] = 'NonSevere'
bugs_df.loc[bugs_df["Severity"] == "trivial", "Severity"] = 'NonSevere'
bugs_df.loc[bugs_df["Severity"] == "S4", "Severity"] = 'NonSevere'


# bugs_df = bugs_df.head(100)
# print("total bugs", len(bugs_df))
# severerity = bugs_df['Severity'].value_counts()
# print(severerity)


dictionary_list = []
file1 = open("output_Experiment14.txt", "w")  # write mode


list_of_random_seeds = []

for i in range(0,10):
    TEST_SIZE = 0.2
    
    rs=random.randint(0, 1000000)
    list_of_random_seeds.append(rs)
    randomseed = {'random_seeds':rs}
     

    
#-----------------------Lexicon Classifier ---------------------------------#
    lexicon_classifer_start_time = helper.cpuexecutiontime()
    
    texts = bugs_df['Summary']
    
#     predicted_labels = [classify_sentiment(text) for text in bugs_df]    
#     predicted_labels = bugs_df['Summary'].apply(lambda x: x.helper.classify_sentiment())

    bugs_df['Predicted_Severity'] = bugs_df['Summary'].apply(helper.classify_sentiment)
   
    dict_resp = helper.evaluate_vader_classifier(bugs_df['Severity'], bugs_df['Predicted_Severity'])
    
    lexicon_classifer_end_time = helper.cpuexecutiontime()
    lexicon_classifer_execution_time =  lexicon_classifer_end_time -  lexicon_classifer_start_time
    
    additional_dict = {'cputime_classifer': lexicon_classifer_execution_time}
    
    lexicon_classifier_results = {**dict_resp, **additional_dict, **randomseed}
        
#     print(lexicon_classifier_results)

#-----------------------List of dictionaries -----------------------------------#
    dictionary_resp_eachiteration = lexicon_classifier_results
    dictionary_list.append(dictionary_resp_eachiteration)
#     print(dictionary_list)

    
    
#--------------------------------Average Results of Lexicon -----------------------------------------------#  
print("************************** Average Result for Lexicon classifier**************************")

average_results_lexicon = helper.calculate_average_vader(dictionary_list)

print("Average Result Lexicon",average_results_lexicon)

# store all lexicon results as JSON
with open('lexicon_results14_vader.json', 'w') as json_file:
    json.dump(dictionary_list, json_file)
# store average lexicon results as JSON
with open('lexicon_average_results14_vader.json', 'w') as json_file:
    json.dump(average_results_lexicon, json_file)
        
        