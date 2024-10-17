# Tests for Demo purpose
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
import helperdemo


     # Define the bug summaries
# bug_summaries = [
#     "The 'Login' button doesn't allow users to log in.",
#     "The program crashes when dividing two numbers instead of adding them together.",
#     "After filling out a medical history form, clicking 'Save and Exit' doesn't save the information.",
#     "A specific function returns incorrect results due to a typo in the code.",
#     "The application allows unauthorized access to sensitive data."
    
# ]
import pandas as pd

# # Define the data
# Define the data
training_data = {
    "Summary": [
        "Application crashes when attempting to save a file seems this is not reliable.",
        "Slow performance while loading the sales dashboard across all browsers.",
        "User authentication fails, unauthorized users are not reliable allowed to login in the system",
        "Search functionality is not good.",
        "Typo mistakes in the home page not reliable headline section.",
        "Data corruption error occurs during not reliable database transactions."
      
    ],
    "Severity": [
        "Severe",
        "NonSevere",
        "Severe",
        "NonSevere",
        "NonSevere",
        "Severe"
      
    ]
}

testing_data = {
    "Summary": [
        "Payment gateway integration fails, blocking users from transactions not reliable.",
        "Spelling mistakes in the text in the footer section.",
        "Security vulnerability allowing unauthorized not reliable access to the web application.",
        "The finance form crashes when discount is given this is not good."
    ],
    "Severity": [
        "Severe",
        "NonSevere",
        "Severe",
        "Severe"
    ]
}

validation_data = {
    "Summary": [
        "The web application allows unauthorized not reliable access to sensitive data this is not good.",
        "On saving a customer profile form, the data is successfully saved, but the processing time is very slow"
    ],
    "Severity": [
       "Severe",
       "NonSevere"
        
    ]
}
# training_data = {
#     "Summary": [
#         "Application crashes when attempting to save a file.",
#         "Slow performance while loading the sales dashboard across all browsers.",
#         "User authentication fails, unauthorized users are allowed to login in the system",
#         "Search functionality returns incorrect results.",
#         "Typo mistakes in the home page headline section.",
#         "Data corruption error occurs during database transactions."
      
#     ],
#     "Severity": [
#         "Severe",
#         "NonSevere",
#         "Severe",
#         "NonSevere",
#         "NonSevere",
#         "Severe"
      
#     ]
# }

# testing_data = {
#     "Summary": [
#         "Payment gateway integration fails, blocking users from transactions.",
#         "Spelling mistakes in the text in the footer section.",
#         "Security vulnerability allowing unauthorized access to the web application.",
#         "The finance form crashes when discount is given."
#     ],
#     "Severity": [
#         "Severe",
#         "NonSevere",
#         "Severe",
#         "Severe"
#     ]
# }

# validation_data = {
#     "Summary": [
#         "The web application allows unauthorized access to sensitive data.",
#         "On saving a customer profile form, the data is successfully saved, but the processing time is very slow"
#     ],
#     "Severity": [
#        "Severe",
#        "NonSevere"
        
#     ]
# }
# Create the DataFrame
training_data = pd.DataFrame(training_data)
validation_data = pd.DataFrame(validation_data)
testing_data = pd.DataFrame(testing_data)

pd.set_option('display.max_colwidth', None)
print("training_data")
print(training_data)
# Display the DataFrame
print("validation_data")
print(validation_data)
print("testing_data")
print(testing_data)

bugs_df = pd.concat([training_data, validation_data, testing_data], ignore_index=True)
print("total bugs", len(bugs_df))
severerity = bugs_df['Severity'].value_counts()
print(severerity)

dictionary_list = []
mlresponse_list = []
file1 = open("output_demotest.txt", "w")  # write mode

list_of_random_seeds = []

  
for i in range(0,1):
    TEST_SIZE = 0.2
    
    rs=random.randint(0, 1000000)
    list_of_random_seeds.append(rs)
    randomseed = {'random_seeds':rs}
   
    
#     training_data, testing_data = train_test_split(bugs_df, test_size=TEST_SIZE, random_state=rs)
#     training_data, validation_data = train_test_split(training_data, test_size=TEST_SIZE, random_state=rs)

    print(f"No. of training data: {training_data.shape[0]}")
    print(f"No. of validation data: {validation_data.shape[0]}")
    print(f"No. of testing data: {testing_data.shape[0]}")
    
    print("dataset random seed:" + str(rs))

    trainingdataset_length = len(training_data)
    testingdataset_length = len(testing_data) 
    validationdataset_length = len(validation_data)

    training_data_df=training_data.reset_index()
    validation_data_df=validation_data.reset_index()
    testing_data_df=testing_data.reset_index()
    
    print("------------------Training dataset-----------------------")
    print(training_data['Summary'])
    print("----------------Validation dataset--------------------------")
    print(validation_data['Summary'])
    print("--------------Testing dataset--------------------------")
    print(testing_data['Summary'])
    
    
    print("------interation------", i)
    file1.write("------Interation------")
    
    
  #----------------------Lexicon Preprocess ------------------------------#
    lexicon_preprocess_start_time = helperdemo.cpuexecutiontime()
    
    payload_train = helperdemo.lexicon_preprocess(trainingdataset_length,training_data_df)
    
    lexicon_preprocess_end_time = helperdemo.cpuexecutiontime()
    lexicon_preprocess_execution_time =  lexicon_preprocess_end_time -  lexicon_preprocess_start_time
    
#-----------------------Lexicon Learner --------------------------------#
    lexicon_learner_start_time = helperdemo.cpuexecutiontime()
    
    severethreshold, nonseverethreshold = helperdemo.lexicon_learner(payload_train, validation_data)
    winning_threshold = {'severe threshold':severethreshold, 'non severe threshold':nonseverethreshold}
    
    lexicon_learner_end_time = helperdemo.cpuexecutiontime()
    lexicon_learner_execution_time =  lexicon_learner_end_time -  lexicon_learner_start_time
    
#-----------------------Lexicon Classifier ---------------------------------------#
    lexicon_classifer_start_time = helperdemo.cpuexecutiontime()
    
    #create lexicon on the the combined dataset of training and validation dataset on the best threshold -Pending
    severedictionary_list,nonseveredictionary_list,severe_threshold, nonsevere_threshold = helperdemo.dictionary_onthresholds(severethreshold, nonseverethreshold, payload_train)
    
    dict_resp = helperdemo.evaluate_lexicon_classifer(testing_data, severedictionary_list, nonseveredictionary_list)
    
    
    lexicon_classifer_end_time = helperdemo.cpuexecutiontime()
    lexicon_classifer_execution_time =  lexicon_classifer_end_time -  lexicon_classifer_start_time
    
    additional_dict = {'cputime_preprocess': lexicon_preprocess_execution_time,'cputime_learner': lexicon_learner_execution_time,'cputime_classifer': lexicon_classifer_execution_time}
    
    lexicon_classifier_results = {**dict_resp, **additional_dict, **winning_threshold,**randomseed}
        
#     print(lexicon_classifier_results)

#-----------------------List of dictionaries -----------------------------------#
    dictionary_resp_eachiteration = lexicon_classifier_results
    dictionary_list.append(dictionary_resp_eachiteration)
#     print(dictionary_list)
      
    
    print("*************************Dictionary Ends**************************")
    file1.write("*******************Dictionary Ends**************************")
     
    
    
#--------------------------------Average Results of Lexicon -----------------------------------------------#  
print("************************** Average Result for Lexicon classifier**************************")
average_results_lexicon = helperdemo.calculate_average_results_lexicon(dictionary_list)
average_results_lexicon_df = pd.DataFrame(average_results_lexicon,index=[0])

print("Average Result Lexicon",average_results_lexicon_df)

# store all lexicon results as JSON
with open('demo_lexicon_results1.json', 'w') as json_file:
    json.dump(dictionary_list, json_file)
# store average lexicon results as JSON
with open('demo_lexicon_average_results1.json', 'w') as json_file:
    json.dump(average_results_lexicon, json_file)
 



