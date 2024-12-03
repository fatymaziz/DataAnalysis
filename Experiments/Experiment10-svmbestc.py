# Experiment 10 Linear SVM Approach with Eclipse training dataset and Firefox testing dataset
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


#--------------------------- Eclipse dataset for training and validation dataset-----------------------------
bugs_eclipse = pd.read_csv("bugs_eclipse.csv")


bugs_eclipse['Type'] = np.where(bugs_eclipse['Severity'] == 'enhancement', "enhancement", "defect")
bugs_df = pd.concat([bugs_eclipse])

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

bugs_df = bugs_df.head(500)
# print(bugs_df)
# print("total bugs", len(bugs_df))
# severerity = bugs_df['Severity'].value_counts()
# print(severerity)

#--------------------------------- Firefox Dataset as a testing dataset----------------------------------
bugs_firefox= pd.read_csv("bugs_firefox.csv")
bugs_calendar= pd.read_csv("bugs_Calendar.csv")


bugs_df_mozilla = pd.concat([bugs_firefox,bugs_calendar])

# Dropped rows with severity level '--'
bugs_df_mozilla = bugs_df_mozilla[bugs_df_mozilla["Severity"].str.contains("--")==False].reset_index()

#Dropped rows with Type "Enhancement" and "Task" because they are not a bug but a new feature
indexSevere = bugs_df_mozilla[(bugs_df_mozilla['Type'] == 'enhancement') & (bugs_df_mozilla['Type'] == 'enhancement') ].index
bugs_df_mozilla.drop(indexSevere , inplace=True)

indexSevere = bugs_df_mozilla[ (bugs_df_mozilla['Type'] == 'task') & (bugs_df_mozilla['Type'] == 'task') ].index
bugs_df_mozilla.drop(indexSevere , inplace=True)


#Catagorise the severity level into a Severe and Non Severe to make it a binary problem
bugs_df_mozilla.loc[bugs_df_mozilla["Severity"] == "blocker", "Severity"] = 'Severe'
bugs_df_mozilla.loc[bugs_df_mozilla["Severity"] == "critical", "Severity"] = 'Severe'
bugs_df_mozilla.loc[bugs_df_mozilla["Severity"] == "major", "Severity"] = 'Severe'
bugs_df_mozilla.loc[bugs_df_mozilla["Severity"] == "S1", "Severity"] = 'Severe'
bugs_df_mozilla.loc[bugs_df_mozilla["Severity"] == "S2", "Severity"] = 'Severe'
bugs_df_mozilla.loc[bugs_df_mozilla["Severity"] == "S3", "Severity"] = 'NonSevere'
bugs_df_mozilla.loc[bugs_df_mozilla["Severity"] == "normal", "Severity"] = 'NonSevere'
bugs_df_mozilla.loc[bugs_df_mozilla["Severity"] == "minor", "Severity"] = 'NonSevere'
bugs_df_mozilla.loc[bugs_df_mozilla["Severity"] == "trivial", "Severity"] = 'NonSevere'
bugs_df_mozilla.loc[bugs_df_mozilla["Severity"] == "S4", "Severity"] = 'NonSevere'

bugs_df_mozilla = bugs_df_mozilla.head(300)
# print(bugs_df_mozilla)
# print("total bugs", len(bugs_df_mozilla))
# severerity = bugs_df_mozilla['Severity'].value_counts()
# print(severerity)

dictionary_list = []
mlresponse_list = []
# file1 = open("output_Experiment10.txt", "w")  # write mode

list_of_random_seeds = []

  
for i in range(0,1):
    TEST_SIZE = 0.2
    
    rs=random.randint(0, 1000000)
    list_of_random_seeds.append(rs)
    randomseed = {'random_seeds':rs}
   
   
#     training_data, testing_data = train_test_split(bugs_df, test_size=TEST_SIZE, random_state=rs)
#     training_data, validation_data = train_test_split(training_data, test_size=TEST_SIZE, random_state=rs)
    training_data, validation_data = train_test_split(bugs_df, test_size=TEST_SIZE, random_state=rs)
    testing_data = bugs_df_mozilla.copy(deep=True)

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
    print("------interation------", i)
#     file1.write("------Interation------")
    
    
 #----------------------Lexicon Preprocess ------------------------------#
    lexicon_preprocess_start_time = helper.cpuexecutiontime()
    
    training_data['Processed_Summary'] = training_data['Summary'].apply(lambda x: helper.nlpsteps(x))
    training_data['Lowered_Summary'] = training_data['Processed_Summary'].apply(lambda x: x.lower())
    
    lexicon_preprocess_end_time = helper.cpuexecutiontime()
    lexicon_preprocess_execution_time =  lexicon_preprocess_end_time -  lexicon_preprocess_start_time
    
#-----------------------Lexicon Learner --------------------------------#
    lexicon_learner_start_time = helper.cpuexecutiontime()
    
    validation_data['Processed_Summary_v'] = validation_data['Summary'].apply(lambda y: helper.nlpsteps(y))
    validation_data['Lowered_Summary_v'] = validation_data['Processed_Summary_v'].apply(lambda y: y.lower())


    severe_lexicons_linearsvm, non_severe_lexicons_linearsvm,C_hyperparameter = helper.linear_svm_features(training_data['Lowered_Summary'],training_data,validation_data['Lowered_Summary_v'],validation_data,training_data_df,validation_data_df)

    
    # Add both severe and non severe dictionaries in a dictionary
    static_dict_resp = {'Severe Lexicons': severe_lexicons_linearsvm, 'NonSevere Lexicon': non_severe_lexicons_linearsvm, 'bestC_hyperparameter':C_hyperparameter}
    
    lexicon_learner_end_time = helper.cpuexecutiontime()
    lexicon_learner_execution_time =  lexicon_learner_end_time -  lexicon_learner_start_time
    
#-----------------------Lexicon Classifier ---------------------------------#
    lexicon_classifer_start_time = helper.cpuexecutiontime()

    dict_resp = helper.evaluate_lexicon_classifer(testing_data ,severe_lexicons_linearsvm, non_severe_lexicons_linearsvm)
    
    lexicon_classifer_end_time = helper.cpuexecutiontime()
    lexicon_classifer_execution_time =  lexicon_classifer_end_time -  lexicon_classifer_start_time
    
    additional_dict = {'cputime_preprocess': lexicon_preprocess_execution_time,'cputime_learner': lexicon_learner_execution_time,'cputime_classifer': lexicon_classifer_execution_time, 'best_C_hyperparameter':C_hyperparameter}
    
    lexicon_classifier_results = {**dict_resp, **additional_dict, **randomseed}
        
#     print(lexicon_classifier_results)

#-----------------------List of dictionaries -----------------------------------#
    dictionary_resp_eachiteration = lexicon_classifier_results
    dictionary_list.append(dictionary_resp_eachiteration)
#     print(dictionary_list)


      
    
    print("*************************Dictionary Ends**************************")
#     file1.write("*******************Dictionary Ends**************************")
 

    
#--------------------------------Average Results of Lexicon -----------------------------------------------#  
print("************************** Average Result for Lexicon classifier**************************")
average_results_lexicon = helper.calculate_average_results_lexicon(dictionary_list)
average_results_lexicon_df = pd.DataFrame(average_results_lexicon,index=[0])

print("Average Result Lexicon",average_results_lexicon_df)

# store all lexicon results as JSON
with open('lexicon_results10_LSVM_bestc.json', 'w') as json_file:
    json.dump(dictionary_list, json_file)
# store average lexicon results as JSON
with open('lexicon_average_results10_LSVM_bestc.json', 'w') as json_file:
    json.dump(average_results_lexicon, json_file)
        
# store static dictionary for Eclispse as json
with open('static_dictionary_eclipse_LSVM_bestc.json', 'w') as json_file:
     json.dump(static_dict_resp, json_file,indent=2)
     