# Experiment 1 with Eclipse dataset only
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import collections
import random
import itertools
from helper import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
from sklearn.dummy import DummyClassifier
import time


# bugs_df= pd.read_csv("bugs_calendar.csv")
bugs_eclipse = pd.read_csv("bugs_eclipse.csv")
# bugs_firefox= pd.read_csv("bugs_firefox.csv")
# bugs_calendar= pd.read_csv("bugs_calendar.csv")

bugs_eclipse['Type'] = np.where(bugs_eclipse['Severity'] == 'enhancement', "enhancement", "defect")
bugs_df = pd.concat([bugs_eclipse])

# Dropped rows with severity level '--'
bugs_df = bugs_df[bugs_df["Severity"].str.contains("--")==False].reset_index()

#Dropped rows with Type "Enhancement" and "Task" because they are not a bug but a new feature
indexSevere = bugs_df[ (bugs_df['Type'] == 'enhancement') & (bugs_df['Type'] == 'enhancement') ].index
bugs_df.drop(indexSevere , inplace=True)

indexSevere = bugs_df[ (bugs_df['Type'] == 'task') & (bugs_df['Type'] == 'task') ].index
bugs_df.drop(indexSevere , inplace=True)

#------this needs to be deleted ---------------
indexSevere = bugs_df[ (bugs_df['Severity'] == 'normal') & (bugs_df['Severity'] == 'normal') ].index
bugs_df.drop(indexSevere , inplace=True)

indexSevere = bugs_df[ (bugs_df['Severity'] == 'S3') & (bugs_df['Severity'] == 'S3') ].index
bugs_df.drop(indexSevere , inplace=True)
#-----up till here------------------


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

dictionary_list = []
mlresponse_list = []
file1 = open("output_Experiment1.txt", "w")  # write mode

list_of_random_seeds = []


    
for i in range(0,2):
    TEST_SIZE = 0.2
    
    rs=random.randint(0, 1000000)
    list_of_random_seeds.append(rs)
   
    
    training_data, testing_data = train_test_split(bugs_df, test_size=TEST_SIZE, random_state=rs)
    training_data, validation_data = train_test_split(training_data, test_size=TEST_SIZE, random_state=rs)

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
    file1.write("------Interation------")
    
    
 #----------------------Lexicon Preprocess ------------------------------#
    lexicon_preprocess_start_time = cpuexecutiontime()
    
    payload_train = lexicon_preprocess(trainingdataset_length,training_data_df)
    
    lexicon_preprocess_end_time = cpuexecutiontime()
    lexicon_preprocess_execution_time =  lexicon_preprocess_end_time -  lexicon_preprocess_start_time
    
#-----------------------Lexicon Learner --------------------------------#
    lexicon_learner_start_time = cpuexecutiontime()
    
    severethreshold, nonseverethreshold = lexicon_learner(payload_train, validation_data)
    
    lexicon_learner_end_time = cpuexecutiontime()
    lexicon_learner_execution_time =  lexicon_learner_end_time -  lexicon_learner_start_time
    
#-----------------------Lexicon Classifier ---------------------------------#
    lexicon_classifer_start_time = cpuexecutiontime()
    dict_resp = lexicon_classifier(severethreshold,nonseverethreshold,testing_data,payload_train)
    
    lexicon_classifer_end_time = cpuexecutiontime()
    lexicon_classifer_execution_time =  lexicon_classifer_end_time -  lexicon_classifer_start_time
    
    additional_dict = {'cputime_preprocess': lexicon_preprocess_execution_time,'cputime_learner': lexicon_learner_execution_time,'cputime_classifer': lexicon_classifer_execution_time}
    
    lexicon_classifier_results = {**dict_resp, **additional_dict}
        
    print(lexicon_classifier_results)

#-----------------------List of dictionaries ---------------------------------#
    dictionary_resp_eachiteration = lexicon_classifier_results
    dictionary_list.append(dictionary_resp_eachiteration)
    
      
    
    print("*************************Dictionary Ends**************************")
    file1.write("*******************Dictionary Ends**************************")
    
    mlclassifierresp =  mlclassifier_outerloop(trainingdataset_length,testingdataset_length,validationdataset_length,training_data_df,validation_data_df,testing_data_df,training_data)
    
    print(mlclassifierresp)
    ml_resp_eachiteration = mlclassifierresp
    mlresponse_list.append(ml_resp_eachiteration)
    
    print("********************One Iteration completed***********************")
    
    
    
    #Average results and write the response of dictionary in the txt file
print("************************** Average Result for Lexicon classifier**************************")
average_results_lexicon = calculate_average_results_lexicon(dictionary_list)
average_results_lexicon_df = pd.DataFrame(average_results_lexicon,index=[0])

print(average_results_lexicon_df)

#     #Average results and write the response of dictionary in the txt file
#average_accuracy, average_f1_score = calculate_average_results(mlresponse_list)
# average_results_ml = calculate_average_results(mlresponse_list)
# average_results_ml_df = pd.DataFrame(average_results_ml,index=[0])



file1.write(str(average_results_lexicon_df))
# file1.write(str(average_results_ml_df))

  