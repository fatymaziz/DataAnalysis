#Experiment 4: Dataset is WITHOUT Bug severity level Normal
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


# bugs_df= pd.read_csv("bugs_calendar.csv")
bugs_eclipse = pd.read_csv("bugs_eclipse.csv")
bugs_firefox= pd.read_csv("bugs_firefox.csv")
bugs_calendar= pd.read_csv("bugs_calendar.csv")

bugs_eclipse['Type'] = np.where(bugs_eclipse['Severity'] == 'enhancement', "enhancement", "defect")
bugs_df = pd.concat([bugs_firefox,bugs_calendar,bugs_eclipse])



# Dropped rows with severity level '--' and 'Normal' and 'S3'
bugs_df = bugs_df[bugs_df["Severity"].str.contains("--")==False].reset_index()

#Dropped rows with Type "Enhancement" and "Task" because they are not a bug but a new feature
indexSevere = bugs_df[ (bugs_df['Type'] == 'enhancement') & (bugs_df['Type'] == 'enhancement') ].index
bugs_df.drop(indexSevere , inplace=True)

indexSevere = bugs_df[ (bugs_df['Type'] == 'task') & (bugs_df['Type'] == 'task') ].index
bugs_df.drop(indexSevere , inplace=True)

indexSevere = bugs_df[ (bugs_df['Severity'] == 'normal') & (bugs_df['Severity'] == 'normal') ].index
bugs_df.drop(indexSevere , inplace=True)

indexSevere = bugs_df[ (bugs_df['Severity'] == 'S3') & (bugs_df['Severity'] == 'S3') ].index
bugs_df.drop(indexSevere , inplace=True)

#Drop last column 
bugs_df = bugs_df.iloc[: , :-1]


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
file1 = open("output_Experiment4.txt", "w")  # write mode


    
for i in range(0,2):
    TEST_SIZE = 0.2
   
    
    training_data, testing_data = train_test_split(bugs_df, test_size=TEST_SIZE)
    training_data, validation_data = train_test_split(training_data, test_size=TEST_SIZE)

    print(f"No. of training data: {training_data.shape[0]}")
    print(f"No. of validation data: {validation_data.shape[0]}")
    print(f"No. of testing data: {testing_data.shape[0]}")
    

    trainingdataset = len(training_data)
    testingdataset = len(testing_data) 
    validationdataset = len(validation_data)

    training_data_df=training_data.reset_index()
    validation_data_df=validation_data.reset_index()
    testing_data_df=testing_data.reset_index()
    print("------interation------", i)
    file1.write("------Interation------")
    
    dict_resp = outer_loop(TEST_SIZE,bugs_df,trainingdataset,testingdataset,validationdataset,training_data_df,validation_data_df,testing_data_df,validation_data,testing_data)

    print(dict_resp)
    dictionary_resp_eachiteration = dict_resp
    dictionary_list.append(dictionary_resp_eachiteration)
    
    print("*************************Dictionary Ends**************************")
    file1.write("*******************Dictionary Ends**************************")
    
    mlclassifierresp =  mlclassifier_outerloop(TEST_SIZE,bugs_df, trainingdataset,testingdataset,validationdataset,training_data_df,validation_data_df,testing_data_df,training_data)
    
    print(mlclassifierresp)
    ml_resp_eachiteration = mlclassifierresp
    mlresponse_list.append(ml_resp_eachiteration)
    print("********************One Iteration completed***********************")
    
    
    
    #write response of dictionary and Ml CLassifiers in the txt file
file1.write(str(dictionary_list))
file1.write(str(mlresponse_list))

  