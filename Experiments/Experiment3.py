#Experiment 3: with complete dataset and bug reports includes bugs with Normal severity level
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

bugs_eclipse = pd.read_csv("bugs_eclipse.csv")
bugs_firefox= pd.read_csv("bugs_firefox.csv")
bugs_calendar= pd.read_csv("bugs_Calendar.csv")

bugs_eclipse['Type'] = np.where(bugs_eclipse['Severity'] == 'enhancement', "enhancement", "defect")
bugs_df = pd.concat([bugs_firefox,bugs_calendar,bugs_eclipse])


# Dropped rows with severity level '--'
bugs_df = bugs_df[bugs_df["Severity"].str.contains("--")==False].reset_index()

#Dropped rows with Type "Enhancement" and "Task" because they are not a bug but a new feature
indexSevere = bugs_df[ (bugs_df['Type'] == 'enhancement') & (bugs_df['Type'] == 'enhancement') ].index
bugs_df.drop(indexSevere , inplace=True)

indexSevere = bugs_df[ (bugs_df['Type'] == 'task') & (bugs_df['Type'] == 'task') ].index
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
file1 = open("output_Experiment3.txt", "w")  # write mode


list_of_random_seeds = []

for i in range(0,10):
    TEST_SIZE = 0.2
    
    rs=random.randint(0, 1000000)
    list_of_random_seeds.append(rs)
    randomseed = {'random_seeds':rs}
   
    
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
    lexicon_preprocess_start_time = helper.cpuexecutiontime()
    
    payload_train = helper.lexicon_preprocess(trainingdataset_length,training_data_df)
    
    lexicon_preprocess_end_time = helper.cpuexecutiontime()
    lexicon_preprocess_execution_time =  lexicon_preprocess_end_time -  lexicon_preprocess_start_time
    
#-----------------------Lexicon Learner --------------------------------#
    lexicon_learner_start_time = helper.cpuexecutiontime()
    
    severethreshold, nonseverethreshold = helper.lexicon_learner(payload_train, validation_data)
    winning_threshold = {'severe threshold':severethreshold, 'non severe threshold':nonseverethreshold}
    
    lexicon_learner_end_time = helper.cpuexecutiontime()
    lexicon_learner_execution_time =  lexicon_learner_end_time -  lexicon_learner_start_time
    
#-----------------------Lexicon Classifier ---------------------------------#
    lexicon_classifer_start_time = helper.cpuexecutiontime()
    dict_resp = helper.lexicon_classifier(severethreshold,nonseverethreshold,testing_data,payload_train)
    
    lexicon_classifer_end_time = helper.cpuexecutiontime()
    lexicon_classifer_execution_time =  lexicon_classifer_end_time -  lexicon_classifer_start_time
    
    additional_dict = {'cputime_preprocess': lexicon_preprocess_execution_time,'cputime_learner': lexicon_learner_execution_time,'cputime_classifer': lexicon_classifer_execution_time}
    
    lexicon_classifier_results = {**dict_resp, **additional_dict, **winning_threshold,**randomseed}
        
#     print(lexicon_classifier_results)

#-----------------------List of dictionaries ---------------------------------#
    dictionary_resp_eachiteration = lexicon_classifier_results
    dictionary_list.append(dictionary_resp_eachiteration)
#     print(dictionary_list)
    
      
    
    print("*************************Dictionary Ends**************************")
    file1.write("*******************Dictionary Ends**************************")
 

 #--------------------------------ML Models -----------------------------------------------#
    mlclassifierresp =  helper.mlclassifier_outerloop(trainingdataset_length,testingdataset_length,validationdataset_length,training_data_df,validation_data_df,testing_data_df,training_data,rs)
    
    print(mlclassifierresp)
    ml_resp_eachiteration = mlclassifierresp
    mlresponse_list.append(ml_resp_eachiteration)
#     print(mlresponse_list)
 
    print("********************One Iteration completed***********************")
    
    
    
#--------------------------------Average Results of Lexicon -----------------------------------------------#  
print("************************** Average Result for Lexicon classifier**************************")
average_results_lexicon = helper.calculate_average_results_lexicon(dictionary_list)
average_results_lexicon_df = pd.DataFrame(average_results_lexicon,index=[0])

print("Average Result Lexicon",average_results_lexicon_df)

# store all lexicon results as JSON
with open('lexicon_results3.json', 'w') as json_file:
    json.dump(dictionary_list, json_file)
# store average lexicon results as JSON
with open('lexicon_average_results3.json', 'w') as json_file:
    json.dump(average_results_lexicon, json_file)
 
 #--------------------------------Average Results for ML -----------------------------------------------------#    
print("************************** Average Result for ML classifier**************************")

#  Average results and write the response of ML Models in the txt file
avg_confusionmatrices,average_accuracy, average_f1score,avg_meanf1score, avg_preprocesscputime,avg_learnercputime,avg_classifiercputime = helper.calculate_average_results_ML(mlresponse_list)

average_results_ml = {'Avg Confusion Matrix': avg_confusionmatrices,'Avg Accuracy': average_accuracy,'Avg F1-Score': average_f1score,'Avg Mean F1score':avg_meanf1score,'Avg Preprocess CPUTime': avg_preprocesscputime, 'Avg Learner CPUTime': avg_learnercputime,'Avg Classifer CPUTime': avg_classifiercputime}


average_results_ml_df = pd.DataFrame(average_results_ml)
print("Average result ML",average_results_ml_df)


# Initialize an empty dictionary to store the values of confusion matrix for each model
model_values_CM = {}

for model_name, model_array in avg_confusionmatrices.items():
    model_values_CM[model_name] = model_array.tolist()
# Create a JSON object
average_ml_json_data = {'Avg Confusionmatrix': model_values_CM, 'Accuracy': average_accuracy,'Avg F1-Score': average_f1score,'Avg Mean F1score':avg_meanf1score,'Avg Preprocess CPUTime': avg_preprocesscputime, 'Avg Learner CPUTime': avg_learnercputime,'Avg Classifer CPUTime': avg_classifiercputime}


# store all ML results as JSON
with open('ml_results3.json', 'w') as json_file:
     json.dump(mlresponse_list, json_file)
# store average ML results as JSON
with open('ml_average_results3.json', 'w') as json_file:
     json.dump(average_ml_json_data, json_file)
        

#write response of dictionary and Ml CLassifiers in the txt file
file1.write(str(average_results_lexicon_df))
file1.write(str(average_results_ml_df))