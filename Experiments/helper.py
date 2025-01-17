import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import collections
import random
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,precision_score, recall_score, f1_score
import pandas as pd
from sklearn.dummy import DummyClassifier
import time
from sklearn import metrics
import matplotlib.pyplot as plt
import json
from sklearn.svm import LinearSVC
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import vstack


# Download necessary resources
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')



# Function to classify sentiment from Vader Lexicon
def classify_sentiment(texts):
    """
    Classify the new bug report using vader lexicon
    Arg: 
        text: Summary column of the training dataset
    Returns:
        Tags: Severe or NonSevere
    """
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(texts)
    if score['compound'] >= 0:
#         print("Bug is NonSevere", texts,score)
        return 'NonSevere'
        
    else:
#         print("Bug is Severe", texts,score)
        return 'Severe'
    
# Function to evaluate the vader classifier
def evaluate_vader_classifier(true_labels, predicted_labels):
    
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=['NonSevere', 'Severe'])
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1_per_class = f1_score(true_labels, predicted_labels, average=None, labels=['NonSevere', 'Severe'])
    f1_mean = sum(f1_per_class) / len(f1_per_class)
#     print("f1_per_class and their mean",f1_per_class,f1_mean)
    
    vader_dict = {
        "confusionmatrix": conf_matrix.tolist(),
        "Accuracy": accuracy,
        "F1Score": f1_per_class.tolist(),
        "F1Score_mean": f1_mean
    }
    
    return vader_dict

# Function to calculates the average evaluation result of vader
def calculate_average_vader(results):
   
    avg_confusionmatrix = np.mean([result['confusionmatrix'] for result in results], axis=0)
    avg_accuracy = np.mean([result['Accuracy'] for result in results])
    avg_f1score = np.mean([result['F1Score'] for result in results], axis=0)
    avg_f1score_mean = np.mean([result['F1Score_mean'] for result in results])
    avg_cputime = np.mean([result['cputime_classifer'] for result in results])
    
    average_result = {
        'confusionmatrix': avg_confusionmatrix.tolist(),
        'Accuracy': avg_accuracy,
        'F1Score': avg_f1score.tolist(),
        'F1Score_mean': avg_f1score_mean,
        'cputime_classifer': avg_cputime
    }
    
    return average_result
def get_SVM_best_C_hyperparamter(X_train,Y_train,X_validation,y_validation):
    """
    Find the best c parameter for SVM model 

    Args:
        X_train: feature from training dataset
        Y_train: preditive label from training dataset
        X_validation:feature from validation datsset
        y_validation: preditive label from validation dataset dataset
      
    Returns: c paramater of SVM on which the model has performed the best
    """
    C_hyperparameter = [0.1,0.5,1,5,10,20,50,100]
#     C_hyperparameter = [0.1]
   
    SVM_accuracy_list = []
    SVM_list= []
    
    for c in C_hyperparameter:
       
        SVM_dict = {}

        svm_model = SVC(C = c, kernel='linear', gamma='auto')
        svm_model.fit(X_train,Y_train)

        svm_pred = svm_model.predict(X_validation)
        svm_model = confusion_matrix(y_validation, svm_pred)

        SVM_accuracy_list= accuracy_score(y_validation, svm_pred)
        f1_score_svm = f1_score(y_validation, svm_pred, average=None)
        F1ScoreSVM_severe= f1_score_svm[1]
#         print(c,F1ScoreSVM_severe)
        
        SVM_dict = {"C":c,"Accuracy": SVM_accuracy_list, "SVMF1Score": F1ScoreSVM_severe }
        SVM_list.append(SVM_dict)
        SVM_list
        F1Score_df_SVM = pd.DataFrame(SVM_list)
#         print(F1Score_df_SVM)
        
    max_f1Score_svm = F1Score_df_SVM[F1Score_df_SVM['SVMF1Score']==F1Score_df_SVM['SVMF1Score'].max()]
    
    best_c_hyperparamter = max_f1Score_svm['C'].values[0]
#     print("best c", best_c_hyperparamter)
   
        
    return best_c_hyperparamter
      

# def linear_svm_features(summary_training,training_data,summary_validation,validation_data,training_data_df,validation_data_df):
#     """
#     Create a wordlist and their cofficient from linear SVM.
#     Arg: 
#         x: Summary column of the training dataset
#         bug_df: training dataset
#     Returns:
#         severe_lexicons_linearsvm, non_severe_lexicons_linearsvm dictionaries
#     """
#     severe_lexicons_linearsvm = {}  
#     non_severe_lexicons_linearsvm = {} 
   

#     # Initialize CountVectorizer and Transform the processed summary column
#     cv = CountVectorizer()
#     cv.fit(summary_training)

#     X_train = cv.transform(summary_training)
#     X_validation = cv.transform(summary_validation)

#     Y_train = training_data_df['Severity'].apply(lambda x: 1 if x == 'Severe' else 0).values
#     y_validation = validation_data_df['Severity'].apply(lambda x: 1 if x == 'Severe' else 0).values

#     # Y_train = training_data_df.iloc[:, -2].values  # target column
#     # y_validation = validation_data_df.iloc[:, -2].values  # target column
    

# # call function to find the best c parameter
#     C_hyperparameter = get_SVM_best_C_hyperparamter(X_train,Y_train,X_validation,y_validation)
# #     print("C_hyperparameter",C_hyperparameter)
   
#   #initialize and fit model
#     svm = LinearSVC(C = C_hyperparameter)
#     svm.fit(X_train, Y_train)

#     # Get the coefficients from the trained SVM model
#     coef = svm.coef_.ravel()
#     # print("coef",coef)
#     # feature names from CountVectorizer
#     feature_names = cv.get_feature_names_out()

#     # dictionary mapping feature names to coefficients
#     word_coefficients = {feature_names[i]: coef[i] for i in range(len(feature_names))}
    
#     word_coefficients

#     # word list and their coefficients
#     for word, coefficient in word_coefficients.items():

#         if coefficient > 0:   
#             severe_lexicons_linearsvm[word] = {"ratio": coefficient}
          
#         elif coefficient < 0:  
#             non_severe_lexicons_linearsvm[word] = {"ratio": coefficient}
            
# #     print("severe_lexicons_linearsvm_before", severe_lexicons_linearsvm)
# #     print("non_severe_lexicons_linearsvms_before", non_severe_lexicons_linearsvm)
                       
#     return severe_lexicons_linearsvm, non_severe_lexicons_linearsvm, C_hyperparameter

def linear_svm_features(summary_training, summary_validation, training_data_df, validation_data_df):
    """
    Create a wordlist and their coefficient from linear SVM.
    Args: 
        summary_training (list): Summary column of the training dataset.
        summary_validation (list): Summary column of the validation dataset.
        training_data_df (DataFrame): Training dataset DataFrame.
        validation_data_df (DataFrame): Validation dataset DataFrame.
    Returns:
        severe_lexicons_linearsvm, non_severe_lexicons_linearsvm dictionaries, and best C hyperparameter.
    """
    severe_lexicons_linearsvm = {}  
    non_severe_lexicons_linearsvm = {} 

    # Initialize CountVectorizer and transform the processed summary column
    cv = CountVectorizer()
    cv.fit(summary_training)

    X_train = cv.transform(summary_training)
    X_validation = cv.transform(summary_validation)

    Y_train = training_data_df['Severity'].apply(lambda x: 1 if x == 'Severe' else 0).values
    y_validation = validation_data_df['Severity'].apply(lambda x: 1 if x == 'Severe' else 0).values

    # Call function to find the best C parameter
    C_hyperparameter = get_SVM_best_C_hyperparamter(X_train, Y_train, X_validation, y_validation)

    # Combine the training and validation datasets
    X_combined = vstack([X_train, X_validation])
    Y_combined = np.concatenate([Y_train, y_validation])

    # Initialize and fit the model
    svm = LinearSVC(C = C_hyperparameter)
    svm.fit(X_combined, Y_combined)

    # Get the coefficients from the trained SVM model
    coef = svm.coef_.ravel()
    # Feature names from CountVectorizer
    feature_names = cv.get_feature_names_out()

    # Dictionary mapping feature names to coefficients
    word_coefficients = {feature_names[i]: coef[i] for i in range(len(feature_names))}

    # Word list and their coefficients
    for word, coefficient in word_coefficients.items():
        if coefficient > 0:   
            severe_lexicons_linearsvm[word] = {"ratio": coefficient}
        elif coefficient < 0:  
            non_severe_lexicons_linearsvm[word] = {"ratio": coefficient}

    return severe_lexicons_linearsvm, non_severe_lexicons_linearsvm, C_hyperparameter

def zero_equal():
    """
    Randomly tags a bug as Severe or Non Severe.

    Returns:
        str: "Severe" or "Non Severe" based on random selection.
    """
    # Generate a random number (0 or 1)
    random_number = random.randint(0, 1)

    # Assign severity based on the random number
    if random_number == 0:
        return "NonSevere"
    else:
        return "Severe"
    
def nonzero_equal(summary, severe_words, nonsevere_words):
    """
    Analyzes the data items which are tagged Neutral or non-zero equal and calculates their percentage by their index position and tags them Severe or Nonsevere

    Args:
        summary: A list of words representing the summary text.
        severe_words: A list of severe words.
        nonsevere_words: A list of non-severe words.

    Returns:
       Returns the category which has a minimum percentage
    """
    
    sortedwords_severe = sorted(severe_words.items(), key=lambda x: x[1]['ratio'], reverse=True)
    sortedwords_nonsevere = sorted(nonsevere_words.items(), key=lambda x: x[1]['ratio'], reverse=True)
    
    # Convert severe_words and nonsevere_words to dictionaries
    severe_dict = {word: data['ratio'] for word, data in sortedwords_severe}
    nonsevere_dict = {word: data['ratio'] for word, data in sortedwords_nonsevere}

    severe_percentages = []
    nonsevere_percentages = []

    for word in summary:
        lower_word = word.lower()  # case-insensitive matching

        if lower_word in severe_dict:
            index = list(severe_dict.keys()).index(lower_word)
            severe_percentages.append(index / len(severe_dict) * 100)
           
        if lower_word in nonsevere_dict:
            index = list(nonsevere_dict.keys()).index(lower_word)
            nonsevere_percentages.append(index / len(nonsevere_dict) * 100)

    # Create separate DataFrames for severe and non-severe percentages
    df_severe = pd.DataFrame({'Severe': severe_percentages})
    df_nonsevere = pd.DataFrame({'Non-severe': nonsevere_percentages})
    
    # Calculate the minimum value for each category
    min_severe = df_severe.min().values[0]
    min_nonsevere = df_nonsevere.min().values[0]
    
#     print("min_severe",min_severe)
#     print("min_nonsevere",min_nonsevere)
    
    # Determine the category with the lower percentage
    if min_severe <= min_nonsevere:
        return 'Severe'
    else:
        return 'NonSevere'

#Function that returns the average result for the ML classfiers
def calculate_average_results_ML(ml_results):
    """
    Calculate the average result for the Machine Learning Models(SVM, LG, NB)

    Args:
        ml_results: list of results for ML Model wise
        

    Returns: Variables that has average results for each column in the result.
    """  
    
    # Create dictionaries to store values for each variables for each model
    model_accuracy_sum = {}
    model_f1score_sum = {}
    model_f1score_mean = {}
    model_preprocess_cputime_sum = {}
    model_learner_cputime_sum = {}
    model_classifer_cputime_sum = {}
    model_confusion_matrices = {}
    model_count = {}

    # Iterate through the data and calculate sums
    for result_set in ml_results:
        for model_data in result_set:
            model_name = model_data['Model']
            
            if model_name not in model_accuracy_sum:
                model_accuracy_sum[model_name] = 0
                model_f1score_sum[model_name] = [0, 0]  # Store F1 scores as an array [f1score_1_sum, f1score_2_sum]
                model_f1score_mean[model_name] = 0
                model_preprocess_cputime_sum[model_name] = 0
                model_learner_cputime_sum[model_name] = 0
                model_classifer_cputime_sum[model_name] = 0
                model_confusion_matrices[model_name] = np.zeros((2, 2))  # 2x2 confusion matrix
                model_count[model_name] = 0
            
            model_confusion_matrices[model_name] += model_data['confusionmatrix']
            model_accuracy_sum[model_name] += model_data['Accuracy']
            model_f1score_sum[model_name][0] += model_data['F1Score'][0]
            model_f1score_sum[model_name][1] += model_data['F1Score'][1]
            model_f1score_mean[model_name] += model_data['F1Score_mean']
            model_preprocess_cputime_sum[model_name] += model_data.get('ModelPreprocessCPUTime', 0)
            model_learner_cputime_sum[model_name] += model_data.get('ModelLearnerCPUTime', 0)
            model_classifer_cputime_sum[model_name] += model_data.get('ModelClassiferCPUTime', 0)
    
            model_count[model_name] += 1
   
    # Calculate averages
    
    model_average_accuracy = {model: accuracy_sum / model_count[model] for model, accuracy_sum in model_accuracy_sum.items()}
    
    model_average_confusionmatrices = {model: confusionmatrix_sum / model_count[model] for model, confusionmatrix_sum in model_confusion_matrices.items()}
    
    model_average_preprocesscputime = {model: preprocesscputime_sum / model_count[model] for model, preprocesscputime_sum in model_preprocess_cputime_sum.items()}
    
    model_average_learnercputime = {model: learnercputime_sum / model_count[model] for model, learnercputime_sum in model_learner_cputime_sum.items()}
    
    model_average_classifiercputime = {model: classifercputime_sum / model_count[model] for model, classifercputime_sum in model_classifer_cputime_sum.items()}
    
    model_average_f1score = {
        model: [f1score_sum[0] / model_count[model], f1score_sum[1] / model_count[model]]
        for model, f1score_sum in model_f1score_sum.items()
    }
    model_average_meanf1score = {model: meanf1score_sum / model_count[model] for model, meanf1score_sum in model_f1score_mean.items()}
    
   
    return model_average_confusionmatrices, model_average_accuracy, model_average_f1score, model_average_meanf1score, model_average_preprocesscputime, model_average_learnercputime, model_average_classifiercputime


            
#Function that returns the average result for the lexicon classfier
def calculate_average_results_lexicon(lexicon_results):
    """
    Calculate the average result for the Lexicon classifer

    Args:
        lexicon_results: list of results for lexicon classifer
        

    Returns: A dictionary which has the average results of each column in the result of Lexicon classifer
    """
    total_results = len(lexicon_results)
    average_results = {}
    sum_results = {}

    for result in lexicon_results:
        for key, value in result.items():
            if key not in sum_results:
                sum_results[key] = value
            else:
                sum_results[key] += value

    for key, value in sum_results.items():
        average_results[key] = value / total_results

    return average_results

#CPU execution time
def cpuexecutiontime():
    """
    Calculate the CPU execution time for each process

    Returns: returns current time
    """
    current_time = time.time()
    return current_time

def nlpsteps(text):
    """
    Preprocesses text by handling negation, tokenizing, removing stopwords, and lemmatizing.

    Args:
        text (str): The text to be processed.

    Returns:
        str: The processed text after tokenizing, removing stopwords, and lemmatizing.
    """
    # Remove punctuation
    removed_punctuation = re.sub('[^a-zA-Z]', ' ', str(text))
    removed_punctuation = removed_punctuation.lower()
    tokens = removed_punctuation.split()

    # Remove stopwords and 'not' is preserved
    all_stopwords = set(stopwords.words('english'))
    all_stopwords.remove('not')
    filtered_tokens = [token for token in tokens if token.lower() not in all_stopwords]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in filtered_tokens]  # Use 'v' for verbs
    
    return ' '.join(lemmatized_tokens)

def convert(corpus_trainingdata):
    """
    Data after preprocessing splitting into separate words

    Args:
        corpus_trainingdata: Preprocessed data of the training dataset
     
    Returns: Splitted words
    """
    return [i for item in corpus_trainingdata for i in item.split()]

def getwordcounts(splittedWords):
    occurrences = collections.Counter(splittedWords)
    return occurrences

def get_r1(ns, s):
    """
    Calculate ratio for severe

    Args:
        ns:Number of counts for a word as nonsevere
        s: Number of counts for a word as severe
      
    Returns: severe ratio for a given word
    """
    return s / (s + ns)

def get_r2(ns, s):
    """
    Calculate ratio for nonsevere

    Args:
        ns:Number of counts for a word as nonsevere
        s: Number of counts for a word as severe
      
    Returns: non-severe ratio for a given word
    """
    return ns / (s + ns)

def get_distribution(training_data_df):
    """
    Collects word counts for Severe and NonSevere categories.

    Args:
        training_data_df: DataFrame containing the training dataset with 'Summary' and 'Severity' columns.

    Returns:
        dict, dict: Two dictionaries containing word counts for Severe and NonSevere categories.
    """
    # Preprocess the summary text in the dataset using .loc
    training_data_df.loc[:, 'Summary'] = training_data_df['Summary'].apply(lambda x: nlpsteps(x))
    
    # Separate the dataset into Severe and NonSevere
    severe_df = training_data_df[training_data_df['Severity'] == 'Severe']
    nonsevere_df = training_data_df[training_data_df['Severity'] == 'NonSevere']
    
    # Convert summaries into lists of words
    severe_words = convert(severe_df['Summary'])
    nonsevere_words = convert(nonsevere_df['Summary'])
    
    # Get word counts
    severe_word_counts = getwordcounts(severe_words)
    nonsevere_word_counts = getwordcounts(nonsevere_words)
    
    return severe_word_counts, nonsevere_word_counts

def lexicon_preprocess(severe_word_counts, nonsevere_word_counts):
    """
    Calculates the ratios for Severe and NonSevere categories.

    Args:
        severe_word_counts: Dictionary containing word counts for Severe category.
        nonsevere_word_counts: Dictionary containing word counts for NonSevere category.

    Returns:
        DataFrame: Contains words with their counts and ratios for Severe and NonSevere categories.
    """
    # Combine word counts into a single dictionary
    all_words = set(severe_word_counts.keys()).union(set(nonsevere_word_counts.keys()))
    all_data = {word: {'Severe': severe_word_counts.get(word, 0), 'NonSevere': nonsevere_word_counts.get(word, 0)} for word in all_words}

    # Calculate ratios and prepare payload
    payload_train = {}
    for word, counts in all_data.items():
        ns = counts.get('NonSevere', 0)
        s = counts.get('Severe', 0)
        r1 = get_r1(ns, s)
        r2 = get_r2(ns, s)
        payload_train[word]= {
            'r1': r1,
            'r2': r2
        }
        # Convert to DataFrame and transpose
        payload_train_df = pd.DataFrame(payload_train).T 

    return payload_train_df

def calculate_total_counts(data):
    """
    Calculate total number of bugs category wise (Severe, NonSevere, NonZero_Equal, Zero_Equal)

    Args:
        data: The results represents the the list of of results for each iteration
        

    Returns:
        Returns a dictioanry that has the total counts category wise 
    """
    total_counts = {}
    df_totalcounts = pd.DataFrame(data)
   
    
    total_severecounts = df_totalcounts['severe_counts'].sum()
    total_nonseverecounts = df_totalcounts['nonsevere_counts'].sum()
    total_bothZero_counts = df_totalcounts['neutral_bothZero_counts'].sum()
    total_NoZero_counts = df_totalcounts['neutral_NoZero_counts'].sum()
    total_somethingElse_counts = df_totalcounts['neutral_somethingElse_counts'].sum()
    
    total_counts = {
    "total_severecounts": total_severecounts,
    "total_nonseverecounts": total_nonseverecounts,
    "total_bothZero_counts": total_bothZero_counts,
    "total_NoZero_counts": total_NoZero_counts,
    "total_somethingElse_counts": total_somethingElse_counts
    }
#     print("Counts per interation", total_counts)
    
    return total_counts

#Classifer function to check all the bug category  according to their ratios
def classifier_counts(Summary,severedictionary_list,nonseveredictionary_list):
    """
    Classify a data item as severe or nonsevere Zero Equal, NonZero Equal from validation and test dataset

    Args:
        Summary: textual data from summary column from a validation and test dataset
        severedictionary_list: severe dictionary having words that falls in severe category
        nonseveredictionary_list: nonsevere dictionary having words that falls in severe category
      
    Returns: Tags as Severe Nonsevere, total_bothZero_counts,total_NoZero_counts,total_somethingElse_counts
    """
  
    summaryList = Summary.split()
    mytest_severe = len(set(severedictionary_list).intersection(summaryList))
    mytest_nonsevere = len(set(nonseveredictionary_list).intersection(summaryList))
     
    if mytest_severe > mytest_nonsevere:
        tag = "Severe"
    elif mytest_severe < mytest_nonsevere:
        tag = "NonSevere"
    elif mytest_severe == 0 and mytest_nonsevere == 0:
        tag = "Neutral_bothZero"                            
    elif mytest_severe == mytest_nonsevere:
        tag =  "Neutral_bothNoZero"
    else:
        tag = "Neutral_WithSomethingElse"
   
    return tag

# Function that returns the total counts for each severeity level on all combination of threshold.
def severeity_counts(severe_threshold, nonsevere_threshold, dataset, payload_train):
    """
    Calculate the counts of bug category wise and and for each combination of severe and nonsevere threshold

    Args:
        severe_threshold: Severe threshold defined
        nonsevere_threshold: Nonsevere threshold defined
        dataset: Validtion dataset for testing each created dictionary
        payload_train: The wordlist created from training dataset having words wit its severe and nonsevere ratios
   
    Returns:
        Returns a dictioanry that has the total counts category wise and on each combination of thresholds
    """

    severe_dictionary = {}
    nonsevere_dictionary = {}
    dict_counts = {}
    dict1 = {}
    Severitycounts_list = []
    
    
    
    for keyy in payload_train:
        if payload_train[keyy]['r1'] >= severe_threshold:
            severe_dictionary[keyy] = payload_train[keyy]
        
        if payload_train[keyy]['r2'] >= nonsevere_threshold:
            nonsevere_dictionary[keyy] = payload_train[keyy]
            
    severedictionary_list = list(severe_dictionary.keys())
#     print(severedictionary_list)
    nonseveredictionary_list = list(nonsevere_dictionary.keys())
#     print(nonseveredictionary_list)
    
    dataset["Summary"]  = dataset["Summary"].apply(lambda x: nlpsteps(x))
    
    dataset["my_tag"] = dataset["Summary"].apply(lambda x: classifier_counts(x,severedictionary_list,nonseveredictionary_list))
    
    severe_threshold, nonsevere_threshold = severe_threshold, nonsevere_threshold
    severe_counts = dataset[dataset.my_tag == 'Severe'].shape[0]
    nonsevere_counts = dataset[dataset.my_tag == 'NonSevere'].shape[0]
    neutral_bothZero_counts = dataset[dataset.my_tag == 'Neutral_bothZero'].shape[0]
    neutral_NoZero_counts = dataset[dataset.my_tag == 'Neutral_bothNoZero'].shape[0]
    neutral_somethingElse_counts = dataset[dataset.my_tag == 'Neutral_WithSomethingElse'].shape[0]
    
    
    dict_counts = {"severe_threshold":severe_threshold, "nonsevere_threshold":nonsevere_threshold,"severe_counts":severe_counts, "nonsevere_counts":nonsevere_counts, "neutral_bothZero_counts":neutral_bothZero_counts,"neutral_NoZero_counts": neutral_NoZero_counts,"neutral_somethingElse_counts":neutral_somethingElse_counts}
    
#     print("Severity counts category wise")
        
#     print(dict_counts)
    
    return dict_counts
 
#Incase of equal frequency the classifier will be Severe
def classifier(Summary,severedictionary_list,nonseveredictionary_list):
    """
    Classify a data item as severe or nonsevere from validation and test dataset

    Args:
        Summary: textual data from summary column from a validation and test dataset
        severedictionary_list: severe dictionary having words that falls in severe category
        nonseveredictionary_list: nonsevere dictionary having words that falls in severe category
      
    Returns: Tags as severe and nonsevere
    """
   
    summaryList = Summary.split()
    mytest_severe = len(set(severedictionary_list).intersection(summaryList))
    mytest_nonsevere = len(set(nonseveredictionary_list).intersection(summaryList))

    # print("mytest_severe length------------------------",mytest_severe)
    # print("mytest_nonsevere length-----------------------", mytest_nonsevere)
    
#    ------ -----DEMO ----------
    
    # print("---------Intersection logic for a bug with severe and nonsevere Lexicon-------")
    # print("summaryList", summaryList)
    # print("severe word counts", mytest_severe)
    # print("nonsevere word counts", mytest_nonsevere)
    
    severe_words = set(severedictionary_list).intersection(summaryList)
    nonsevere_words = set(nonseveredictionary_list).intersection(summaryList)
    
    # print("summaryList------------------------",summaryList)
    # print("Severe Words-----------------------", severe_words)
    # print("NonSevere Words----------------------", nonsevere_words)
    
    
    if mytest_severe > mytest_nonsevere:
        tag = "Severe"
        # print(f"Bug severity: {Summary} {tag}")
    elif mytest_severe < mytest_nonsevere:
        tag = "NonSevere"
        # print(f"Bug severity: {Summary} {tag}")
    elif mytest_severe == 0 and mytest_nonsevere == 0:
        tag = zero_equal()                            #returns tag as Severe or NonSevere
        # print(f"Bug severity: {Summary} {tag}")
    elif mytest_severe == mytest_nonsevere:
        if isinstance(severedictionary_list, dict) and all('ratio' in word_dict for word_dict in severedictionary_list.values()):
            tag = nonzero_equal(summaryList, severedictionary_list,nonseveredictionary_list) #retuns tag as Severe or NonSevere
            # print(f"Bug severity: {Summary} {tag}")
        else: 
            tag = zero_equal()
            # print(f"Bug severity: {Summary} {tag}")
    else:
        tag = "Neutral_WithSomethingElse"
        
    return tag
    
        
def dictionary_onthresholds(severe_threshold, nonsevere_threshold, payload_train):
    """
    Create dictionaries on each combination of severe and nonsevere threshold

    Args:
        severe_threshold: threshold set manually for severe from 0.1 to 1.0
        nonsevere_threshold: threshold set manually for nonsevere from 0.1 to 1.0
        payload_train: DataFrame having words with its counts as severe and nonsevere from the training dataset

    Returns: severe_dictionary, nonsevere_dictionary, severe_threshold, nonsevere_threshold
    """
   
    severe_dictionary = {}
    nonsevere_dictionary = {}

    for keyy in payload_train.index:
        # Check for 'r1' existence and value for severe threshold
        if 'r1' in payload_train.columns and float(payload_train.at[keyy, 'r1']) >= float(severe_threshold):
            severe_dictionary[keyy] = {'ratio': float(payload_train.at[keyy, 'r1'])}  # Store value and ratio as float

        # Check for 'r2' existence and value for non-severe threshold
        if 'r2' in payload_train.columns and float(payload_train.at[keyy, 'r2']) >= float(nonsevere_threshold):
            nonsevere_dictionary[keyy] = {'ratio': float(payload_train.at[keyy, 'r2'])}  # Store value and ratio as float

    # print("severe_dictionary inside dictionary_onthresholds function", severe_dictionary)
    # print("nonsevere_dictionary inside dictionary_onthresholds function", nonsevere_dictionary)

    return severe_dictionary, nonsevere_dictionary, severe_threshold, nonsevere_threshold




def evaluate_lexicon_classifer(dataset, severedictionary_list, nonseveredictionary_list):
    
    dataset["Summary"] = dataset["Summary"].apply(lambda x: nlpsteps(x))
    x = dataset["Summary"].apply(lambda x: x.lower())
    
    dataset["my_tag"] = dataset["Summary"].apply(lambda x: classifier(x,severedictionary_list,nonseveredictionary_list))
    
# #  # DEMO Test Example print in the console
    
#     print("----------Severe Lexicon----------------------------------")
#     print(severedictionary_list)
#     print("---------- Non-Severe Lexicon----------------------------------")
#     print(nonseveredictionary_list)
# #     print("---------------Severe & NonSevere Threshold-------------------------------------------")
    
#     print(dataset.loc[:, ["Summary","Severity","my_tag"]])  # DEMO Test Example print in the console
    
    
    TP = 0 
    for d in dataset.iterrows():
        if ((d[1]["my_tag"]== "Severe") & (d[1]["Severity"]== "Severe")):
            TP = TP+1
    FP = 0 
    for d in dataset.iterrows():
        if (d[1]["my_tag"]== "Severe" )& (d[1]["Severity"]== "NonSevere"):
            FP = FP+1
    TN = 0 
    for d in dataset.iterrows():
        if (d[1]["my_tag"]== "NonSevere") & (d[1]["Severity"]== "NonSevere"):
            TN = TN+1
    FN = 0 
    for d in dataset.iterrows():
        if (d[1]["my_tag"]== "NonSevere") & (d[1]["Severity"]== "Severe"):
            FN = FN+1

    confusion_matrix = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN
        
    }
#     F1Score for Severe

    confusion_matrix["Precision"]= confusion_matrix["TP"]/(confusion_matrix["TP"]+confusion_matrix["FP"]) if (confusion_matrix["TP"]) !=0 else 0
    confusion_matrix["Recall"]= confusion_matrix["TP"]/(confusion_matrix["TP"]+confusion_matrix["FN"]) if (confusion_matrix["TP"]) !=0 else 0
    confusion_matrix["F1Score"] = 2*(confusion_matrix["Precision"] * confusion_matrix["Recall"])/(confusion_matrix["Precision"] + confusion_matrix["Recall"]) if (confusion_matrix["Precision"] + confusion_matrix["Recall"]) != 0 else 0
 
 #     F1Score for NonSevere
    confusion_matrix["Precision_nonsevere"]= confusion_matrix["TN"]/(confusion_matrix["TN"]+confusion_matrix["FN"]) if (confusion_matrix["TN"]) !=0 else 0
    confusion_matrix["Recall_nonsevere"]= confusion_matrix["TN"]/(confusion_matrix["TN"]+confusion_matrix["TN"]) if (confusion_matrix["TN"]) !=0 else 0
    confusion_matrix["F1Score_nonsevere"] = 2*(confusion_matrix["Precision_nonsevere"] * confusion_matrix["Recall_nonsevere"])/(confusion_matrix["Precision_nonsevere"] + confusion_matrix["Recall_nonsevere"]) if (confusion_matrix["Precision_nonsevere"] + confusion_matrix["Recall_nonsevere"]) != 0 else 0
    
 #     F1Score average for Severe and NonSevere
    confusion_matrix["F1Score_Average"]= (confusion_matrix["F1Score"]+confusion_matrix["F1Score_nonsevere"])/2 if (confusion_matrix["F1Score_nonsevere"]) !=0 else 0
                                                                                                   
    
    return confusion_matrix
   


################# outerloop breakdown##########################################################################

# Function that returns the created dictionaries on possible combination of the severe and non severe threshold and returns best thresholds

def lexicon_learner(payload_train, validation_data):
    """
    Identify the best threshold for severe and nonsevere on which the best dictionary has been created and saves counts of bugs categorywise in a new file

    Args:
        payload_train: DataFrame containing words with severe and nonsevere counts from the training dataset
        validation_data: validation dataset for testing the created dictionaries
      
    Returns: best threshold for severe and nonsevere on which the created dictionary has the highest f1score
    """
    
    severe_threshold = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    nonsevere_threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
   
    possibleThresholdCombination = list(itertools.product(severe_threshold, nonsevere_threshold))
    result_list = []

    for severe_randomthreshold, nonsevere_randomthreshold in possibleThresholdCombination:
        # Generate dictionaries based on current threshold combination
        severe_dictionary, nonsevere_dictionary, _, _ = dictionary_onthresholds(severe_randomthreshold, nonsevere_randomthreshold, payload_train)
        
        # Evaluate the created dictionaries
        count = evaluate_lexicon_classifer(validation_data, severe_dictionary, nonsevere_dictionary)
       
        # Store results
        result_dictionary = {
            "severe_threshold": severe_randomthreshold,
            "nonsevere_threshold": nonsevere_randomthreshold
        }
        result_dictionary.update(count)
        result_list.append(result_dictionary)

    # print("result_list-------",result_list)
    
    F1Score_df = pd.DataFrame(result_list)
    
    # Find the best threshold based on the highest F1 score
    maxf1Score = F1Score_df[F1Score_df['F1Score'] == F1Score_df['F1Score'].max()]
   

    severethreshold_ = maxf1Score['severe_threshold'].values[0]
    nonseverethreshold_ = maxf1Score['nonsevere_threshold'].values[0]
    # print("Best Threshold", severethreshold_, nonseverethreshold_)
    
    return severethreshold_, nonseverethreshold_



############### outerloop breakdown ends #############

# def get_SVM_best_C_hyperparamter(X_train,Y_train,X_validation,y_validation):
#     """
#     Find the best c parameter for SVM model 

#     Args:
#         X_train: feature from training dataset
#         Y_train: preditive label from training dataset
#         X_validation:feature from validation datsset
#         y_validation: preditive label from validation dataset dataset
      
#     Returns: c paramater of SVM on which the model has performed the best
#     """
#     C_hyperparameter = [0.1,0.5,1,5,10,20,50,100]
# #     C_hyperparameter = [0.1]
   
#     SVM_accuracy_list = []
#     SVM_list= []
    
#     for c in C_hyperparameter:
       
#         SVM_dict = {}

#         svm_model = SVC(C = c, kernel='linear', gamma='auto')
#         svm_model.fit(X_train,Y_train)

#         svm_pred = svm_model.predict(X_validation)
#         svm_model = confusion_matrix(y_validation, svm_pred)

#         SVM_accuracy_list= accuracy_score(y_validation, svm_pred)
#         f1_score_svm = f1_score(y_validation, svm_pred, average=None)
#         F1ScoreSVM_severe= f1_score_svm[1]
# #         print(c,F1ScoreSVM_severe)
        
#         SVM_dict = {"C":c,"Accuracy": SVM_accuracy_list, "SVMF1Score": F1ScoreSVM_severe }
#         SVM_list.append(SVM_dict)
#         SVM_list
#         F1Score_df_SVM = pd.DataFrame(SVM_list)
# #         print(F1Score_df_SVM)
        
#     max_f1Score_svm = F1Score_df_SVM[F1Score_df_SVM['SVMF1Score']==F1Score_df_SVM['SVMF1Score'].max()]
    
#     best_c_hyperparamter = max_f1Score_svm['C'].values[0]
# #     print("best c", best_c_hyperparamter)
   
        
#     return best_c_hyperparamter
      
    

# #---------------------------ML Classifier Starts---------------------------------


def mlclassifier_outerloop(trainingdataset_length,testingdataset_length,validationdataset_length,training_data_df,validation_data_df,testing_data_df,training_data, rs):
    """
    Tokenise, train validate and test the machine learning models i.e SVM, Logistic Regression, Nayes Bayes

    Args:
        trainingdataset_length: size of training dataset
        testingdataset_length: size of testing dataset
        validationdataset_length:size of validation dataset
        training_data_df: training dataset in a dataframe
        validation_data_df:validation dataset in a dataframe
        testing_data_df:testing dataset in a dataframe
        training_data: training dataset
        rs:
      
    Returns: Confusion matrix, accuracy score, f1score-severe, f1score-nonsevere, f1score-mean, ml_preprocess_cputime, ml_learner_cputime, ml_classifer_cputime
    
    """
    

    ml_starttime_preprocess = cpuexecutiontime()

    # Combine the two tokenized lists
    combined_train_validation_data = pd.concat([training_data_df, validation_data_df], ignore_index=True)
   

    #Tokenised the training data and validdation dataset
    combined_train_validation_data.loc[:, 'Summary'] = combined_train_validation_data['Summary'].apply(lambda x: nlpsteps(x))
    # print(combined_train_validation_data['Summary'])
    #Tokenised the testing data
    testing_data_df.loc[:, 'Summary'] = testing_data_df['Summary'].apply(lambda x: nlpsteps(x))
  

    max_feature_list = []
    max_feature_accuracy = []
    SVM_list= []


#     features = [1000,10000,15000]
    features = [15000]
    

    for i in features:

        cv = CountVectorizer()
        X_train = cv.fit_transform(combined_train_validation_data['Summary']).toarray()
        Y_train = combined_train_validation_data['Severity'].apply(lambda x: 1 if x == 'Severe' else 0).values
       

      # Vectorize the testing data
        X_test = cv.transform(testing_data_df['Summary']).toarray()
        y_test = testing_data_df['Severity'].apply(lambda x: 1 if x == 'Severe' else 0).values
      

        # # Print shapes to ensure alignment
        # print("X_test shape:", X_test.shape)
        # print("y_test shape:", len(y_test))

       
        ml_endtime_preprocess = cpuexecutiontime()
        ml_preprocess_cputime = ml_endtime_preprocess - ml_starttime_preprocess
    
        # print("X_train shape:", X_train.shape)
        # print("Y_train shape:", len(Y_train))
        
#         #------------ test purpose, remove later------------------------

        # Display feature names and document-term matrix for verification
#         feature_names = cv.get_feature_names_out() 
#         df = pd.DataFrame(X_train, columns=feature_names)
#         print("Feature Names:", feature_names) 
#         pd.set_option('display.max_columns', None)
#         print(df)

#         print("X_train", X_train)
#         print("Y_train", Y_train)
#         print("X_validation", X_validation)
#         print("y_validation", y_validation)
#         print("X_test", X_test)
#         print("y_test", y_test)
# #         ------------ test purpose, remove later------------------------
        

    #------------------------------------SVM------------------------------------------------------------------
        SVM_learner_starttime = cpuexecutiontime()

#         C_hyperparameter = get_SVM_best_C_hyperparamter(X_train,Y_train,X_validation,y_validation)
        C_hyperparameter = 1.0

        SVM_accuracy_list = []

        SVM_dict = {}

        svm_model = SVC(C = C_hyperparameter, kernel='linear', gamma='auto')
        svm_model.fit(X_train,Y_train)

        SVM_learner_endtime = cpuexecutiontime()
        SVM_learner_cputime = SVM_learner_endtime - SVM_learner_starttime

        SVM_classifier_starttime = cpuexecutiontime()
        svm_pred = svm_model.predict(X_test)
        svm_model = confusion_matrix(y_test, svm_pred)
        
        
        #convert to numpy
        numpy_array_CM = np.array(svm_model)
        _svm_model = numpy_array_CM.tolist()
       
        
        
#         print("------------------------Confusion Matrix display test-------------------")
#         numpy_array_CM_shape = numpy_array_CM.shape
#         cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = svm_model, display_labels = [False, True])

#         cm_display.plot()
#         plt.show()
#         print("------------------------Confusion Matrix display test end-------------------")
        
        SVM_accuracy_list= accuracy_score(y_test, svm_pred)
        f1_score_svm = f1_score(y_test, svm_pred, average=None)
        

        f1score_SVM_mean = np.mean(f1_score_svm)

        SVM_classifer_endtime = cpuexecutiontime()
        SVM_classifer_cputime = SVM_classifer_endtime - SVM_classifier_starttime



        SVM_dict = {"features":i,"c": C_hyperparameter,"Model":'SVM', "confusionmatrix":_svm_model,"Accuracy": SVM_accuracy_list, "F1Score": f1_score_svm.tolist(),"F1Score_mean": f1score_SVM_mean,"ModelPreprocessCPUTime": ml_preprocess_cputime, "ModelLearnerCPUTime":SVM_learner_cputime, "ModelClassiferCPUTime":SVM_classifer_cputime,"RandomSeeds":rs}
        SVM_list.append(SVM_dict)
        SVM_list
        max_feature_list.append(SVM_dict)


     #-------------------------------------Naive Bayes-------------------------------------------------------------
        NB_learner_starttime = cpuexecutiontime()

        classifier = MultinomialNB()
        classifier.fit(X_train, Y_train)

        NB_learner_endtime = cpuexecutiontime()
        NB_learner_cputime = NB_learner_endtime - NB_learner_starttime

        NB_classifier_starttime = cpuexecutiontime()

        MultinomialNB_pred = classifier.predict(X_test)

        cm_MB = confusion_matrix(y_test, MultinomialNB_pred)
        #convert to numpy
        numpy_array_CM = np.array(cm_MB)
        cm_MB = numpy_array_CM.tolist()
        
        
        max_feature_accuracy = accuracy_score(y_test, MultinomialNB_pred)
        f1_score_MB = f1_score(y_test, MultinomialNB_pred, average=None)

        f1score_MB_mean = np.mean(f1_score_MB)


        NB_classifier_endtime = cpuexecutiontime()
        NB_classifer_cputime = NB_classifier_endtime - NB_classifier_starttime

        maxfeature_dict = {"features":i,"Model":'MultinomialNB', "confusionmatrix": cm_MB ,"Accuracy": max_feature_accuracy, "F1Score": f1_score_MB.tolist(), "F1Score_mean":f1score_MB_mean, "ModelPreprocessCPUTime": ml_preprocess_cputime,"ModelLearnerCPUTime":NB_learner_cputime, "ModelClassiferCPUTime":NB_classifer_cputime,"RandomSeeds":rs}
        max_feature_list.append(maxfeature_dict)


        #-------------------------------------Logistic Regression-------------------------------------------------------------
        LR_learner_starttime = cpuexecutiontime()

        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train,Y_train)

        LR_learner_endtime = cpuexecutiontime()
        LR_learner_cputime = LR_learner_endtime - LR_learner_starttime

        LR_classifer_starttime = cpuexecutiontime()
        lr_pred = lr_model.predict(X_test)


        cm_lr = confusion_matrix(y_test, lr_pred)
         #convert to numpy
        numpy_array_lr = np.array(cm_lr)
        cm_lr = numpy_array_lr.tolist()
        
        max_feature_accuracy =  accuracy_score(y_test, lr_pred)
        f1_score_lr = f1_score(y_test, lr_pred, average=None)

        f1score_LR_mean = np.mean(f1_score_lr)


        LR_classifer_endtime = cpuexecutiontime()
        LR_classifer_cputime = LR_classifer_endtime - LR_classifer_starttime

        maxfeature_dict = {"features":i, "Model":'LogisticRegression', "confusionmatrix": cm_lr ,"Accuracy": max_feature_accuracy, "F1Score": f1_score_lr.tolist(),"F1Score_mean":f1score_LR_mean,"ModelPreprocessCPUTime": ml_preprocess_cputime, "ModelLearnerCPUTime":LR_learner_cputime, "ModelClassiferCPUTime":LR_classifer_cputime,"RandomSeeds":rs}
        max_feature_list.append(maxfeature_dict)


       # -------------------------------Dummy Classification--------------------------------------
        dummy_learner_starttime = cpuexecutiontime()

        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(X_train, Y_train)
        dummy_learner_endtime = cpuexecutiontime()
        dummy_learner_cputime = dummy_learner_endtime - dummy_learner_starttime

        dummy_classifer_starttime = cpuexecutiontime()

        DummyClassifier(strategy='most_frequent')
        dummy_pred = dummy_clf.predict(X_test)
        cm_dummy = confusion_matrix(y_test, dummy_pred)
         #convert to numpy
        numpy_array_dummy = np.array(cm_dummy)
        cm_dummy = numpy_array_dummy.tolist()
      
        
        
        dummy_accuracy = dummy_clf.score(X_test, y_test)
        f1_score_dummy = f1_score(y_test, dummy_pred, average=None)
       

        f1score_dummy_mean = np.mean(f1_score_dummy)

        dummy_classifer_endtime = cpuexecutiontime()
        dummy_classifer_cputime = dummy_classifer_endtime - dummy_classifer_starttime

        maxfeature_dict = {"features":i, "Model":'DummyClassifier', "confusionmatrix": cm_dummy ,"Accuracy": dummy_accuracy, "F1Score": f1_score_dummy.tolist(),"F1Score_mean":f1score_dummy_mean, "ModelPreprocessCPUTime": ml_preprocess_cputime, "ModelLearnerCPUTime":dummy_learner_cputime, "ModelClassiferCPUTime":dummy_classifer_cputime,"RandomSeeds":rs }

        max_feature_list.append(maxfeature_dict)

        return max_feature_list
       #-------------------------------------ML Classifer & Dummy classifier Ended here---------------------------
                           
       


