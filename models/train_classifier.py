# import libraries
import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pickle
from sklearn.externals import joblib 
import xgboost as xgb

nltk.download('stopwords') # download for stopwords
nltk.download('wordnet') # download for lemmatization
nltk.download('punkt')  # download punkt
nltk.download('averaged_perceptron_tagger')
    

def avg_f1(y_test, y_pred):
    report = pd.DataFrame(columns=['label','f1'])
    for i,col in enumerate(y_test.columns):
        report.loc[i,'f1'] = f1_score(y_test[col], y_pred[:,idx], average= 'weighted')
    return report['f1'].mean()

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Data',engine)
    X = df['message']
    y = df.loc[:,'related':'direct_report']
    
    return X, y

def tokenize(text):
    # tokenize text after normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text.lower())
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # apply a stemmer
    words = [PorterStemmer().stem(w) for w in words]
    return words

def build_model():
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(xgb.XGBClassifier(max_depth=8, min_child_weight = 1, random_state=42, n_jobs=-1)))])
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # defining df
    report = pd.DataFrame(columns=['label','accuracy','precision','recall','f1'])
    for i,col in enumerate(y_test.columns):
        idx = y_test.columns.get_loc(col)
        report.loc[i,'label'] = col
        report.loc[i,'accuracy'] = accuracy_score(y_test[col], y_pred[:,idx])
        report.loc[i,'precision'] = precision_score(y_test[col], y_pred[:,idx], average= 'weighted',labels=np.unique(y_pred))
        report.loc[i,'recall'] = recall_score(y_test[col], y_pred[:,idx], average= 'weighted')
        report.loc[i,'f1'] = f1_score(y_test[col], y_pred[:,idx], average= 'weighted')
    return report


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        report = evaluate_model(model, X_test, y_test)
        print(report)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()