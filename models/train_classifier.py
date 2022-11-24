import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

##for tokenizing
#cleaning
import re
#removing stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#getting the roots of the words
from nltk.stem.wordnet import WordNetLemmatizer
#getting the message separated by words
from nltk.tokenize import word_tokenize

#for testing the model
from sklearn.model_selection import train_test_split, GridSearchCV

#for building the pipeline
from sklearn.pipeline import Pipeline
#for vectorizing the message
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#classifying
from sklearn.multioutput import MultiOutputClassifier    
from sklearn.ensemble import RandomForestClassifier

#for classification report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#save as pickle
import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('categories', con = engine)
    X = np.array(df['message'])
    Y = np.array(df[['related', 'request', 'offer',\
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',\
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',\
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',\
       'infrastructure_related', 'transport', 'buildings', 'electricity',\
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',\
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',\
       'other_weather', 'direct_report']])
    category_names = ['related', 'request', 'offer',\
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',\
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',\
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',\
       'infrastructure_related', 'transport', 'buildings', 'electricity',\
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',\
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',\
       'other_weather', 'direct_report']    
    return X, Y, category_names
    
def tokenize(text):
    text = text.lower()
    #extract all the character strings that are not numbers or letters
    text = re.sub(pattern = '[^a-zA-Z0-9]', repl = " ", string = text)
    #separate in words
    tokens = word_tokenize(text)
    #lemmatizer
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in nltk.corpus.stopwords.words('english')]
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfdi', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    pipeline.get_params()

    parameters = {
        'clf__estimator__min_samples_split': [2, 3]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(0, Y_test.shape[1]):
        print(category_names[i], "\n",
              "Precision score: {:.2f}, ".format(precision_score(Y_test[:,i],Y_pred[:,i])),
              "Recall score: {:.2f}, ".format(recall_score(Y_test[:,i],Y_pred[:,i])),
              "F1 score: {:.2f}".format(f1_score(Y_test[:,i],Y_pred[:,i])),
              "\n"
             )
    
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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