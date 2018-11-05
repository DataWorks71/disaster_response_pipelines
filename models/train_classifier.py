import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.preprocessing import StandardScaler

import pickle


def load_data(database_filepath):
    """Load dataset and return X-vars, Y-categories and category_names.

    Args:
    database_filepath: str. The relative filepath of database.  

    Returns:
    X: pandas.DataFrame. Columns of candidate explainatory variables.
    Y: pandas.DataFrame. Columns of target categories.
    category_names: list. category names of Y's
    """
    
    # load data from database / table
    my_table_name = 'ModelTrainData'
    engine = create_engine('sqlite:///' + database_filepath)

    df = pd.read_sql_table(my_table_name, engine)
    
    # Set X-DataFrame
    X_names = ['message', 'original', 'genre']
    X = df[X_names]
    
    # Set Y-DataFrame
    Y_names = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
               'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food',
               'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
               'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools',
               'hospitals', 'shops', 'aid_centers', 'other_infrastructure','weather_related',
               'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    Y = df[Y_names]
    # Get category_names from Y-DataFrame
    category_names = list(Y.columns.values)
    
    return X, Y, category_names


def tokenize(text):
    """Takes a text and tokenizes it by splitting
    into words and lemmatizing it.

    Args:
    text: str. Any doc/string
    
    Returns
    clean_tokens: list. Tokenized word from text.

    Credit: Function borrowed from a Udacity lesson
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Build a MultiOutputClassifier for categories (Y)
    for grid search. Model is based on single 'message' 
    text column (X['message']) by creating a pipeline using 
    NLP and RandomForest classifier and a parameter set 
    for grid search. 
    
    Args:
    none (use global X_train, Y_train parameters)
    
    Returns:
    model: model ready for optimization/fit.
    """
    
    # Builds pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(
                criterion='entropy',
                class_weight='balanced_subsample',
                random_state=1,
                n_jobs=-1),
            n_jobs=-1)
        )
    ])
    
    # Set parameters for grid search
    parameters = {'clf__estimator__max_depth': [100, None],
                  'clf__estimator__min_samples_leaf': [2, 3],
                  'clf__estimator__min_samples_split': [2, 3],
                  'vect__ngram_range': [(1, 1), (1, 2)]
             }

    # Set model
    model = GridSearchCV(pipeline, param_grid=parameters, 
                  scoring='f1_micro', 
                  verbose=2, n_jobs=-1, cv=5)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate trained MultiOutputClassifier model on test-set
    Predicted results is printet for each category (MultiOutput). 
    
    Args:
    model: Optimized/fitted model for prediction/evalueation
    X_test: Explanatory variables for test
    Y_test: Labeled category test set
    category_names: list. Category names
    
    Returns:
    none.
    """

    Y_pred = model.predict(X_test)
    
    ### display results
    
    # report start
    print('    ### Classification_report for MultiOutput Classifier ###\n')

    # number of categories for dispaly in report
    n_categories = len(category_names)

    # print report for every category (MultiOutput)
    for i in range(n_categories):    

        print('"{2}" (category) (#{0} of {1})'.format(i+1, n_categories, category_names[i]))
        print(classification_report(Y_test[Y_test.columns[i]], Y_pred[:,i]))

    # report end
    print('    ### END: Classification_report ###') 
    
    
def save_model(model, model_filepath):

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X['message'], Y, test_size=0.2)
        
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