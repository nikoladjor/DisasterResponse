from pathlib import Path
import sys
import joblib

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import itertools

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, make_scorer

import time
from joblib import dump, load

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


def load_data(database_filepath):
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    tbl_name = database_filepath.split('.db')[0]
    df = pd.read_sql_table('data/DisasterResponse',engine)
    category_names = df.drop(['id','message','original','genre'],axis=1).columns
    Y = df.drop(['id','message','original','genre'],axis=1).values
    X = df['message']

    return X, Y, category_names

def tokenize(text):
    """
    Tokenize the input text. This functions first devides text into separate sentences (if any).
    In the next step, every sentence is normalized (lowercase + punctuation removal).
    After normalization, text is tokenize and lemmatized using WordNetLemmatizer.
    
    Args:
        text (string): input text.
    Returns:
        Tokenized text (list[string]).
    
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # Check if message contains more than one sentence
    _text_sent = sent_tokenize(text)
    _tokenized = []
    for _ss in _text_sent:
        # Remove puncts and normalize
        _text_norm = re.sub(r"[^a-zA-Z0-9]"," ", _ss.lower())
        _text_tokens = word_tokenize(_text_norm)
        _tokenized.append([lemmatizer.lemmatize(_word) for _word in _text_tokens if _word not in stop_words])
    
    _tokenized = list(itertools.chain.from_iterable(_tokenized))
    return _tokenized

def acc_score(y, y_pred):
    acc_tuned_model = (y_pred == y).mean()
    return acc_tuned_model

def build_model():
    model = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize, ngram_range=(1,2), max_features=400)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10), n_jobs=-1))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_features': [None, 1000, 2000, 4000],
        'clf__estimator__n_estimators': [100, 200, 300],
        'clf__estimator__min_samples_split': [4, 6, 8],
        'clf__estimator__ccp_alpha': np.linspace(0, 0.05, 5)
    }
    
    custom_scoring = make_scorer(acc_score)

    model_cv = GridSearchCV(model, param_grid=parameters, verbose=4, scoring=custom_scoring)

    return model_cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Transform the sklrean.metrics.classification_report and extract the precision, recall and f1_score for each class in the multi-class problem.
    
    Args:
        model: sklearn pipeline that implements predict method.
        X_test: test portion of features --> used for pipeline.predict()
        y_test: testing values
        category_names [list of str]: list of labels for easier output
    Returns:
        pandas dataframe with precision, recall and f1
    
    """
    num_class = y_test.shape[1]
    y_pred = model.predict(X_test)
    _report_lines = []
    print("Model evaluation:\n")
    print("Category:\t<Accuracy>\t<Precision>\t<Recall>\t<F1_score>\n")
    for _i in range(num_class):
        _lbl = category_names[_i]
        _clf_rep = classification_report(y_true=y_test[:,_i],y_pred=y_pred[:,_i],labels=[0],target_names=[_lbl])
        _last_line = _clf_rep.strip().split('\n')[-1]
        _precision, _recall, _f1, _support = [np.float32(xx) for xx in re.findall(r'[\d+\.\d+]+', _last_line)]
        _report_lines.append([_lbl, _precision, _recall, _f1, _support])
        print(f"{_lbl}: {_precision}, {_recall}, {_f1}")
        
    acc_model = (y_test == y_pred).mean()
    print(f"Overall model accuracy: {100*acc_model}%")

    _report_lines.append(['overall_accuracy', 100*acc_model])
    dump(_report_lines, Path('.') / 'production_model_report.joblib')

def save_model(model, model_filepath):
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        print('Building model...')
        model = build_model()
        
        start_time = time.time()
        print('Training model...')
        model.fit(X_train, Y_train)
        end_time = time.time()

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

        print(f'Total training time: {end_time - start_time}')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()