# Disaster Response Pipeline Project

## About
Web Application using machine learning pipeline to categorize emergency messages based on the needs communicated by the sender.

The source code does not include the trained model, therefore please follow the steps written in the instructions section of this README to generate the trained model required and use the app.


## ETL Pipeline
The ETL pipeline to prepare data for model training is started as described below.

The pipeline consist of three conceptually siple steps:
 * Loading data
 * Cleaning-up data
 * Storing data to a database

Loading data is done by reading data from two separate files, one file containing messages with their unique IDs, and another one with messages already classified into specific categories. Two data frames created in this way are joined together since their common column is holding unique ID of the message. The message column is in model training step further processed using NLP routines to create feature matrix, while matrix denoting categories is a matrix holding only zeros or ones, thus marking categories into which specific messages are classified.

Full code for running ETL pipeline is located at [data/process_data.py](/data/process_data.py).

## Model training
Since every message can be classified into multiple categories, in this project, the model needs to support multiple classes and multiple outputs. Nice overview of the possible models to be used can be found on [scikit-learn](https://scikit-learn.org/stable/modules/multiclass.html) web page. 

Best overall results were achieved by training `Pipeline` model with two steps:

1. `TfidfVectorizer` using custom `tokenize` function.
1. `MultiOutputClassifier` using `RandomForestClassifier` as a base estimator.

The code for training the model is located in [models/train_classifier.py](/models/train_classifier.py).

The output of this script are two files - the trained model and the classification report which is used to show some date about the trained model on the home page of the web-app.

The final model is trained using the `GridSearchCV` with 5 cross-validation splits (default).

>**NOTE**: Model needs to be trained in order for the app to work! Settings in the `train_classifier.py` will require longer time to train the model, so if only slight modifications in the web-app is needed, one should train the model with smaller number of leaves and <50 estimators, to get the feealing of what the app is doing.

## Usage
### Instructions:
1. Install python 3.8 (the easiest way is to create virtual env) and run `pip install -r requirements.txt` in the projects's root directory. This step is needed only before the first usage of the app.

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier_production.joblib`

3. Go to `app` directory: `cd app`

4. Run your web app: `python run.py`

5. Click the `PREVIEW` button to open the homepage

If the app is used only on a local server, no additional settings are needed. However, on deployment to a cloud storage, additional settings are required based on 

## Examples

Below, you can find three examples with different messages entered after following the instructions above and server started.

**Message 1**: 
>We need food and medical aid.
![Message 1](/docs/udacity_screenshot_1.png?raw=true "Message 1")

**Message 2**:
>Flood in the forest.
![Message 2](/docs/udacity_screenshot_2.png)

**Message 3**:
This one is good example of weak points in this model. Although there is a complete sentence specifically stating that people are missing, the model fails to put this message in the `Missing People` category.
>Earthquake demolished houses. People are missing.
![Message 3](/docs/udacity_screenshot_3.png)