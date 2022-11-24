# Disaster Response Pipeline Project

## Project Motivation
Quick disaster response is crucial to help effectively. This project aims to classify messages to ONGs in order to take action in function of the type of need required.
A Machine Learning model and a Web app were build up to reach the objective.

## Content

1. app
- template
    - master.html # main page of web app
    - go.html # classification result page of web app
- run.py # Flask file that runs app

2. data
- disaster_categories.csv # data to process
- disaster_messages.csv # data to process
- process_data.py
- DisasteResponse.db # database to save clean data to

3. models
- train_classifier.py
- classifier.pkl # saved model

README.md

## Instructions
1. Run the following commands

    - To run ETL pipeline
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run the web app: `python run.py`

