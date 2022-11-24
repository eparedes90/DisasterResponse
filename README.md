# Disaster Response Pipeline Project

## Project Motivation
Pipeline classifier of messages

### Instructions:
1. Run the following commands

    - To run ETL pipeline
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline
        `python models/train_classifier.py data/DisasterResponse2.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run the web app: `python run.py`

