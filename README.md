# Disaster Response Pipeline Project

## Required packages:
- numpy
- pandas
- sklearn
- Flask
- xgboost
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
     
     This will output the following dataframe:
     ![alt text](https://github.com/Mnegh/Disaster-Response-Pipeline/blob/master/illustrations/clean_data.PNG?raw=true)
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
     It will also provide an evaluation of the model:
     
     ![alt text](https://github.com/Mnegh/Disaster-Response-Pipeline/blob/master/illustrations/model_eval.PNG?raw=true)
2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to access the webapp, where you can classify any message you enter.

![alt text](https://github.com/Mnegh/Disaster-Response-Pipeline/blob/master/illustrations/webapp.PNG?raw=true)
