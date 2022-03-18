#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The address of the metadata server
SERVER_IP = "128.2.204.215"
# The port of the metadata server
SERVER_PORT = 8080
# the path to the input file 
ipath = '/Users/gxchris/Desktop/m.csv'
# the path to the output file
opath1 = '/Users/gxchris/Desktop/meta_user.csv'
opath2 = '/Users/gxchris/Desktop/meta_movie.csv'
# batch size
LIMIT = 100000


# In[2]:


from kafka import KafkaConsumer 
import logging
import os
import time
import pandas as pd
import requests
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split


# In[3]:


def read_events(f,LIMIT):
    # Create a consumer to read data from kafka
    consumer = KafkaConsumer(
        'movielog16',
        bootstrap_servers=['localhost:9092'],
        # Read from the start of the topic; Default is latest
        auto_offset_reset='earliest'
    )
    # Prints all messages, again and again!
    processed =0
    for message in consumer:
        # Default message.value type is bytes!
        event = message.value.decode('utf-8')
        processed +=1
        if (event.split(",")[-1].split("/")[0].strip()== 'GET'): 
            if event.split(",")[-1].split("/")[-2]!= 'rate':
                user_id = event.split(",")[1]
                movie_name = event.split(",")[-1].split("/")[-2]
                rating = event.split(",")[-1].split("/")[-1].strip('.mpg')
                f.write(f"{user_id},{movie_name},{rating}\n")
        if processed == LIMIT:
            break


# In[4]:


def download():
    with open (ipath, 'w') as f:
        return read_events(f,LIMIT)


# In[5]:


download()


# In[6]:


from whylogs import get_or_create_session
import pandas as pd

session = get_or_create_session()

df = pd.read_csv(ipath, names=['userid','movieid','rating'])

with session.logger(dataset_name="movie_data") as logger:
    logger.log_dataframe(df)


# In[7]:


prof = session.log_dataframe(df)
summary = prof.flat_summary()
stats_df = summary['summary']
stats_df


# In[ ]:


from whylogs.app import Session
from whylogs.app.writers import WhyLabsWriter
from whylogs import get_or_create_session

os.environ["WHYLABS_API_KEY"] = "pFovK8RVfX.KB8Oh3Qrb0Icz8Tqj6eTIXtUz9wMxHaL949Gnd1X9jrWTSrXq7DuP"
os.environ["WHYLABS_DEFAULT_ORG_ID"] = "org-FRbeeg"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "model-5"

# Create a consumer to read data from kafka
consumer = KafkaConsumer(
    'movielog16',
    bootstrap_servers=['localhost:9092'],
    # Read from the start of the topic; Default is latest
    auto_offset_reset='earliest')
processed =0
writer = WhyLabsWriter()
session = Session(project="movie-test2", pipeline="test2-pipeline", writers=[writer])
with session.logger(dataset_name="dataset", with_rotation_time="s") as logger:
    while True:
        for message in consumer:
            # Default message.value type is bytes!
            event = message.value.decode('utf-8')
            processed +=1
            if (event.split(",")[-1].split("/")[0].strip()== 'GET'): 
                if event.split(",")[-1].split("/")[-2]!= 'rate':
                    user_id = event.split(",")[1]
                    movie_name = event.split(",")[-1].split("/")[-2]
                    rating = event.split(",")[-1].split("/")[-1].strip('.mpg')
                    logger.log({"user_id": user_id, "movie_name": movie_name, "rating": rating})


# In[ ]:


from whylogs.app import Session
from whylogs.app.writers import WhyLabsWriter

os.environ["WHYLABS_API_KEY"] = "pFovK8RVfX.KB8Oh3Qrb0Icz8Tqj6eTIXtUz9wMxHaL949Gnd1X9jrWTSrXq7DuP"
os.environ["WHYLABS_DEFAULT_ORG_ID"] = "org-FRbeeg"

# Adding the WhyLabs Writer to utilize WhyLabs platform
writer = WhyLabsWriter("", formats=[])

session = Session(project="movie-test1", pipeline="test1-pipeline", writers=[writer])

# Point to your local CSV if you have your own data
df = pd.read_csv(ipath, names=['userid','movieid','rating'])

# Run whylogs on current data and upload to WhyLabs.
# Note: the datasetId is the same as the modelId
# The selected model project "MODEL-NAME" is "model-1"

with session.logger(tags={"datasetId": "model-4"}) as ylog:
    ylog.log_dataframe(df)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

import random
import time
import mlflow
import whylogs


# In[ ]:


from whylogs import get_or_create_session
assert whylogs.__version__ >= "0.1.13" # we need 0.1.13 or later for MLflow integration
session = get_or_create_session(".whylogs_mlflow.yaml")
whylogs.enable_mlflow(session)


# In[ ]:


import numpy as np
def target_col(testset):
    target = np.array(testset)
    targetset = target[:,-1].astype(np.float)
    return targetset.tolist()


# In[ ]:


from whylogs.app import Session
from whylogs.app.writers import WhyLabsWriter
from whylogs import get_or_create_session
from whylogs.proto import ModelType


os.environ["WHYLABS_API_KEY"] = "pFovK8RVfX.KB8Oh3Qrb0Icz8Tqj6eTIXtUz9wMxHaL949Gnd1X9jrWTSrXq7DuP"
os.environ["WHYLABS_DEFAULT_ORG_ID"] = "org-FRbeeg"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "model-5"

# Create an MLflow experiment 
experiment_name = "movie_reco"
mlflow.set_experiment(experiment_name)

writer = WhyLabsWriter()
session = Session(project="movie-test2", pipeline="test2-pipeline", writers=[writer])

reader = Reader()
df = pd.read_csv(ipath, names=['userid','movieid','rating'])
dataset = Dataset.load_from_df(df[['userid', 'movieid', 'rating']], reader)
trainset, testset = train_test_split(dataset, test_size=.25)
svd = SVD()
svd.fit(trainset)
with mlflow.start_run(run_name="movie_reco"):
    output = svd.test(testset) #prediction
    rmse = accuracy.rmse(output)
    mlflow.log_metric("rmse", rmse)
    # use whylogs to log data quality metrics for the current batch
    test_df = pd.DataFrame(testset,columns = ['userid', 'movieid', 'rating'])
    mlflow.whylogs.log_pandas(df)
    target = target_col(testset)
    with session.logger(tags={"datasetId": "model-5"}) as ylog:
        ylog.log_dataframe(df)
        ylog.log_metrics(target, [i.r_ui for i in output],model_type=ModelType.REGRESSION)


# In[ ]:


client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
experiment.name, experiment.experiment_id


# In[ ]:


whylogs.mlflow.list_whylogs_runs(experiment.experiment_id)

