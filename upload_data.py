from pymongo.mongo_client import MongoClient
import pandas as pd
import json

uri="mongodb+srv://swastikgautam76:UQgIw7LHe2t8SiYz@cluster0.obktxkg.mongodb.net/?retryWrites=true&w=majority"

client=MongoClient(uri)

DATABASE_NAME="pwskills"
COLLECTION_NAME="waferfault"

df=pd.read_csv("C:\sensor_project\notebooks\wafer_23012020_041211.csv")

df=df.drop("Unnamed: 0",axis=1)

json_record=list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
#you you