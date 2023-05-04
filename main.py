import pymongo
import json
import os
import time

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['DBL_test']
collection = db['data']
directory = "data/data"

array = []
for filename in os.listdir(directory):
    array.append(filename)
counter = 0

for file_name in array:
    start_time = time.time()
    counter += 1

    with open(f'data/data/{str(file_name)}') as f:
        for line in f:
            try:
                data = json.loads(line)
                collection.insert_one(data)
            except json.decoder.JSONDecodeError as e:
                print("JSONDecodeError:", e)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time: ", elapsed_time, file_name)
    break
