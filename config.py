import pymongo
import json
import os
import time
import matplotlib
import pandas

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['DBL_test']
collection = db['data']
directory = "C:/DBL_Data_Challenge/All Data/data"
airlines = {"KLM": "56377143", "AirFrance": "106062176", "British_Airways": "18332190", "AmericanAir": "22536055",
            "Lufthansa": "124476322", "AirBerlin": "26223583", "AirBerlin assist": "2182373406", "easyJet": "38676903",
            "RyanAir": "1542862735", "SingaporeAir": "253340062", "Qantas": "218730857", "EtihadAirways": "45621423",
            "VirginAtlantic": "20626359"}