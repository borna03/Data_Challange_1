import pymongo
import json
import os
import time
import matplotlib as plt
import pandas
import numpy as np

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['DBL_test']
collection = db['data']
directory = "C:/Users/mattl/OneDrive/Desktop/data"
airlines = {'KLM': {'id_str': '56377143', 'tweet_count': 36221, 'tweeted_at_count': 236089},
            'AirFrance': {'id_str': '106062176', 'tweet_count': 10076, 'tweeted_at_count': 96299},
            'British_Airways': {'id_str': '18332190', 'tweet_count': 113289, 'tweeted_at_count': 564278},
            'AmericanAir': {'id_str': '22536055', 'tweet_count': 124751, 'tweeted_at_count': 806206},
            'Lufthansa': {'id_str': '124476322', 'tweet_count': 13989, 'tweeted_at_count': 128386},
            'AirBerlin': {'id_str': '26223583', 'tweet_count': 0, 'tweeted_at_count': 411},
            'AirBerlin assist': {'id_str': '2182373406', 'tweet_count': 0, 'tweeted_at_count': 3},
            'easyJet': {'id_str': '38676903', 'tweet_count': 58083, 'tweeted_at_count': 343945},
            'RyanAir': {'id_str': '1542862735', 'tweet_count': 21917, 'tweeted_at_count': 338877},
            'SingaporeAir': {'id_str': '253340062', 'tweet_count': 13604, 'tweeted_at_count': 73325},
            'Qantas': {'id_str': '218730857', 'tweet_count': 12863, 'tweeted_at_count': 168063},
            'EtihadAirways': {'id_str': '45621423', 'tweet_count': 1513, 'tweeted_at_count': 80441},
            'VirginAtlantic': {'id_str': '20626359', 'tweet_count': 22813, 'tweeted_at_count': 163544}}


