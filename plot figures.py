import pymongo
import matplotlib
import pandas

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['DBL_test']
collection = db['data']

airlines = {"KLM": "56377143", "AirFrance": "106062176", "British_Airways": "18332190", "AmericanAir": "22536055",
            "Lufthansa": "124476322", "AirBerlin": "26223583", "AirBerlin assist": "2182373406", "easyJet": "38676903",
            "RyanAir": "1542862735", "SingaporeAir": "253340062", "Qantas": "218730857", "EtihadAirways": "45621423",
            "VirginAtlantic": "20626359"}


def tweet_count(id_str):
    return db['data'].count_documents({'user.id_str': id_str})


def tweeted_at_count(id_str):
    return db['data'].count_documents({'entities.user_mentions.id_str': id_str})


