from config import *


def tweet_count(id_str):
    return collection.count_documents({'user.id_str': id_str})

def tweeted_at_count(id_str):
    return collection.count_documents({'entities.user_mentions': {'$elemMatch': {'id_str': id_str}}})







