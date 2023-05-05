import pymongo
import json
import os
import time

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['DBL_test2']
collection = db['data']
directory = 'E:/Uni/DBL Data Challenge/All Data/data2'  # change according to where you stored the data

array = []
for filename in os.listdir(directory):
    array.append(filename)
counter = 0


def remove_nested_keys(dictionary, keys_to_remove):
    """
    Recursively removes key-value pairs from a nested dictionary (or JSON) structure
    :param dictionary: dictionary (JSON entry)
    :param keys_to_remove: list of keys matching the ones you want removed
    :return: dictionary (JSON entry) without the removed keys.
    """
    for key in keys_to_remove:
        if key in dictionary:
            del dictionary[key]
    for value in dictionary.values():
        if isinstance(value, dict):
            remove_nested_keys(value, keys_to_remove)
    return dictionary


for file_name in array:
    start_time = time.time()
    counter += 1

    with open(f'{directory}/{str(file_name)}') as f:
        for line in f:
            try:
                data = json.loads(line)
                remove_nested_keys(data,
                                   ['id', 'source', 'url', 'description', 'protected', 'friends_count', 'listed_count',
                                    'favourites_count', 'utc_offset', 'time_zone', 'lang', 'contributors_enabled',
                                    'is_translator', 'profile_background_color', 'profile_background_image_url',
                                    'profile_background_image_url_https', 'profile_background_tile',
                                    'profile_link_color', 'profile_sidebar_border_color', 'profile_sidebar_fill_color',
                                    'profile_text_color', 'profile_use_background_image', 'profile_image_url',
                                    'profile_image_url_https', 'profile_banner_url', 'default_profile',
                                    'default_profile_image', 'following', 'follow_request_sent', 'notifications',
                                    'geo', 'contributors', 'timestamp_ms', 'extended_entities',
                                    'quoted_status_permalink', 'possibly_sensitive', 'retweeted', 'favorited',
                                    'quoted_status', 'is_quote_status', 'quote_status_id', 'geo_enabled',
                                    'in_reply_to_screen_name', 'in_reply_to_user_id', 'in_reply_to_status_id',
                                    'truncated', 'media', 'urls', 'bounding_box', 'attributes', 'symbols', 'name',
                                    'screen_name', 'translator_type'])
                collection.insert_one(data)
            except json.decoder.JSONDecodeError as err:
                print("JSONDecodeError:", err)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time: ", elapsed_time, file_name)
