from data_load import *

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


def clean():
    array = []
    for filename in os.listdir(directory):
        array.append(filename)
    counter = 0

    for file_name in array:
        start_time = time.time()
        counter += 1

        with open(f'{directory}/{str(file_name)}') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    remove_nested_keys(data,
                                       ['id', 'source', 'url', 'description', 'protected', 'friends_count', 'listed_count',
                                        'favourites_count', 'utc_offset', 'time_zone', 'contributors_enabled',
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
                    try:
                        del(data['user']['created_at'])
                        del(data['user']['lang'])
                        del(data['retweeted_status']['display_text_range'])
                        del(data['retweeted_status']['extended_tweet'])
                        del(data['retweeted_status']['created_at'])
                        del(data['retweeted_status']['text'])
                        del(data['retweeted_status']['in_reply_to_status_id_str'])
                        del(data['retweeted_status']['coordinates'])
                        del(data['retweeted_status']['place'])
                        del(data['retweeted_status']['quote_count'])
                        del(data['retweeted_status']['reply_count'])
                        del(data['retweeted_status']['retweet_count'])
                        del(data['retweeted_status']['favorite_count'])
                        del(data['retweeted_status']['entities'])
                        del(data['retweeted_status']['filter_level'])
                        del(data['retweeted_status']['user']['location'])
                        del(data['retweeted_status']['user']['verified'])
                        del(data['retweeted_status']['user']['followers_count'])
                        del(data['retweeted_status']['user']['statuses_count'])
                        del(data['retweeted_status']['user']['created_at'])
                        try:
                            for obj_numb in range(0, 6):
                                del (data['entities']['user_mentions'][obj_numb]['screen_name'])
                                del (data['entities']['user_mentions'][obj_numb]['name'])
                                del (data['entities']['user_mentions'][obj_numb]['id'])
                        except:
                            pass
                    except:
                        pass
                    collection.insert_one(data)
                except json.decoder.JSONDecodeError as err:
                    print("JSONDecodeError:", err)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Time: ", elapsed_time, file_name)


# Removes all deleted tweets and duplicates and indexes id_str if necessary
def clean_2(coll):
    delete_duplicates(coll)
    delete_deleted(coll)
    indices = collection.list_indexes()
    for index in indices:
        if 'id_str' in index.get('key'):
            print('id_str is already an index')
            return
    print('Indexing id_str')
    collection.create_index("id_str", unique=True)


# Deletes all duplicates based on id_str
def delete_duplicates(coll):
    print('Start removing duplicates')
    # Get all documents with a duplicate id_str
    pipeline = [
        {"$group": {"_id": "$id_str", "count": {"$sum": 1}, "ids": {"$push": "$_id"}}},
        {"$match": {"_id": {"$ne": None}, "count": {"$gt": 1}}}
    ]
    duplicates = coll.aggregate(pipeline)

    # Delete all but one for every duplicate
    result = coll.delete_many({"_id": {"$in": [_id for duplicate in duplicates for _id in duplicate['ids'][1:]]}})
    print("Deleted duplicates:", result.deleted_count)


# Deletes all documents that are deleted tweets
def delete_deleted(coll):
    # Get all documents with a duplicate id_str and delete them
    print('Start removing deleted tweets')
    duplicates = coll.find({"id_str": None})
    result = coll.delete_many({"_id": {"$in": [duplicate['_id'] for duplicate in duplicates]}})
    print("Deleted count:", result.deleted_count)
