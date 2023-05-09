from config import *
from datetime import datetime
from datetime import datetime
from collections import defaultdict


def tweet_count(id_str):
    return collection.count_documents({'user.id_str': id_str})


def tweeted_at_count(id_str):
    return collection.count_documents({'entities.user_mentions': {'$elemMatch': {'id_str': id_str}}})



def tweeted_at_count_month(chosen_airline_id):
    tweet_count_per_month = {}

    for doc in collection.find({
        'entities.user_mentions': {
            '$elemMatch': {'id_str': chosen_airline_id}
        }
    }):
        created_at = doc['created_at']
        created_at_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")  # Parse the date string
        if created_at_date.year == 2019 and 6 <= created_at_date.month <= 12:
            month_year = created_at_date.strftime('%Y-%m')  # Format the date as "YYYY-MM"
            tweet_count_per_month.setdefault(month_year, 0)
            tweet_count_per_month[month_year] += 1
        elif created_at_date.year == 2020 and created_at_date.month <= 3:
            month_year = created_at_date.strftime('%Y-%m')  # Format the date as "YYYY-MM"
            tweet_count_per_month.setdefault(month_year, 0)
            tweet_count_per_month[month_year] += 1

    return tweet_count_per_month

chosen_airline_id = "20626359"
tweet_count_per_month = tweeted_at_count_month(chosen_airline_id)

print("Tweet count per month:")
for month_year, count in tweet_count_per_month.items():
    print(f"Month: {month_year}, Count: {count}")


def replied_at_count_month(chosen_airline_id):
    reply_count_per_month = {}

    for doc in collection.find({
        'entities.user_mentions': {
            '$elemMatch': {'id_str': chosen_airline_id}
        }
    }):
        if doc['in_reply_to_user_id_str'] == chosen_airline_id:
            created_at = doc['created_at']
            created_at_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")  # Parse the date string
            if created_at_date.year == 2019 and 6 <= created_at_date.month <= 12:
                month_year = created_at_date.strftime('%Y-%m')  # Format the date as "YYYY-MM"
                reply_count_per_month.setdefault(month_year, 0)
                reply_count_per_month[month_year] += 1
            elif created_at_date.year == 2020 and created_at_date.month <= 3:
                month_year = created_at_date.strftime('%Y-%m')  # Format the date as "YYYY-MM"
                reply_count_per_month.setdefault(month_year, 0)
                reply_count_per_month[month_year] += 1

    print("Reply count per month:")
    for month_year, count in reply_count_per_month.items():
        print(f"Month: {month_year}, Count: {count}")

    return reply_count_per_month

chosen_airline_id = "20626359"
reply_count_per_month = replied_at_count_month(chosen_airline_id)

def tweet_and_reply_count_month(chosen_airline_id):
    mention_count_per_month = {}
    reply_count_per_month = {}

    for doc in collection.find({
        'entities.user_mentions': {
            '$elemMatch': {'id_str': chosen_airline_id}
        }
    }):
        created_at = doc['created_at']
        created_at_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")  # Parse the date string
        if (created_at_date.year == 2019 and 6 <= created_at_date.month <= 12) or (created_at_date.year == 2020 and created_at_date.month <= 3):
            month_year = created_at_date.strftime('%Y-%m')  # Format the date as "YYYY-MM"
            mention_count_per_month.setdefault(month_year, 0)
            mention_count_per_month[month_year] += 1

            if doc['in_reply_to_user_id_str'] == chosen_airline_id:
                reply_count_per_month.setdefault(month_year, 0)
                reply_count_per_month[month_year] += 1

    print("Mention and Reply count per month:")
    for month_year, mention_count in mention_count_per_month.items():
        reply_count = reply_count_per_month.get(month_year, 0)
        print(f"Month: {month_year}, Mention Count: {mention_count}, Reply Count: {reply_count}")

    return mention_count_per_month, reply_count_per_month

chosen_airline_id = "20626359"
mention_count_per_month, reply_count_per_month = tweet_and_reply_count_month(chosen_airline_id)
