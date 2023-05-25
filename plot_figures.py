from data_load import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime


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


def replied_at_count_month(chosen_airline_id):
    print('Starting function')
    reply_count_per_month = {}
    responses = [response for response in
                 collection.find({'user.id_str': chosen_airline_id, 'in_reply_to_status_id_str': {'$ne': None}},
                                 {'in_reply_to_status_id_str': 1, "_id": 0}).distinct('in_reply_to_status_id_str')]
    print('Found responses')
    for tweet in collection.find({'entities.user_mentions': {'$elemMatch': {'id_str': chosen_airline_id}},
                                  'in_reply_to_status_id_str': None}, {'id_str': 1, 'created_at': 1, '_id': 0}):
        if tweet['id_str'] in responses:
            print('Found tweet')
            created_at = tweet['created_at']
            created_at_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")  # Parse the date string
            if (created_at_date.year == 2019 and 6 <= created_at_date.month <= 12) or (
                    created_at_date.year == 2020 and created_at_date.month <= 3):
                month_year = created_at_date.strftime('%Y-%m')  # Format the date as "YYYY-MM"
                reply_count_per_month.setdefault(month_year, 0)
                reply_count_per_month[month_year] += 1
            print('Tweet handled')

    return reply_count_per_month


def tweeted_at_lang(id_str):
    langs = {}
    for tweet in collection.find({'entities.user_mentions': {'$elemMatch': {'id_str': id_str}}}):
        if tweet['lang'] in langs:
            langs[tweet['lang']] += 1
        else:
            langs[tweet['lang']] = 1
    return langs


def responded_to_lang(id_str):
    langs = {}
    responses = [response for response in
                 collection.find({'user.id_str': id_str, 'in_reply_to_status_id_str': {'$ne': None}},
                                 {'in_reply_to_status_id_str': 1, "_id": 0}).distinct('in_reply_to_status_id_str')]

    for tweet in collection.find({'entities.user_mentions': {'$elemMatch': {'id_str': id_str}},
                                  'in_reply_to_status_id_str': None}, {'id_str': 1, 'lang': 1, '_id': 0}):
        if tweet['id_str'] in responses:
            if tweet['lang'] in langs:
                langs[tweet['lang']] += 1
            else:
                langs[tweet['lang']] = 1
    return langs


def plot_bar_chart():
    # creating dataframe
    df = pd.DataFrame({
        'Airline': airlines.keys(),
        'Tweets': [airlines[airline]['tweet_count'] for airline in airlines],
        'Tweeted at': [airlines[airline]['tweeted_at_count'] for airline in airlines]
    })

    # plotting graph
    df.plot(x="Airline", y=["Tweets", "Tweeted at"], kind="bar")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title("Amount of tweets tweeted by and at airlines")
    plt.ylabel('Amount')
    plt.tight_layout()
    plt.show()


def plot_line_plot(tweet_count_per_month_us, reply_count_per_month_us, tweet_count_per_month_them,
                   reply_count_per_month_them):
    percentage_dict_our_airline = {}
    percentage_dict_competitor_airline = {}

    for tweet, reply in zip(tweet_count_per_month_us, reply_count_per_month_us):
        ratio = reply_count_per_month_us[reply] / tweet_count_per_month_us[tweet] * 100
        percentage_dict_our_airline[tweet] = ratio

    for tweet, reply in zip(tweet_count_per_month_them, reply_count_per_month_them):
        ratio = reply_count_per_month_them[reply] / tweet_count_per_month_them[tweet] * 100
        percentage_dict_competitor_airline[tweet] = ratio

    # Extract x-axis (keys) and y-axis (values) from the dictionaries
    x_values = list(percentage_dict_our_airline.keys())
    y_values1 = list(percentage_dict_our_airline.values())
    y_values2 = list(percentage_dict_competitor_airline.values())

    # Plot the line plots
    plt.plot(x_values, y_values1, marker='o', label='VirginAtlantic')
    plt.plot(x_values, y_values2, marker='o', label='British_Airways')

    plt.xlabel('Month')
    plt.ylabel('Percentage of replies')
    plt.title('Line Plot of replies from VirginAtlantic')
    plt.legend()

    # Rotate x-axis labels if needed
    plt.xticks(rotation=30)

    plt.show()


def plot_nested_pie(values):
    fig, ax = plt.subplots()

    size = 0.4
    vals = np.array(values)

    outer_colors = ['royalblue', 'forestgreen', 'firebrick']
    inner_colors = ['silver', 'gold']

    ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
           wedgeprops=dict(width=size, edgecolor='w'))

    ax.pie(vals.flatten(), radius=1 - size, colors=inner_colors,
           wedgeprops=dict(width=size, edgecolor='w'))

    ax.set(aspect="equal", title='Nested Pie Chart')

    # Combine outer and inner labels
    outer_labels = ['English', 'Undefined', 'ROW']
    inner_labels = ['Ignored', 'Replied']
    all_labels = outer_labels + inner_labels

    # Combine outer and inner colors
    all_colors = outer_colors + inner_colors

    # Add legend
    ax.legend(all_labels, title="Legend", loc="upper right", bbox_to_anchor=(0.2, 1))

    ax.set(aspect="equal", title='Replies in different languages (British Airways)')

    plt.show()
