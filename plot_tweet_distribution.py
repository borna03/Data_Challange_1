from data_load import *
from data_for_plots4 import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime


def tweet_sentiment_count_single(chosen_airline_id):
    tweet_count_per_day = {}

    for doc in collection.find({
        'entities.user_mentions': {
            '$elemMatch': {'id_str': chosen_airline_id}
        }
    }):
        created_at = doc['created_at']
        created_at_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")  # Parse the date string
        if created_at_date.year == 2019 and 6 <= created_at_date.month <= 12:
            day = created_at_date.strftime('%Y-%m-%d')  # Format the date as "YYYY-MM-DD"
            tweet_count_per_day.setdefault(day, {'positive': 0, 'negative': 0, 'neutral': 0, 'uncertain': 0})
            sentiment = doc['sentiment']
            tweet_count_per_day[day][sentiment] += 1
        elif created_at_date.year == 2020 and created_at_date.month <= 3:
            day = created_at_date.strftime('%Y-%m-%d')  # Format the date as "YYYY-MM-DD"
            tweet_count_per_day.setdefault(day, {'positive': 0, 'negative': 0, 'neutral': 0, 'uncertain': 0})
            sentiment = doc['sentiment']
            tweet_count_per_day[day][sentiment] += 1

    return tweet_count_per_day


# print(tweet_sentiment_count_single("20626359"))
# print(tweet_sentiment_count_single("18332190"))

def tweet_sentiment_count_multiple(included_airline_ids):
    tweet_count_per_day = {}

    for doc in collection.find({
        'entities.user_mentions': {
            '$elemMatch': {'id_str': {'$in': included_airline_ids}}
        }
    }):
        created_at = doc['created_at']
        created_at_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")  # Parse the date string
        if created_at_date.year == 2019 and 6 <= created_at_date.month <= 12:
            day = created_at_date.strftime('%Y-%m-%d')  # Format the date as "YYYY-MM-DD"
            tweet_count_per_day.setdefault(day, {'tweet_count': 0, 'positive': 0, 'negative': 0, 'neutral': 0,
                                                 'uncertain': 0})
            sentiment = doc['sentiment']
            tweet_count_per_day[day][sentiment] += 1
            tweet_count_per_day[day]['tweet_count'] += 1
        elif created_at_date.year == 2020 and created_at_date.month <= 3:
            day = created_at_date.strftime('%Y-%m-%d')  # Format the date as "YYYY-MM-DD"
            tweet_count_per_day.setdefault(day, {'tweet_count': 0, 'positive': 0, 'negative': 0, 'neutral': 0,
                                                 'uncertain': 0})
            sentiment = doc['sentiment']
            tweet_count_per_day[day][sentiment] += 1
            tweet_count_per_day[day]['tweet_count'] += 1

    return tweet_count_per_day


chosen_airline_ids = ["56377143", "106062176", "22536055", "124476322", "26223583", "2182373406",
                      "38676903", "1542862735", "253340062", "218730857", "45621423"]  # list of chosen airline IDs

# print(tweet_sentiment_count_multiple(chosen_airline_ids))


def plot_sentiment_count_virgin(sentiment_count):
    dates = list(sentiment_count.keys())
    sentiments = list(sentiment_count.values())
    positive = [item['positive'] for item in sentiments]
    negative = [item['negative'] for item in sentiments]
    neutral = [item['neutral'] for item in sentiments]
    uncertain = [item['uncertain'] for item in sentiments]

    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    fig, ax = plt.subplots()

    ax.stackplot(dates, positive, negative, neutral, uncertain, labels=['Positive', 'Negative', 'Neutral', 'Uncertain'])

    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.set_ylabel('Sentiment Count', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_title('Sentiment Count for VirginAtlantic')
    ax.legend(loc='upper left')
    plt.xticks(rotation=45)

    covid_regulation_date = datetime.strptime('2020-03-01', '%Y-%m-%d')
    thanksgiving_date = datetime.strptime('2019-11-25', '%Y-%m-%d')

    ax.annotate('Covid-19 regulation', xy=(covid_regulation_date, 100), xytext=(covid_regulation_date, 300),
                arrowprops=dict(facecolor='red', arrowstyle='->', linewidth=2), fontsize=12, annotation_clip=False,
                ha='right', va='bottom')

    ax.annotate('Thanksgiving', xy=(thanksgiving_date, 200), xytext=(thanksgiving_date, 450),
                arrowprops=dict(facecolor='green', arrowstyle='->', linewidth=2), fontsize=12, annotation_clip=False,
                ha='right', va='bottom')

    plt.tight_layout()
    plt.show()


#plot_sentiment_count_virgin(tweet_sentiment_count_virgin)

def plot_sentiment_count_british(sentiment_count):
    dates = list(sentiment_count.keys())
    sentiments = list(sentiment_count.values())
    positive = [item['positive'] for item in sentiments]
    negative = [item['negative'] for item in sentiments]
    neutral = [item['neutral'] for item in sentiments]
    uncertain = [item['uncertain'] for item in sentiments]

    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    fig, ax = plt.subplots()

    ax.stackplot(dates, positive, negative, neutral, uncertain, labels=['Positive', 'Negative', 'Neutral', 'Uncertain'])

    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.set_ylabel('Sentiment Count')
    ax.set_xlabel('Date')
    ax.set_title('Sentiment Count for British Airways')
    ax.legend(loc='upper left')
    plt.xticks(rotation=45)

    system_f = datetime.strptime('2019-08-01', '%Y-%m-%d')
    covid_regulations = datetime.strptime('2020-03-01', '%Y-%m-%d')

    ax.annotate('System Failure', xy=(system_f, 100), xytext=(system_f, 2500),
                arrowprops=dict(facecolor='red', arrowstyle='->', linewidth=2), fontsize=12, annotation_clip=False,
                ha='right', va='bottom')

    ax.annotate('Covid-19 Regulations', xy=(covid_regulations, 200), xytext=(covid_regulations, 2000),
                arrowprops=dict(facecolor='green', arrowstyle='->', linewidth=2), fontsize=12, annotation_clip=False,
                ha='right', va='bottom')

    plt.tight_layout()
    plt.show()


#plot_sentiment_count_british(tweet_sentiment_count_british)


def plot_sentiment_count_others(sentiment_count):
    dates = list(sentiment_count.keys())
    sentiments = list(sentiment_count.values())
    positive = [item['positive'] for item in sentiments]
    negative = [item['negative'] for item in sentiments]
    neutral = [item['neutral'] for item in sentiments]
    uncertain = [item['uncertain'] for item in sentiments]

    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    fig, ax = plt.subplots()

    ax.stackplot(dates, positive, negative, neutral, uncertain, labels=['Positive', 'Negative', 'Neutral', 'Uncertain'])

    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.set_ylabel('Sentiment Count')
    ax.set_xlabel('Date')
    ax.set_title('Sentiment Count for Other Airlines')
    ax.legend(loc='upper left')
    plt.xticks(rotation=45)

    baggage = datetime.strptime('2019-08-01', '%Y-%m-%d')
    tarmac_delays = datetime.strptime('2019-07-01', '%Y-%m-%d')
    covid_regulations = datetime.strptime('2020-03-01', '%Y-%m-%d')

    ax.annotate('Mishandled Baggage', xy=(baggage, 100), xytext=(baggage + pd.DateOffset(days=3), 7000),
                arrowprops=dict(facecolor='red', arrowstyle='->', linewidth=2), fontsize=12, annotation_clip=False,
                ha='right', va='bottom')
    ax.annotate('Tarmac Delays', xy=(tarmac_delays, 100), xytext=(tarmac_delays + pd.DateOffset(days=2), 6500),
                arrowprops=dict(facecolor='red', arrowstyle='->', linewidth=2), fontsize=12, annotation_clip=False,
                ha='right', va='bottom')

    ax.annotate('Covid-19 Regulations', xy=(covid_regulations, 200), xytext=(covid_regulations, 7000),
                arrowprops=dict(facecolor='green', arrowstyle='->', linewidth=2), fontsize=12, annotation_clip=False,
                ha='right', va='bottom')

    plt.tight_layout()
    plt.show()


plot_sentiment_count_others(tweet_sentiment_count_others)
