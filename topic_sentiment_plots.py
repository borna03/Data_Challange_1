import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_for_plots import *

def stacked_bar_chart_topic_sentiment(type):
    """
    Generates stacked bar chart for topic classification + sentiment.
    Either for British Airways or Virgin Atlantic.
    :param type: 1 if British Airways, 2 if Virgin Atlantic
    """
    if type == 1:
        topic_result = topic_sentiment_to_British
    elif type == 2:
        topic_result = topic_sentiment_to_Virgin
    else:
        raise ValueError('Put either 1 or 2 as type - see docstring')

    try:
        for topic, sentiment in topic_result.items():
            del(topic_result[topic]['id_str'])
    except:
        pass

    sentiment_ratio_dict = dict()
    for topic, sentiment in topic_result.items():
        try:
            sentiment_ratio = {k: sentiment[k] / sum(sentiment[k] for k in sentiment) for k in sentiment}
        except:
            sentiment_ratio = {k: 0.00 for k in sentiment}
        if topic not in sentiment_ratio_dict:
            sentiment_ratio_dict[topic] = sentiment_ratio

    sentiment_ratio_dict_switched = {}
    for topic, sentiment in sentiment_ratio_dict.items():
        for label, value in sentiment.items():
            sentiment_ratio_dict_switched.setdefault(label, {})[topic] = value

    df = pd.DataFrame(sentiment_ratio_dict_switched)
    df = df.loc[(df!=0).any(axis=1)]
    # print(df)

    order = ['daily_life', 'business_&_entrepreneurs', 'pop_culture', 'science_&_technology', 'sports_&_gaming',
             'arts_&_culture', 'Uncertain']
    df = df.loc[order]

    df[['positive', 'negative', 'neutral', 'uncertain']].plot(kind="bar", stacked=True, color=['mediumspringgreen', 'orangered', 'cornflowerblue', 'dimgrey'])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    if type == 1:
        plt.title('Topic classification and sentiment ratio for tweets directed at British Airways')
    elif type == 2:
        plt.title('Topic classification and sentiment ratio for tweets directed at Virgin Atlantic')
    else:
        raise ValueError('Put either 1 or 2 as type - see docstring')
    plt.ylabel('Ratio of total')
    plt.tight_layout()

    ax = plt.subplot(111)
    ax.legend(bbox_to_anchor=(1.01, 0.72))

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    if type == 1:
        plt.savefig(f'plots/Stacked bar chart Topic+Sentiment British Airways', bbox_inches='tight')
    elif type == 2:
        plt.savefig(f'plots/Stacked bar chart Topic+Sentiment Virgin Atlantic', bbox_inches='tight')
    else:
        raise ValueError('Put either 1 or 2 as type - see docstring')
    plt.show()

def pie_chart_topic(type):
    """
    Generates pie chart for topic classification.
    Either for British Airways or Virgin Atlantic.
    :param type: 1 if British Airways, 2 if Virgin Atlantic
    """
    if type == 1:
        topic_result = topic_sentiment_to_British
    elif type == 2:
        topic_result = topic_sentiment_to_Virgin
    else:
        raise ValueError('Put either 1 or 2 as type - see docstring')

    try:
        for topic, sentiment in topic_result.items():
            del (topic_result[topic]['id_str'])
    except:
        pass

    total_per_topic = dict()
    for topic, sentiment in topic_result.items():
        if topic not in total_per_topic:
            total_per_topic[topic] = sum(sentiment.values())

    del(total_per_topic['Uncertain'])
    total_per_topic_sort = sorted(total_per_topic.items(), key=lambda x:x[1], reverse=True)

    topic = [key for key, value in total_per_topic_sort]
    amount = numpy.array([value for key, value in total_per_topic_sort])
    colors = ['royalblue', 'limegreen', 'orangered', 'cyan', 'brown', 'dimgrey']
    explode = [0.0, 0.0, 0.0, 0.0, 0.23, 0.5]

    def show_values(val):
        a = numpy.round(val / 100. * amount.sum(), 0)
        return a

    def show_values_and_percentage(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return my_format

    plt.pie(amount, labels=topic, startangle=180, autopct=show_values_and_percentage(amount), explode=explode, colors=colors)

    if type == 1:
        plt.title('Topic classification share for tweets directed at British Airways')
    elif type == 2:
        plt.title('Topic classification share for tweets directed at Virgin Atlantic')
    else:
        raise ValueError('Put either 1 or 2 as type - see docstring')
    plt.tight_layout()

    ax = plt.subplot(111)
    ax.legend(bbox_to_anchor=(0.92, 0.70))

    if type == 1:
        plt.savefig(f'plots/Pie chart Topic British Airways', bbox_inches='tight')
    elif type == 2:
        plt.savefig(f'plots/Pie chart Topic Virgin Atlantic', bbox_inches='tight')
    else:
        raise ValueError('Put either 1 or 2 as type - see docstring')
    plt.show()

def stacked_bar_chart_sentiment(sentiment_result):
    try:
        for airline, sentiment in sentiment_result.items():
            del (sentiment_result[airline]['id_str'])
    except:
        pass

    sentiment_dict_other = dict()
    for airline, sentiment in sentiment_result.items():
        if (airline != 'British_Airways') and (airline != 'VirginAtlantic'):
            for label, value in sentiment.items():
                if label in sentiment_dict_other:
                    sentiment_dict_other[label] += sentiment[label]
                else:
                    sentiment_dict_other[label] = sentiment[label]

    sentiment_dict_new = dict()
    for airline, sentiment in sentiment_result.items():
        if (airline == 'British_Airways') or (airline == 'VirginAtlantic'):
            if airline not in sentiment_dict_new:
                sentiment_dict_new[airline] = sentiment
    sentiment_dict_new['Other'] = sentiment_dict_other

    sentiment_ratio_dict = dict()
    for topic, sentiment in sentiment_dict_new.items():
        try:
            sentiment_ratio = {k: sentiment[k] / sum(sentiment[k] for k in sentiment) for k in sentiment}
        except:
            sentiment_ratio = {k: 0.00 for k in sentiment}
        if topic not in sentiment_ratio_dict:
            sentiment_ratio_dict[topic] = sentiment_ratio

    sentiment_ratio_dict_switched = {}
    for airline, sentiment in sentiment_ratio_dict.items():
        for label, value in sentiment.items():
            sentiment_ratio_dict_switched.setdefault(label, {})[airline] = value

    df = pd.DataFrame(sentiment_ratio_dict_switched)
    df = df.loc[(df!=0).any(axis=1)]
    print(df)

    df[['positive', 'negative', 'neutral', 'uncertain']].plot(kind="bar", stacked=True, color=['mediumspringgreen', 'orangered', 'cornflowerblue', 'dimgrey'])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title('Sentiment ratio for tweets from airlines')
    plt.ylabel('Ratio of total')
    plt.tight_layout()

    ax = plt.subplot(111)
    ax.legend(bbox_to_anchor=(1.01, 0.65))

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    plt.savefig(f'plots/Stacked bar chart sentiment FROM', bbox_inches='tight')
    plt.show()