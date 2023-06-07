import pandas as pd
import matplotlib.pyplot as plt
import numpy
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


    try:    # Remove 'id_str' key and it's value if there is such a key
        for topic, sentiment in topic_result.items():
            del (topic_result[topic]['id_str'])
    except:
        pass

    sentiment_ratio_dict = dict()
    # Turn dictionary with value into dictionary with ratios
    for topic, sentiment in topic_result.items():
        try:
            sentiment_ratio = {k: sentiment[k] / sum(sentiment[k] for k in sentiment) for k in sentiment}
        except:
            sentiment_ratio = {k: 0.00 for k in sentiment}
        if topic not in sentiment_ratio_dict:
            sentiment_ratio_dict[topic] = sentiment_ratio

    # Switch keys with nested keys in a dictionary
    sentiment_ratio_dict_switched = {}
    for topic, sentiment in sentiment_ratio_dict.items():
        for label, value in sentiment.items():
            sentiment_ratio_dict_switched.setdefault(label, {})[topic] = value

    df = pd.DataFrame(sentiment_ratio_dict_switched)
    df = df.loc[(df != 0).any(axis=1)]

    # Set order of x-axis
    order = ['daily_life', 'business_&_entrepreneurs', 'pop_culture', 'science_&_technology', 'sports_&_gaming',
             'arts_&_culture', 'Uncertain']
    df = df.loc[order]

    df[['positive', 'negative', 'neutral', 'uncertain']].plot(kind="bar", stacked=True,
                                                              color=['mediumspringgreen', 'orangered', 'cornflowerblue',
                                                                     'dimgrey'])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    if type == 1:
        plt.title('Topic classification and sentiment ratio for tweets directed at British Airways')
    elif type == 2:
        plt.title('Topic classification and sentiment ratio for tweets directed at Virgin Atlantic')
    else:
        raise ValueError('Put either 1 or 2 as type - see docstring')
    plt.ylabel('Ratio of total')
    plt.tight_layout()

    # Custom legend placement
    ax = plt.subplot(111)
    ax.legend(bbox_to_anchor=(1.01, 0.72))

    # Grid
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

    try:    # Remove 'id_str' key and it's value if there is such a key
        for topic, sentiment in topic_result.items():
            del (topic_result[topic]['id_str'])
    except:
        pass

    # Make dictionary with total amount of tweets per topic
    total_per_topic = dict()
    for topic, sentiment in topic_result.items():
        if topic not in total_per_topic:
            total_per_topic[topic] = sum(sentiment.values())

    # Delete uncertain key
    del (total_per_topic['Uncertain'])

    # Sort key-value pairs from higher value to lowest value (to control order)
    total_per_topic_sort = sorted(total_per_topic.items(), key=lambda x: x[1], reverse=True)

    topic = [key for key, value in total_per_topic_sort]
    amount = numpy.array([value for key, value in total_per_topic_sort])
    colors = ['royalblue', 'limegreen', 'orangered', 'cyan', 'brown', 'dimgrey']
    explode = [0.0, 0.0, 0.0, 0.0, 0.23, 0.5]

    def show_values(val):
        """
        Shows values in a pie chart
        :param values: numbers used for the pie chart
        :return: values in the pies of the pie chart
        """
        a = numpy.round(val / 100. * amount.sum(), 0)
        return a

    def show_values_and_percentage(values):
        """
        Shows values and percentages in a pie chart
        :param values: numbers used for the pie chart
        :return: values and percentages in the pies of the pie chart
        """
        def my_format(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{:.1f}%\n({v:d})'.format(pct, v=val)

        return my_format

    plt.pie(amount, labels=topic, startangle=180, autopct=show_values_and_percentage(amount), explode=explode,
            colors=colors)

    if type == 1:
        plt.title('Topic classification share for tweets directed at British Airways')
    elif type == 2:
        plt.title('Topic classification share for tweets directed at Virgin Atlantic')
    else:
        raise ValueError('Put either 1 or 2 as type - see docstring')
    plt.tight_layout()

    # Custom legend placement
    ax = plt.subplot(111)
    ax.legend(bbox_to_anchor=(0.92, 0.70))

    if type == 1:
        plt.savefig(f'plots/Pie chart Topic British Airways', bbox_inches='tight')
    elif type == 2:
        plt.savefig(f'plots/Pie chart Topic Virgin Atlantic', bbox_inches='tight')
    else:
        raise ValueError('Put either 1 or 2 as type - see docstring')
    plt.show()

def stacked_bar_chart_sentiment_ratio(sentiment_result):
    try:    # Remove 'id_str' key and it's value if there is such a key
        for airline, sentiment in sentiment_result.items():
            del (sentiment_result[airline]['id_str'])
    except:
        pass

    # Make dictionary with values for airlines other than our airline and its competitor
    sentiment_dict_other = dict()
    for airline, sentiment in sentiment_result.items():
        if (airline != 'British_Airways') and (airline != 'VirginAtlantic'):
            for label, value in sentiment.items():
                if label in sentiment_dict_other:
                    sentiment_dict_other[label] += sentiment[label]
                else:
                    sentiment_dict_other[label] = sentiment[label]

    # Make new dictionary with values for our airline, its competitor and all other airlines.
    sentiment_dict_new = dict()
    for airline, sentiment in sentiment_result.items():
        if (airline == 'British_Airways') or (airline == 'VirginAtlantic'):
            if airline not in sentiment_dict_new:
                sentiment_dict_new[airline] = sentiment
    sentiment_dict_new['Other'] = sentiment_dict_other

    # Turn dictionary with value into dictionary with ratios
    sentiment_ratio_dict = dict()
    for topic, sentiment in sentiment_dict_new.items():
        try:
            sentiment_ratio = {k: sentiment[k] / sum(sentiment[k] for k in sentiment) for k in sentiment}
        except:
            sentiment_ratio = {k: 0.00 for k in sentiment}
        if topic not in sentiment_ratio_dict:
            sentiment_ratio_dict[topic] = sentiment_ratio

    # Switch keys with nested keys in a dictionary
    sentiment_ratio_dict_switched = {}
    for airline, sentiment in sentiment_ratio_dict.items():
        for label, value in sentiment.items():
            sentiment_ratio_dict_switched.setdefault(label, {})[airline] = value

    df = pd.DataFrame(sentiment_ratio_dict_switched)
    df = df.loc[(df != 0).any(axis=1)]

    df[['positive', 'negative', 'neutral', 'uncertain']].plot(kind="bar", stacked=True,
                                                              color=['mediumspringgreen', 'orangered', 'cornflowerblue',
                                                                     'dimgrey'])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title('Sentiment ratio for tweets directed towards airlines')
    plt.ylabel('Ratio of total')
    plt.tight_layout()

    # Custom legend placement
    ax = plt.subplot(111)
    ax.legend(bbox_to_anchor=(1.01, 0.65))

    # Grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    plt.savefig(f'plots/Stacked bar chart ratio sentiment TO', bbox_inches='tight')
    plt.show()

def multiple_bar_chart_sentiment_numbers(sentiment_result):
    try:    # Remove 'id_str' key and it's value if there is such a key
        for airline, sentiment in sentiment_result.items():
            del (sentiment_result[airline]['id_str'])
    except:
        pass

    # Make dictionary with values for airlines other than our airline and its competitor
    sentiment_dict_other = dict()
    for airline, sentiment in sentiment_result.items():
        if (airline != 'British_Airways') and (airline != 'VirginAtlantic'):
            for label, value in sentiment.items():
                if label in sentiment_dict_other:
                    sentiment_dict_other[label] += sentiment[label]
                else:
                    sentiment_dict_other[label] = sentiment[label]

    # Order of which items will be put into a list
    desired_order_list = ['positive', 'negative', 'neutral', 'uncertain']


    for airline, sentiment in sentiment_result.items():
        if (airline == 'British_Airways'):
            reordered_sentiment_british = {k: sentiment[k] for k in desired_order_list}
            british_sent = list(reordered_sentiment_british.values())
        elif (airline == 'VirginAtlantic'):
            reordered_sentiment_virgin = {k: sentiment[k] for k in desired_order_list}
            virgin_sent = list(reordered_sentiment_virgin.values())
    reordered_sentiment_other = {k: sentiment_dict_other[k] for k in desired_order_list}
    other_sent = list(reordered_sentiment_other.values())

    # set width of bars
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = numpy.arange(len(british_sent))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, british_sent, color='royalblue', width=barWidth,
            edgecolor='grey', label='British Airways')
    plt.bar(br2, virgin_sent, color='limegreen', width=barWidth,
            edgecolor='grey', label='Virgin Atlantic')
    plt.bar(br3, other_sent, color='grey', width=barWidth,
            edgecolor='grey', label='Other airways')

    plt.title('Sentiment amount for tweets directed towards airlines')
    plt.xlabel('Sentiment', fontweight='bold', fontsize=15)
    plt.ylabel('Tweet amount', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(british_sent))],
               ['Positive', 'Negative', 'Neutral', 'Uncertain'])

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tight_layout()

    plt.legend()

    ax = plt.subplot(111)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    plt.savefig(f'plots/Multiple bar chart sentiment TO', bbox_inches='tight')
    plt.show()
