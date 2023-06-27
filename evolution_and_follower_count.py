from data_load import *
#from conv_collection import progress_log, debug_mode, evaluate
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from datetime import date

start_date = '2009-09-15'
end_date = '2029-05-23'
all_airline_ids = set(airlines[airline]['id_str'] for airline in airlines)
virgin_atlantic_id = {'20626359'}
british_airways_id = {'18332190'}
other_airline_ids = all_airline_ids.difference(virgin_atlantic_id).difference(british_airways_id)

def create_tweets_df(collection, start_date:str=None, end_date:str=None):
    tweets = pd.DataFrame(list(collection.find({'conversation_id': {'$ne': 0}}, 
                               {'_id':0, 'created_at':1, 'id_str':1, 'in_reply_to_user_id_str':1, 'in_reply_to_status_id_str':1, 'user.followers_count':1, 'user.id_str':1, 'replies':1, 'sentiment':1, 'conversation_id':1, 'sub-conversation_id':1})))
    date_format = '%a %b %d %H:%M:%S +%f %Y'
    tweets['user.id_str'] = tweets['user'].apply(lambda cell: cell['id_str'])
    tweets['user.followers_count'] = tweets['user'].apply(lambda cell: cell['followers_count'])
    tweets['created_at'] = pd.to_datetime(tweets['created_at'], format=date_format)
    tweets = tweets[(tweets['created_at'] >= start_date) & (tweets['created_at'] <= end_date)]
    return tweets

def calc_sent(df, sentiment:str, n:int):
    count = 0
    for i in range(n):
        count += df[f'sentiment_{n-1}'].eq(sentiment).sum()
    return count

def reply_chain_sentiment(tweets, airline_id, mode:int=0):
    n = 0
    columns = ['id_str', 'user.id_str', 'sentiment']
    sent_count = dict()
    df_a = tweets.copy()[tweets['in_reply_to_status_id_str'].notnull()][['id_str', 'in_reply_to_status_id_str', 'user.id_str', 'sentiment']]
    df = tweets.copy()[tweets['in_reply_to_status_id_str'].isnull()][['id_str', 'user.id_str', 'sentiment']]
    while n < 20:
        df = pd.merge(df[columns], df_a.copy(), left_on='id_str', right_on='in_reply_to_status_id_str', suffixes=[f'_{n}',''], how='left')
        #df.drop(df_b.filter(regex='in_rely_to_status_id_str$').columns, axis=1, inplace=True)
        columns.extend([f'user.id_str_{n}', f'sentiment_{n}'])
        n += 1
    df = df[df.isin(airline_id).any(axis=1)]
    
    for i in range(n):
        if i == n:
            column = 'sentiment'
        else:
            column = f'sentiment_{i}'
        values = {'positive': df[column].eq('positive').sum(), 
                  'negative': df[column].eq('negative').sum(), 
                  'neutral': df[column].eq('neutral').sum(), 
                  'uncertain': df[column].eq('uncertain').sum()} 
        total = sum(values.values())
        values['positive'] = values['positive'] / total * 100
        values['negative'] = values['negative'] / total * 100
        values['neutral'] = values['neutral'] / total * 100
        values['uncertain'] = values['uncertain'] / total * 100
        sent_count[i] = values

    sc_df = pd.DataFrame(sent_count)
    print(sc_df)
    return sc_df

def plot_evolution(df, ax, title:str=''):
    df = df.transpose()
    df[['negative', 'positive', 'neutral', 'uncertain']].plot(kind="bar", stacked=True,
                                                              color=['orangered', 'mediumspringgreen', 'cornflowerblue',
                                                                     'dimgrey'], ax=ax)
    #plt.xticks(range(len(column)), column)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.set_xticklabels(df.index, rotation=0)
    ax.set_title(title, pad=14)
    ax.set_xlabel('Tweet level')
    ax.get_legend().remove()

def plot_followers_response_time(tweets, airline_id, title, ax):
    replies_df = tweets.copy()
    root_df = tweets.copy()
    replies_df = replies_df[(replies_df['in_reply_to_status_id_str'].notnull()) & (replies_df['user.id_str'].isin(airline_id))][['in_reply_to_status_id_str', 'created_at']]
    print(replies_df.head())
    root_df = root_df[root_df['in_reply_to_status_id_str'].isnull()][['id_str','user.followers_count', 'created_at']]
    difference_df = pd.merge(replies_df, root_df, left_on='in_reply_to_status_id_str', right_on='id_str', suffixes=['_reply',''], how='inner')
    print(difference_df.head())
    difference_df['difference'] = difference_df['created_at_reply'] - difference_df['created_at']
    difference_df['difference'] = difference_df['difference'].dt.total_seconds().div(60)
    ax.scatter(difference_df['difference'],difference_df['user.followers_count'], s=0.4, alpha=0.3)
    
    ax.set_title(title, pad=14)
    ax.set_ylabel('Follower count')
    ax.set_xlabel('Responsetime in minutes')
    ax.set_xlim([0, 180])
    ax.set_yscale('log')

def plot_popularity_figures(tweets):
    # Scatter followers vs response time
    #plt.savefig('plot.pdf')
    # Bar followers and sentiment tweet
    sent_count = dict()
    sent_count_df = tweets.copy()
    sent_count_df = sent_count_df[['user.followers_count', 'user.id_str', 'sentiment']]
    sent_count_df = sent_count_df[~sent_count_df['user.id_str'].isin(all_airline_ids)]
    for n in range(7):
        df = []
        df = sent_count_df.copy()
        df = df[(10**n <= df['user.followers_count'].astype(int)) & (df['user.followers_count'].astype(int) < 10**(n+1))]
        values = {'positive': df['sentiment'].eq('positive').sum() / len(df.index) * 100, 
                  'negative': df['sentiment'].eq('negative').sum() / len(df.index) * 100,
                  'neutral': df['sentiment'].eq('neutral').sum() / len(df.index) * 100,
                  'uncertain': df['sentiment'].eq('uncertain').sum() / len(df.index) * 100}
        sent_count[f'$10^{n}$'] = values
    sc_df = pd.DataFrame(sent_count).transpose()
    print(sc_df)
    ax2 = sc_df[['negative', 'positive', 'neutral', 'uncertain']].plot(kind="bar", stacked=True,
                                                              color=['orangered', 'mediumspringgreen', 'cornflowerblue',
                                                                     'dimgrey'])
    #plt.xticks(range(len(column)), column)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax2.yaxis.set_major_formatter(yticks)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4,)
    ax2.set_xticklabels(sc_df.index, rotation=0)
    ax2.set_title('Follower count and sentiment', pad=28)
    ax2.set_xlabel('Follower count')

    # Make the plot
    #plt.plot(british_airways_df.index.values.tolist(), british_airways_df[0]/2, label='British Airways')
    
    # Bar followers and sentiment reactions
    replies_df = tweets.copy()[['user.followers_count', 'user.id_str', 'in_reply_to_status_id_str', 'sentiment']]
    root_df = tweets.copy()[['id_str', 'user.id_str']]
    sc_df = pd.merge(replies_df, root_df, left_on='in_reply_to_status_id_str', right_on='id_str', how='inner', suffixes=['_replies', ''])
    sc_df = sc_df[(~sc_df['user.id_str'].isin(all_airline_ids)) | (~sc_df['user.id_str_replies'].isin(all_airline_ids))]
    print(sc_df.head())
    for n in range(7):
        df = []
        df = sent_count_df.copy()
        df = df[(10**n <= df['user.followers_count'].astype(int)) & (df['user.followers_count'].astype(int) < 10**(n+1))]
        values = {'positive': df['sentiment'].eq('positive').sum() / len(df.index) * 100, 
                  'negative': df['sentiment'].eq('negative').sum() / len(df.index) * 100,
                  'neutral': df['sentiment'].eq('neutral').sum() / len(df.index) * 100,
                  'uncertain': df['sentiment'].eq('uncertain').sum() / len(df.index) * 100}
        sent_count[f'$10^{n}$'] = values

    sc_df = pd.DataFrame(sent_count).transpose()
    print(sc_df)
    ax3 = sc_df[['negative', 'positive', 'neutral', 'uncertain']].plot(kind="bar", stacked=True,
                                                              color=['orangered', 'mediumspringgreen', 'cornflowerblue',
                                                                     'dimgrey'])
    #plt.xticks(range(len(column)), column)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax3.yaxis.set_major_formatter(yticks)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    ax3.set_xticklabels(sc_df.index, rotation=0)
    ax3.set_title('Follower count and sentiment of replies', pad=28)
    ax3.set_xlabel('Follower count')




def plot_all_evol_figures(collection, start_date, end_date):
    tweets = create_tweets_df(collection, start_date, end_date)
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    plot_followers_response_time(tweets, virgin_atlantic_id, 'Virgin Atlantic', ax2[0])
    plot_followers_response_time(tweets, british_airways_id, 'British Airways', ax2[1])
    #plot_followers_response_time(tweets, other_airline_ids, 'Other Airlines', ax2[2])
    plt.suptitle('Responsetime vs Follower count', fontsize=16, fontweight=700)
    fig2.subplots_adjust(top=0.85, bottom = 0.15)

    plot_popularity_figures(tweets)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    plot_evolution(reply_chain_sentiment(tweets, virgin_atlantic_id), ax[0], 'Virgin Atlantic')
    plot_evolution(reply_chain_sentiment(tweets, british_airways_id), ax[1], 'British Airways')
    plot_evolution(reply_chain_sentiment(tweets, other_airline_ids), ax[2], 'Other Airlines')
    plt.suptitle('Sentiment evolution', fontsize=16, fontweight=700)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.85, bottom = 0.2)
    fig.legend(handles, labels, bbox_to_anchor=[0.5, 0.05], loc='center', ncol=4)
    plt.show()


plot_all_evol_figures(collection, start_date, end_date)
