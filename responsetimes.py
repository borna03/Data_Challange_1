from data_load import *
from conv_collection import progress_log, debug_mode, evaluate
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from datetime import datetime

def backup_df(df, file_path:str):
    '''Backes up the dataframe as a csv file in the specified file.

    Keyword arguments:
    df -- the dataframe you want to backup
    file_path -- the location where you want to backup the file.
    '''

def load_backup_df(file_path:str):
    '''Loads the specified csv file into a dataframe and returns said frame.

    Keyword arguments:
    file_path -- The file you want to load.
    '''
    return df

def create_airline_replies_df(collection, airline_ids:tuple, load_backup:bool):
    '''
    '''
    if load_backup == True:
        file = input('Enter path to backup for airline replies:')
        df = pd.read_csv(file, index_col='id_str_reply')
        return df
    backup_path = input('Press enter to skip \n If you want to backup your df as a csv file, please type the path of your backup file here:')
    progress_log(12, True, False, airline_ids)
    replies = collection.find({'in_reply_to_status_id_str': {'$ne': None}})
    progress_log(2, True)
    airline_replies_list = []
    progress_log(13, True)
    for tweet in tqdm(replies):
        if tweet['user']['id_str'] in airline_ids:
            airline_replies_list.append(
                    {
                        'id_str_reply': str(tweet['id_str']),
                        'created_at_reply': tweet['created_at'],
                        'airline_id_str':str(tweet['user']['id_str']),
                        'in_reply_to_status_id_str': str(tweet['in_reply_to_status_id_str']),
                    }
                    )
    df = pd.DataFrame(airline_replies_list)
    progress_log(14, True)
    if backup_path:
        progress_log(15, True, False, backup_path)
        df.to_csv(backup_path, index=True)
        progress_log(16, True)
    return df

def create_original_messages_df(collection, airline_replies_df, load_backup:bool):
    '''
    '''
    if load_backup == True:
        file = input('Enter path to backup for original replies:')
        df = pd.read_csv(file, index_col='id_str_original')
        return df
    backup_path = input('Press enter to skip \n If you want to backup your df as a csv file, please type the path of your backup file here:')
    progress_log(3, True)
    reply_ids = set(airline_replies_df['in_reply_to_status_id_str'])
    tweets = collection.find()
    original_tweets_list = []
    progress_log(5, True)
    for tweet in tqdm(tweets):
        if tweet['id_str'] in reply_ids:
            original_tweets_list.append(
                    {
                        'id_str_original': str(tweet['id_str']),
                        'created_at_original': tweet['created_at'],
                    }
                    )
        else:
            continue
    df = pd.DataFrame(original_tweets_list)
    if backup_path:
        progress_log(15, True, False, backup_path)
        df.to_csv(backup_path, index=True)
        progress_log(16, True)
    return df

def calc_response_times(collection, airline_ids, use_backup:bool):
    '''
    '''
    if use_backup == True:
        file = input('Enter path to backup for airline replies:')
        df = pd.read_csv(file, index_col='id_str_reply')
        return df
    backup_path = input('Press enter to skip \n If you want to backup your df as a csv file, please type the path of your backup file here:')
    df_replies = create_airline_replies_df(collection, airline_ids, False)
    df_originals = create_original_messages_df(collection, df_replies, False)
    df = pd.merge(df_replies, df_originals, left_on='in_reply_to_status_id_str', right_on='id_str_original', how='inner')
    df_difference_list = []
    date_format = '%a %b %d %H:%M:%S +%f %Y'
    for ind in df.index:
        difference = (datetime.strptime(df['created_at_reply'][ind], date_format) - datetime.strptime(df['created_at_original'][ind], date_format)).total_seconds()
        df_difference_list.append(
                {'in_reply_to_status_id_str': df['in_reply_to_status_id_str'][ind],
                 'difference': difference}
                )
    df_difference = pd.DataFrame(df_difference_list)
    df = pd.merge(df, df_difference, how='left', on='in_reply_to_status_id_str')
    if backup_path:
        progress_log(15, True, False, backup_path)
        df.to_csv(backup_path, index=True)
        progress_log(16, True)
    return df

def pre_process_response_times(df, airline_ids):
    result = dict()
    bin_size = 120
    for ind in tqdm(df.index):
        if str(df['airline_id_str'][ind]) in airline_ids:
            key = int(df['difference'][ind]//bin_size)
            if key in result:
                result[key] += 1
            else:
                result[key] = 1
                print(key)
        else:
            continue
    print(result)
    return result


def plot_figures():
    if False:
        df = calc_response_times(conv_collection, set(airlines[airline]['id_str'] for airline in airlines), True)
        virgin_atlantic_rt = pre_process_response_times(df, {'20626359'})
        virgin_atlantic_df = pd.DataFrame(virgin_atlantic_rt, index=[0])
        virgin_atlantic_df.to_csv('temp/backup_va.csv')
        british_airways_rt = pre_process_response_times(df, {'18332190'})
        british_airways_df = pd.DataFrame(british_airways_rt, index=[0])
        british_airways_df.to_csv('temp/backup_ba.csv')
        all_airlines_rt = pre_process_response_times(df, {airlines[airline]['id_str'] for airline in airlines})
        all_airlines_df = pd.DataFrame(all_airlines_rt, index=[0])
        all_airlines_df.to_csv('temp/backup_all.csv')
    else:
        virgin_atlantic_df = pd.read_csv('temp/backup_va.csv').transpose().drop(index='Unnamed: 0')
        british_airways_df = pd.read_csv('temp/backup_ba.csv').transpose().drop(index='Unnamed: 0')
        all_airlines_df = pd.read_csv('temp/backup_all.csv').transpose().drop(index='Unnamed: 0')
    virgin_atlantic_df.index = virgin_atlantic_df.index.astype('int')
    british_airways_df.index = british_airways_df.index.astype('int')
    all_airlines_df.index = all_airlines_df.index.astype('int')
    virgin_atlantic_df = virgin_atlantic_df.sort_index()
    british_airways_df = british_airways_df.sort_index()
    all_airlines_df = all_airlines_df.sort_index()
    plt.plot(virgin_atlantic_df.index.values.tolist(), virgin_atlantic_df[0]/2, label='Virgin Atlantic')
    plt.plot(british_airways_df.index.values.tolist(), british_airways_df[0]/2, label='British Airways')
    plt.plot(all_airlines_df.index.values.tolist(), all_airlines_df[0]/2, label='All Airlines')
    plt.legend()
    plt.title('Concentration of replytimes')
    ax = plt.gca()
    ax.set_ylabel('replies per minute')
    ax.set_xlabel('minutes after original post')
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 4000])
    #ax.set_yscale('log')
    plt.savefig('plot.pdf')

#plot_figures()
df = calc_response_times(conv_collection, set(airlines[airline]['id_str'] for airline in airlines), True)

print(df['difference'].mean()/60)
