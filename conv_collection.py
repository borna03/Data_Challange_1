from data_load import *
import pymongo
import time
from tqdm import tqdm


# ===========================================
# !!IMPORTANT !!!
# During the last time I ran it I had to run the program twice because it failed
# to add the root tweets initially.
# If you also encounter this problem run it again following these steps:
#    1) chose N to enter custom setup
#    2) chose default everywhere (by pressing enter)
#    3) EXCEPT for skip replies, enter Y here.
#    4) if you make a mistake interrupt the program (ctrl + c) for terminal
#    5) if you did it correctly it should add an additional 
#
# If you did not encounter this problem please delete this section
# because it is no longer relevant.
# ============================================

def progress_log(message_id: int, verbose: bool, force: bool = False, message: str = ''):
    """Prints the requested message in the terminal if verbose or force argument is True

    Keyword arguments:
    message_id -- The id of the requested message, see list of options below in the code
    verbose -- Use this argument to check whether a function is run in verbose mode
    force -- Use this argument to force the message, regardless of whether you are running
             in verbose mode.
    message -- Optional input that can be used to pass some numbers along, required for some messages.
    """
    messages = [(1, 'Querying all tweets that are a reply...'),
                (2, 'Query completed successfully'),
                (3, 'Generating set of reply_IDs...'),
                (4, f'Set generated successfully, found {message} reply_IDs'),
                (5, 'Iterating over remaining tweets and comparing to the set...'),
                (6, 'Adding query results to the database'),
                (7, f'{message} results were successfully added to the new collection'),
                (8, f'Done, {message} tweets have been added to the new database.'),
                (9, 'Initialising...'),
                (10, 'This step takes approximately 30 minutes and 2 million iterations.'),
                (11, 'This step takes approximately 15 minutes and 5 million iterations.')]
    if verbose == True or force == True:
        for message in messages:
            if message[0] == message_id:
                print(message[1])
                break
        else:
            print(f'Minor error: unknown message ID: \"{message_id}\"')


def debug_mode(option: int, debug: bool, extra_argument: any):
    """This function contains some tools that can be useful when debugging

    Keyword arguments:
    debug -- Setting this to true will activate the tool
    option -- This determines which functionality is used
    extra_argument -- Different information is required for different tools.
    """
    if debug:
        match option:
            case 1:
                '''Print the first n items of a iterable object
                extra_argument -- (<number of items>, <object>)'''
                n = 0
                print('showing first n items of object:')
                for item in extra_argument[1]:
                    print(item)
                    n += 1
                    if n == extra_argument[0]:
                        break
            case 2:
                '''Print the length of an iterable object
                extra_argument -- iterable
                '''
                print(f'length of iterable: {len(extra_argument)}')
            case 100:
                '''Print the message if debug mode is enabled
                extra_argument -- str of message to be displayed'''
                print(extra_argument)
            case _:
                print(f'Minor error: unknown debug option: {option}')
    else:
        return


def evaluate(yn_var, default):
    if default == True:
        if yn_var.lower()[0] == 'n':
            return False
        else:
            return True
    else:
        if yn_var.lower()[0] == 'y':
            return True
        else:
            return False


def create_new_collection(client, db, o_collection, n_collection, only_replies: bool,
                          verbose: bool, debug: bool, skip_replies: bool):
    ''' Creates a new collection in the MongoDB which contains replies and their root tweets.
    
    Keyword arguments:
    client -- The client (note, this is not just a string)
    db -- The database (note, this is not just a string)
    o_collection -- The old collection (note, this is not just a string)
    n_collection -- The name of the new collection
    only_replies -- If this is set to True root tweets of conversations will be ignored
                    this speeds up the proces.
    verbose -- If set to False you will not receive any indication of progress in the terminal.
    debug -- Setting this to True will enable debug mode and will show more inbetween steps.
    '''

    progress_log(9, verbose)  # initializing
    n_collection = db[n_collection]
    total_count = 0

    # Query all tweets that are a reply
    progress_log(1, verbose)
    replies = o_collection.find({'in_reply_to_status_id_str': {'$ne': None}})
    progress_log(2, verbose)  # Query completed
    debug_mode(1, debug, (2, replies))  # print the first two replies in debug mode

    # Add replies to new collection
    if not skip_replies:
        progress_log(6, verbose)
        progress_log(10, verbose)
        for tweet in tqdm(replies, disable=not verbose):
            try:
                n_collection.insert_one(tweet)
                total_count += 1
            except pymongo.errors.DuplicateKeyError:
                continue
        progress_log(7, verbose, False, total_count)

    # Optional break point in the program, mainly for testing purposes
    if only_replies == True:
        progress_log(8, verbose, True, total_count)
        return

    # Find root tweets (start of conversation)
    else:
        # Generate a list with IDs of root tweets using the reply IDs.
        progress_log(3, verbose)
        replies = o_collection.find({'in_reply_to_status_id_str': {'$ne': None}})
        reply_id_set = set()
        id_set = set()
        old_count = total_count
        for doc in tqdm(replies, disable=not verbose):
            reply_id_set.add(doc['in_reply_to_status_id_str'])
            id_set.add(doc['id_str'])
        root_id_set = reply_id_set.difference(id_set)
        debug_mode(1, debug, (10, root_id_set))
        debug_mode(2, debug, root_id_set)
        progress_log(4, verbose, False, len(root_id_set))

        # Query (valid) tweets that do not have a reply ID
        progress_log(5, verbose)
        progress_log(11, verbose)
        root_tweets = o_collection.find({'$and': [
            {'in_reply_to_status_id_str': None},
            {'id_str': {'$ne': None}}]})

        # Compare query results to the earlier generated set to retrieve the root tweets
        for tweet in tqdm(root_tweets, disable=not verbose):
            if tweet['id_str'] in root_id_set:
                try:
                    n_collection.insert_one(tweet)
                    total_count += 1
                except pymongo.errors.DuplicateKeyError:
                    continue
            else:
                continue
        progress_log(7, verbose, False, total_count - old_count)
    if total_count != 2290760:
        print(
            '\033[31mred ERROR: The program should have added 2290760 files to the collection, please look into this: If you open the python file there is a remark about this error at the top. \033[0m')
    progress_log(8, verbose, True, total_count)


# Setting all parameters
def set_param_and_run(client, db, collection):
    print(
        'You are about to generation a new collection in your mongoDB\nwhich includes all tweets that are part of a conversation.\nThe default settings assume you have the original DBL_data.data collection.')
    presets = input('Generate using presets Y/n: ') or 'y'

    # Set default presets
    if evaluate(presets, True):
        print('loading presets')
        client = client
        db = db
        o_collection = collection
        n_collection = 'conversations'
        only_replies = False
        verbose = True
        debug = False
        skip_replies = False

    # Ask user to set parameters.
    else:
        print('Custom generation setup, press enter to skip step')
        client = client
        db = input('database name:') or db
        o_collection = input('original collection name:') or collection
        n_collection = input('new collection name:') or 'conversations'
        only_replies = input('only query replies (faster) y/N: ') or 'n'
        only_replies = evaluate(only_replies, False)
        verbose = input('operate in verbose mode Y/n: ') or 'y'
        verbose = evaluate(verbose, True)
        debug = input('operate in debug mode y/N: ') or 'n'
        debug = evaluate(debug, False)
        skip_replies = input('skip writing the replies to the new collection, only write root tweets y/N: ') or 'n'
        skip_replies = evaluate(skip_replies, False)
        print(
            f'running with database = {db}, original collection = {o_collection}, new collection = {n_collection}, only replies = {only_replies}, verbose mode = {verbose}, debug mode = {debug}, skip replies = {skip_replies}.')
        print('\033[93m If this is not correct quit the program as soon as possible!!! \033[0m')
        time.sleep(10)
    create_new_collection(client, db, o_collection, n_collection, only_replies, verbose, debug, skip_replies)


# Run the program and create the new collection
set_param_and_run(client, db, collection)
