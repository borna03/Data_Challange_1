from data_load import *
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
from tqdm import tqdm
import time


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def sentiment_analysis_to_airline(id_str):
    """
    Sentiment analysis for a certain airline
    Using roBERTa model for sentiment analysis (see https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment).
    For all tweets for which the airline is mentioned (@) or is replied to.
    Not including re-tweets
    :param id_str: id of airline
    """
    # Setup sentiment analysis model
    print('Loading Model')
    MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)
    print('Loading Model Successful')

    # Setup counters & dictionaries
    expected_count = 0
    total_count = 0
    pos_sent = 0
    neu_sent = 0
    neg_sent = 0
    unc_sent = 0
    sentiment_counter = dict()

    start_time = time.time()

    # Find all tweets for which the airline is mentioned (@) or is replied to:
    query = {'$or': [{'in_reply_to_user_id_str': id_str}, {'entities.user_mentions': {'$elemMatch': {'id_str': id_str}}}]}

    print(f'Expected amount of iterations/tweets: {collection.count_documents(query)}')

    all_tweets = collection.find(query)

    print('Starting analysis:')
    # Iterate over all tweets, with live counter
    for tweet in tqdm(all_tweets):
        total_count += 1
        text = tweet['text']
        processed_text = preprocess(text)

        # Sentiment analysis model
        encoded_input = tokenizer(processed_text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        score_pos = scores[2]
        score_neu = scores[1]
        score_neg = scores[0]

        # Cutoff for probabilities under 50%
        if (score_pos > score_neu) and (score_pos > score_neg) and (score_pos > 0.5):
            label = 'positive'
            pos_sent += 1
        elif (score_neu > score_pos) and (score_neu > score_neg) and (score_neu > 0.5):
            label = 'neutral'
            neu_sent += 1
        elif (score_neg > score_pos) and (score_neg > score_neu) and (score_neg > 0.5):
            label = 'negative'
            neg_sent += 1
        else:
            label = 'uncertain'
            unc_sent += 1

        # Counter to put sentiments in a dictionary
        if label in sentiment_counter:
            sentiment_counter[label] += 1
        else:
            sentiment_counter[label] = 1

    end_time = time.time()

    usable_lines = (total_count - unc_sent) / total_count
    ratio_pos = pos_sent / total_count
    ratio_neu = neu_sent / total_count
    ratio_neg = neg_sent / total_count
    ratio_unc = unc_sent / total_count

    print(f'Total lines: {total_count}')
    print(f'Ratio non-uncertain lines: {usable_lines}')
    print('\n')

    print(sentiment_counter)
    print('\n')

    print(f'Ratio positive tweets: {ratio_pos}')
    print(f'Ratio neutral tweets: {ratio_neu}')
    print(f'Ratio negative tweets: {ratio_neg}')
    print(f'Ratio uncertain tweets: {ratio_unc}')
    print('\n')

    elapsed_time = end_time - start_time
    print("Time: ", elapsed_time)
    print("Time/Item: ", elapsed_time / total_count)

def sentiment_analysis_from_airline(id_str):
    """
    Sentiment analysis for a certain airline.
    Using roBERTa model for sentiment analysis (see https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment).
    For all tweets for which the airline is the one tweeting.
    Not including re-tweets
    :param id_str: id of airline
    """
    # Setup sentiment analysis model
    print('Loading Model')
    MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)
    print('Loading Model Successful')

    # Setup counters & dictionaries
    expected_count = 0
    total_count = 0
    pos_sent = 0
    neu_sent = 0
    neg_sent = 0
    unc_sent = 0
    sentiment_counter = dict()

    start_time = time.time()

    # Find all tweets for which the airline is the one tweeting:
    query = {'$or': [{'user.id_str': id_str}]}

    print(f'Expected amount of iterations/tweets: {collection.count_documents(query)}')

    all_tweets = collection.find(query)

    print('Starting analysis:')
    # Iterate over all tweets, with live counter
    for tweet in tqdm(all_tweets):
        total_count += 1
        text = tweet['text']
        processed_text = preprocess(text)

        # Sentiment analysis model
        encoded_input = tokenizer(processed_text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        score_pos = scores[2]
        score_neu = scores[1]
        score_neg = scores[0]

        # Cutoff for probabilities under 50%
        if (score_pos > score_neu) and (score_pos > score_neg) and (score_pos > 0.5):
            label = 'positive'
            pos_sent += 1
        elif (score_neu > score_pos) and (score_neu > score_neg) and (score_neu > 0.5):
            label = 'neutral'
            neu_sent += 1
        elif (score_neg > score_pos) and (score_neg > score_neu) and (score_neg > 0.5):
            label = 'negative'
            neg_sent += 1
        else:
            label = 'uncertain'
            unc_sent += 1

        # Counter to put sentiments in a dictionary
        if label in sentiment_counter:
            sentiment_counter[label] += 1
        else:
            sentiment_counter[label] = 1

    end_time = time.time()

    usable_lines = (total_count - unc_sent) / total_count
    ratio_pos = pos_sent / total_count
    ratio_neu = neu_sent / total_count
    ratio_neg = neg_sent / total_count
    ratio_unc = unc_sent / total_count

    print(f'Total lines: {total_count}')
    print(f'Ratio non-uncertain lines: {usable_lines}')
    print('\n')

    print(sentiment_counter)
    print('\n')

    print(f'Ratio positive tweets: {ratio_pos}')
    print(f'Ratio neutral tweets: {ratio_neu}')
    print(f'Ratio negative tweets: {ratio_neg}')
    print(f'Ratio uncertain tweets: {ratio_unc}')
    print('\n')

    elapsed_time = end_time - start_time
    print("Time: ", elapsed_time)
    print("Time/Item: ", elapsed_time / total_count)
