from datetime import datetime
from gensim.utils import smart_open

def parse_datetime(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')


def group_tweets_by_day(tweets):
    tweets_by_day = []
    last_date = datetime(1970, 1, 1)
    for tweet in tweets:
        line = tweet.strip().split('\t')
        user = line[0]
        comp_user = line[1]
        timestamp = parse_datetime(line[2])
        text = line[3]

        if timestamp.date() != last_date.date():
            tweets_by_day.append('{}\t{}\t{}\t{}'.format(user, comp_user, timestamp.date(), text))
            last_date = timestamp
        else:
            last_line = tweets_by_day.pop()
            concatenated_text = last_line.strip().split('\t')[3]
            tweets_by_day.append('{}\t{}\t{}\t{}'.format(user, comp_user, timestamp.date(),
                                                         concatenated_text + ' ' + text))

    return tweets_by_day


def read_users(tweets_in):
    last_user = None
    tweets_raw = []

    with smart_open(tweets_in) as fin:
        for line in fin:
            user = line.strip().split('\t')[0]
            if last_user is None:
                last_user = user
            if user == last_user:
                tweets_raw.append(line)
            else:
                yield tweets_raw
                last_user = user
                tweets_raw = []

    yield tweets_raw
