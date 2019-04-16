import random
import gzip
import logging
from yaml import load

# enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


def generate_compromised_dataset(twitter_dataset, dataset_out, user_lst='data/user_lst.tsv', perc_compromised=0.5, prob_compromised=0.5):
    '''
    Use a regular tweet dataset to generate artificially compromised user accounts.
    :param twitter_dataset: The original dataset
    :param dataset_out: The output of the dataset containing compromised and non-compromised accounts
    :param user_lst: Output a list of user names that are contained in the dataset
    :param perc_compromised: The percentage of tweets that are compromised within a user account
    :param prob_compromised: The probability of an account being compromised, e.g. 0.5 = roughly half of the accounts will
    be comrpomised.
    :return:
    '''
    users = []
    users.append([])

    logging.info('loading tweet data to memory...')

    with open(user_lst, 'w') as user_list:

        current_user = None

        i = 0
        index = 0
        for line in gzip.open(twitter_dataset, 'rb'):
            line = line.strip().split('\t')
            user = line[0]

            if user == current_user:
                users[i].append(line[1:])
            else:
                if current_user is not None:
                    users.append([])
                    users[i+1].append(line[1:])
                    user_list.write('{}\t{}\n'.format(i, current_user))
                    i += 1
                current_user = user

            index += 1

            if index % 1000000 == 0:
                logging.info('{}M lines processed...'.format(index / 1000000))


    logging.info('generating compromised accounts...')

    number_users = len(users)

    with gzip.open(dataset_out, 'wb') as out:

        tweets_omitted = 0

        for i in range(number_users):
            if random.uniform(0.0, 1.0) < prob_compromised:
                for tweet in users[i]:
                    if len(tweet) > 1:
                        out.write('{}\t{}\t{}\t{}\n'.format(i, None, tweet[0], tweet[1]))
                    else:
                        tweets_omitted += 1
            else:
                number_tweets = len(users[i])
                number_compromised_tweets = int(number_tweets * perc_compromised)
                j = i
                num_j = 0
                while j == i or num_j <= number_compromised_tweets:
                    j = random.randint(0, number_users - 1)
                    num_j = len(users[j])

                begin_comp = random.randint(0, max(1, number_tweets - number_compromised_tweets - 1))
                end_comp = begin_comp + number_compromised_tweets

                tweets_j = users[j]

                begin_comp_j = random.randint(0, max(1, len(tweets_j) - number_compromised_tweets - 1))

                for k, tweet in enumerate(users[i]):
                    if k <= begin_comp or k > end_comp:
                        if len(tweet) > 1:
                            out.write('{}\t{}\t{}\t{}\n'.format(i, None, tweet[0], tweet[1]))
                        else:
                            tweets_omitted += 0
                    else:
                        comp_tweet = tweets_j[begin_comp_j]
                        if len(comp_tweet) > 1:
                            out.write('{}\t{}\t{}\t{}\n'.format(i, j, comp_tweet[0], comp_tweet[1]))
                        else:
                            tweets_omitted += 1
                        begin_comp_j += 1

            if i % 1000 == 0:
                logging.info('{} users processed...'.format(i))

        logging.info('...done. {} tweets omitted'.format(tweets_omitted))


if __name__ == '__main__':
    config = load(open('config.yaml'))

    for perc_comp in config['percs_compromised']:
        f_out = config['synth_dataset'].format(perc_comp)
        generate_compromised_dataset(config['tweet_file'], f_out,
                                     perc_compromised=perc_comp, prob_compromised=config['prob_compromised'])

