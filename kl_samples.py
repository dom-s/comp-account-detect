import random
import user_lm
import time
import comp_dataset
import logging
import json
from multiprocessing.pool import Pool
from contextlib import contextmanager
from tools import get_user_info
from yaml import load

# set encoding to utf8, globally
import sys
reload(sys)
sys.setdefaultencoding('utf8')


# enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

def create_kl_samples(tweets, sample_size, is_comp_dataset):

    number_tweets = len(tweets)

    if number_tweets < 2:  # at least 2 days of tweets (required by sampling alogirhtm)
        return

    A = []

    sample_indexes = []

    while len(sample_indexes) < sample_size:
        begin = random.randint(0, number_tweets-2)
        end = random.randint(begin+1, number_tweets-1)
        sample_indexes.append((begin, end))

    for i, sample_index in enumerate(sample_indexes):
        begin = sample_index[0]
        end = sample_index[1]

        user_tweets = tweets[:begin]
        comp_tweets = tweets[begin:end]
        user_tweets.extend(tweets[end:number_tweets])

        user_langmodel = user_lm.build_user_lm(user_tweets, comp_dataset=is_comp_dataset)
        comp_langmodel = user_lm.build_user_lm(comp_tweets, comp_dataset=is_comp_dataset)

        kl_div = user_lm.calculate_KL_divergence(user_langmodel, comp_langmodel)[0]
        A.append(kl_div)

        # del user_tweets

    return A


def sample_kl_for_user(user_tweets):
    t_start = time.time()

    user_id, comp_id, comp_begin, comp_end, comp_begin_day, comp_end_day = get_user_info(user_tweets)

    user_tweets = comp_dataset.group_tweets_by_day(user_tweets)


    logging.debug('processing userId: {}'.format(user_id))

    kl_samples = create_kl_samples(user_tweets, sample_size, is_comp_dataset=True)

    out = {'user_id': user_id, 'comp_id': comp_id, 'comp_begin': comp_begin, 'comp_end': comp_end,
                        'comp_begin_day': comp_begin_day, 'comp_end_day': comp_end_day,
                        'samples': kl_samples}

    t_end = time.time()
    logging.debug('user {} finished, {}'.format(user_id, t_end-t_start))

    return out


@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()


if __name__ == '__main__':
    config = load(open('config.yaml'))

    percs = config['percs_compromised']

    for perc in percs:
        dataset = config['synth_dataset'].format(perc)
        logging.info('processing dataset: {}...'.format(dataset))

        sample_size = config['sample_size']

        user_tweets = comp_dataset.read_users(dataset)

        logging.info('starting pool...')

        with poolcontext(processes=config['processes']) as pool:
            results = pool.map(sample_kl_for_user, user_tweets)


        with open(config['samples_file'].format(perc), 'w') as fout:
            for line in results:
                fout.write('{}\n'.format(json.dumps(line)))
