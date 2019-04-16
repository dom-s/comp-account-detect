import re
import nltk
import numpy as np
from nltk.probability import FreqDist, LaplaceProbDist
from scipy.stats import entropy

tokenizer = nltk.tokenize.WordPunctTokenizer()

SIMPLE_URL_REGEX = re.compile(r'(https?://\S+)', re.IGNORECASE)


def build_user_lm(tweets, preserve_case=False, remove_links=False, remove_retweets=False, comp_dataset=False):
    text = []

    for line in tweets:
        line = line.strip().split('\t')
        if not comp_dataset:
            tweet = line[2]
        else:
            tweet = line[3]
        if remove_retweets and tweet.startswith('RT'):
            continue
        text.extend(tokenize_tweet(tweet, preserve_case, remove_links))

    freq = FreqDist(text)
    return freq


def tokenize_tweet(tweet, preserve_case=False, remove_links=False):
    if remove_links:
        tweet = SIMPLE_URL_REGEX.sub('', tweet)

    text = tokenizer.tokenize(tweet)

    if not preserve_case:
        text = [tok.lower() for tok in text]

    return text


def calculate_KL_divergence(lm1, lm2):
    vocabulary = lm1.keys()
    vocabulary.extend(lm2.keys())

    prob_dist_lm1 = np.empty(shape=(len(vocabulary), 1))
    prob_dist_lm2 = np.empty(shape=(len(vocabulary), 1))

    smooth_lm1 = LaplaceProbDist(lm1)
    smooth_lm2 = LaplaceProbDist(lm2)

    for i, word in enumerate(vocabulary):
        prob_dist_lm1[i] = smooth_lm1.prob(word)
        prob_dist_lm2[i] = smooth_lm2.prob(word)
        # print word, smooth_lm1.prob(word), smooth_lm1.prob(word)

    return entropy(prob_dist_lm1, prob_dist_lm2)
