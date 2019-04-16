import logging
import numpy as np
import sklearn.metrics
import json
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from yaml import load

# enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def _generate_feature_matrix(kl_samples_path, feature_matrix_path, sample_sizes):
    with open(kl_samples_path) as fin:
        users = [json.loads(line) for line in fin]
    feature_matrix = np.empty(shape=(len(users), len(sample_sizes) * num_features + 2))
    for k, user in enumerate(users):
        '''
        This needs to be done to align the kl-features file
        with the word-count and doc2vec feature files
        '''
        if k == len(users) - 1:
            k = 0
        else:
            k = k+1
        if user['user_id'] is None:
            feature_matrix[k][0] = -1
        else:
            feature_matrix[k][0] = int(user['user_id']) # user_id
        label = 1
        if user['comp_id'] is None or user['comp_id'] == 'None':
            label = 0
        feature_matrix[k][1] = label
        i = 0
        for sample_size in sample_sizes:
            for j, feature in enumerate(get_features(user['samples'], sample_size)):
                feature_matrix[k][i+j+2] = feature
            i += num_features

    np.save(feature_matrix_path, feature_matrix)


def get_features(A, sample_size):
    features = list()
    if A is None:
        return [0] * num_features

    choice = np.random.choice(A, sample_size)

    # average
    features.append(np.mean(choice))

    # max
    features.append(max(choice))

    # min
    features.append(min(choice))

    # var
    features.append(np.var(choice))

    return features


def evaluation(y_test, y_pred):
    print

    cm = confusion_matrix(y_test, y_pred)
    print cm

    acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    print 'accuracy:', acc

    p = sklearn.metrics.precision_score(y_test, y_pred)
    print 'precision:', p

    r = sklearn.metrics.recall_score(y_test, y_pred)
    print 'recall:', r

    f1 = sklearn.metrics.f1_score(y_test, y_pred)
    print 'f1:', f1


def get_xy(feature_matrix_path):
    feature_matrix = np.load(feature_matrix_path)
    X = feature_matrix[:, 2:]
    y = feature_matrix[:, 1]
    return X, y


def classify_svm_cross_validation(X, y, folds):
    scaler = StandardScaler()
    classifier = LinearSVC(dual=False, fit_intercept=False)

    cross_validation(make_pipeline(scaler, classifier), X, y, folds, n_jobs=-1)


def cross_validation(classifer, X, y, folds, n_jobs):
    scoring = {'Accuracy': 'accuracy', 'F1': 'f1',
              'Precision': 'precision', 'Recall': 'recall'}

    scores = cross_validate(classifer, X, y, scoring=scoring, cv=folds, n_jobs=n_jobs)

    for name, score in scores.items():
        logging.info("%s: %0.2f (+- %0.2f)" % (name, score.mean(), score.std() * 2))


def generate_feature_matrix(perc_comps, samples_file, feature_matrix, sample_sizes):
    for perc_comp in perc_comps:
        _generate_feature_matrix(samples_file.format(perc_comp),
                                 feature_matrix.format(perc_comp),
                                 sample_sizes)


def clf_training(folds, perc_comps, feature_matrix):
    for perc_comp in perc_comps:
        X, y = get_xy(feature_matrix.format(perc_comp))

        print
        logging.info('SVM with {} fold cross validation with {} % of tweets compromised'.format(folds, perc_comp*100))
        classify_svm_cross_validation(X, y, folds)


def clf_ablation(folds, perc_comps, feature_matrix, sample_sizes):
    num_samples = len(sample_sizes)
    indx_dict = ['avg', 'max', 'min', 'var']
    for perc_comp in perc_comps:
        X, y = get_xy(feature_matrix.format(perc_comp))
        for i, feature in enumerate(indx_dict):
            indxs = [i+j*4 for j in range(num_samples)]
            logging.info(
                    'SVM with {} fold cross validation with {} % comp, feature:{}'.format(folds, perc_comp*100, feature))
            classify_svm_cross_validation(X[:,indxs], y, folds)


if __name__ == '__main__':
    config = load(open('config.yaml'))
    perc_comps = config['percs_compromised']
    folds = config['cv_folds']
    feature_matrix = config['feature_matrix']
    samples_file = config['samples_file']
    sample_sizes = config['sub_sample_sizes']
    num_features = 4

    logging.info('Generating feature matrix...')
    generate_feature_matrix(perc_comps, samples_file, feature_matrix, sample_sizes)

    logging.info('Training classifier...')
    clf_training(folds, perc_comps, feature_matrix)

    logging.info('Performing feature ablation...')
    clf_ablation(folds, perc_comps, feature_matrix, sample_sizes)
