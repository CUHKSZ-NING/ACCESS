import math
import time
import numpy as np
import random
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.ensemble import BalancedBaggingClassifier as BBC
from sklearn.neighbors import KNeighborsClassifier as KNN
from imblearn.under_sampling import RandomUnderSampler as CC


class SelfAdaptiveStackingEnsemble(object):  # Adaptive contribution and confidence based evolutionary semi-stacking
    def __init__(self, base_classifiers=None, n_splits=5, population_size=10, iteration=5, warm_up_rounds=2,
                 learning_rate='auto', sampling_ratio=0.3, evolve=True, verbose=True, confidence=True, mutate=True,
                 meta_learner=BBC(base_estimator=NB()), contribution=True):

        # Input parameters
        self.base_classifiers_ = deepcopy(base_classifiers)
        self.n_splits_ = n_splits
        self.population_size_ = population_size
        self.iteration_ = iteration
        self.warm_up_rounds_ = warm_up_rounds
        self.learning_rate_ = learning_rate
        self.sampling_ratio_ = sampling_ratio
        self.evolve_ = evolve
        self.verbose_ = verbose
        self.contribution_ = contribution
        self.confidence_ = confidence
        self.mutate_ = mutate

        # Key components
        self.population = {}
        self.X_train = None
        self.y_train = None
        self.imbalance_ratio = 0
        self.label_ratio = 0

        self.meta_learner_ = {'default': deepcopy(meta_learner), 'voting': deepcopy(Voting())}
        self.X_meta = {'default': None}
        self.y_meta = {'default': None}

    def fit(self, X_train, y_train):
        time_start = time.time()
        self.initialize(X_train, y_train)
        self.evolutionary_cotraining()
        self.stacking()
        time_end = time.time()

        if self.verbose_:
            print('Training time: %.1fs.\n' % (time_end - time_start))
            pass

    def initialize(self, X_train, y_train):
        self.X_train = deepcopy(X_train)
        self.y_train = deepcopy(y_train)

        self.imbalance_ratio = len(np.where(y_train == 0)[0]) / len(np.where(y_train == 1)[0])
        self.label_ratio = len(np.where(y_train != -1)[0]) / len(y_train)

        for classifier in self.base_classifiers_:
            if 'min_child_samples' in classifier.get_params():
                positive_train_approx = len(np.where(y_train == 1)[0]) * (1 - 1 / self.n_splits_)
                classifier.min_child_samples = min(20, int(positive_train_approx / 4 + 0.5))

        if self.learning_rate_ == 'auto':
            self.learning_rate_ = \
                0.02 * len(np.where(y_train == 1)[0]) * (1 / self.label_ratio - 1) ** 0.5
        else:
            self.learning_rate_ = \
                self.learning_rate_ * len(np.where(y_train == 1)[0]) * (1 / self.label_ratio - 1) ** 0.5

        key = 0
        skf = StratifiedKFold(n_splits=self.n_splits_)
        for train_i, test_i in skf.split(self.X_train, self.y_train):
            self.population[key] = {}
            positive_train = np.where(self.y_train[train_i] == 1)[0]
            negative_train = np.where(self.y_train[train_i] == 0)[0]
            unlabeled_train = np.where(self.y_train[train_i] == -1)[0]

            labeled_val = np.where(self.y_train[test_i] != -1)[0]
            unlabeled_val = np.where(self.y_train[test_i] == -1)[0]

            self.population[key]['data'] = {'X_positive_train': deepcopy(self.X_train[train_i][positive_train]),
                                            'X_negative_train': deepcopy(self.X_train[train_i][negative_train]),
                                            'X_unlabeled_train': deepcopy(self.X_train[train_i][unlabeled_train]),
                                            'Pl_train': np.zeros(len(unlabeled_train), dtype=int) - 1,
                                            'X_validation': deepcopy(self.X_train[test_i][labeled_val]),
                                            'y_validation': deepcopy(self.y_train[test_i][labeled_val]),
                                            'X_unlabeled_validation': deepcopy(self.X_train[test_i][unlabeled_val]),
                                            'Pl_validation': np.zeros(len(unlabeled_val), dtype=int) - 1,
                                            }
            self.population[key]['chromosome'] = {}
            units = self.population[key]['chromosome']
            classifier_index = 0
            for classifier in self.base_classifiers_:
                units[classifier_index] = []
                for i in range(0, self.population_size_):
                    units[classifier_index].append({'clf': deepcopy(classifier),
                                                    'samples': np.zeros(len(negative_train), dtype=int),
                                                    'fitness': 0,
                                                    'elite': False
                                                    })
                    random_i = np.array(random.sample(range(0, len(negative_train)), len(positive_train)))
                    units[classifier_index][i]['samples'][random_i] = 1
                classifier_index += 1
            self.population[key]['A_star'] = {'train': KNN(weights='distance'),
                                              'validation': KNN(weights='distance')}
            key += 1

        self.train_all()

    def evolutionary_cotraining(self):
        for i in range(0, self.iteration_):
            for j in range(0, self.warm_up_rounds_):
                if self.evolve_ and self.population_size_ >= 2:
                    self.get_fitness()
                    self.cross_over(elite=bool(self.warm_up_rounds_ == j + 1))
                    if self.mutate_:
                        self.mutate()
                self.update()
                self.train_all()

            self.co_training(progress=(i / self.iteration_))

            if not (self.evolve_ and self.population_size_ >= 2):
                self.update()
                self.train_all()

    def train_all(self):
        for key in self.population:
            X_positive = deepcopy(self.population[key]['data']['X_positive_train'])
            X_negative = deepcopy(self.population[key]['data']['X_negative_train'])
            X_unlabeled_index = np.where(self.population[key]['data']['Pl_train'] == -1)[0]
            X_unlabeled = deepcopy(self.population[key]['data']['X_unlabeled_train'][X_unlabeled_index])
            units = self.population[key]['chromosome']
            for clf in units:
                for i in range(0, len(units[clf])):
                    unit = units[clf]
                    X_train = np.concatenate((X_positive, X_negative[np.where(unit[i]['samples'] == 1)[0]]), axis=0)
                    y_train = np.concatenate((np.ones(len(X_positive), dtype=int),
                                              np.zeros(len(X_train) - len(X_positive), dtype=int)))
                    unit[i]['clf'].fit(X_train, y_train)

            X_train = combine((X_positive, X_negative, X_unlabeled))
            y_train = combine((np.ones(len(X_positive), dtype=int), np.zeros(len(X_negative), dtype=int),
                               np.zeros(len(X_unlabeled), dtype=int) - 1))

            US = CC()
            X_new, y_new = US.fit_resample(X_train, y_train)
            self.population[key]['A_star']['train'].fit(X_new, y_new)

            X_unlabeled_index = np.where(self.population[key]['data']['Pl_validation'] == -1)[0]
            X_unlabeled = deepcopy(self.population[key]['data']['X_unlabeled_validation'][X_unlabeled_index])
            X_train = combine((self.population[key]['data']['X_validation'], X_unlabeled))
            y_train = combine((self.population[key]['data']['y_validation'], np.zeros(len(X_unlabeled), dtype=int) - 1))
            US = CC()
            X_new, y_new = US.fit_resample(X_train, y_train)
            self.population[key]['A_star']['validation'].fit(X_new, y_new)

    def generate_mate_features(self, key=None, X_test=None):
        mate_features = {}

        if X_test is None:
            X_test = deepcopy(self.population[key]['data']['X_validation'])

        units = self.population[key]['chromosome']
        for clf in units:
            mate_features[clf] = None
            weight = len(units[clf])
            for i in range(0, weight):
                unit = units[clf][i]
                if mate_features[clf] is None:
                    mate_features[clf] = deepcopy(unit['clf'].predict_proba(X_test)) / weight
                else:
                    mate_features[clf] += deepcopy(unit['clf'].predict_proba(X_test)) / weight

        return mate_features

    def stacking(self):
        label_pred = {}
        for key in self.population:
            mate_features = self.generate_mate_features(key=key)
            for clf in mate_features:
                label_pred[clf] = mate_features[clf][..., 0].reshape(-1, 1)

            self.X_meta['default'] = combine((self.X_meta['default'],
                                              deepcopy(combine(label_pred, axis=1))))
            self.y_meta['default'] = combine((self.y_meta['default'],
                                              deepcopy(self.population[key]['data']['y_validation'])))

        self.meta_learner_['default'].fit(self.X_meta['default'], self.y_meta['default'])

    def predict(self, X_test):
        label_pred_proba = self.predict_proba(X_test)
        label_pred = np.zeros(len(label_pred_proba), dtype=int)
        positive_pred = np.where(label_pred_proba[..., 1] > 0.5)
        label_pred[positive_pred] = 1

        return label_pred

    def predict_proba(self, X_test, option='default'):
        X_meta = None
        label_pred = {}
        weight = len(self.population.keys())
        for key in self.population:
            mate_features = self.generate_mate_features(key=key, X_test=X_test)
            for clf in mate_features:
                label_pred[clf] = mate_features[clf][..., 0].reshape(-1, 1)

            if X_meta is None:
                X_meta = deepcopy(combine(label_pred, axis=1)) / weight
            else:
                X_meta += deepcopy(combine(label_pred, axis=1)) / weight

        return self.meta_learner_[option].predict_proba(X_meta)

    def co_training(self, progress=0.0):
        if type(self.learning_rate_) == 'str':
            self.learning_rate_ = -1
            print('Invalid learning_rate')

        for key in self.population:
            label_confidence = self.learning_rate_ * math.exp(progress) * 1 / self.n_splits_
            label_contribution = self.learning_rate_ * math.exp(progress) * (1 - 1 / self.n_splits_)

            if random.random() < label_confidence - int(label_confidence):
                label_confidence += 1
            if random.random() < label_contribution - int(label_contribution):
                label_contribution += 1

            label_confidence = int(label_confidence)
            label_contribution = int(label_contribution)
            data_temp = self.population[key]['data']

            # examples with confidence, from validation data
            X_unlabeled_validation = data_temp['X_unlabeled_validation']
            num_unlabeled = len(X_unlabeled_validation)
            if num_unlabeled > 0 and self.confidence_:
                sampling = random.sample(range(0, num_unlabeled), int(num_unlabeled * self.sampling_ratio_ + 0.5))
                mate_features = self.generate_mate_features(key=key, X_test=X_unlabeled_validation[sampling])
                scores = multiply(mate_features)
                A_star = self.population[key]['A_star']['validation']
                score_knn = A_star.predict_proba(X_unlabeled_validation[sampling])
                weight = 0.2
                score_positive = scores[..., 1] * (1 - weight + weight * (score_knn[..., 2] - score_knn[..., 1]))
                score_negative = scores[..., 0] * (1 - weight + weight * (score_knn[..., 1] - score_knn[..., 2]))
                pseudo_positive = score_positive.argsort()[:: -1][0: label_confidence]
                pseudo_negative = score_negative.argsort()[:: -1][0: label_confidence]

                for i in range(0, label_confidence):
                    if data_temp['Pl_validation'][sampling[pseudo_positive[i]]] == -1:
                        data_temp['Pl_validation'][sampling[pseudo_positive[i]]] = 1
                        data_temp['X_validation'] = np.concatenate((data_temp['X_validation'],
                                                                    [data_temp['X_unlabeled_validation'][
                                                                         sampling[pseudo_positive[i]]]]))
                        data_temp['y_validation'] = np.append(data_temp['y_validation'], [1])
                    if data_temp['Pl_validation'][sampling[pseudo_negative[i]]] == -1:
                        data_temp['Pl_validation'][sampling[pseudo_negative[i]]] = 0
                        data_temp['X_validation'] = np.concatenate((data_temp['X_validation'],
                                                                    [data_temp['X_unlabeled_validation'][
                                                                         sampling[pseudo_negative[i]]]]))
                        data_temp['y_validation'] = np.append(data_temp['y_validation'], [0])

            # examples with contribution, from training data
            X_unlabeled = self.population[key]['data']['X_unlabeled_train']
            num_unlabeled = len(X_unlabeled)
            if num_unlabeled > 0 and self.contribution_:
                sampling = random.sample(range(0, num_unlabeled), int(num_unlabeled * self.sampling_ratio_ + 0.5))
                mate_features = self.generate_mate_features(key=key, X_test=X_unlabeled[sampling])
                scores = multiply(mate_features)
                # ditto
                A_star = self.population[key]['A_star']['train']
                score_knn = A_star.predict_proba(X_unlabeled[sampling])
                weight = 0.2
                IR = self.imbalance_ratio
                score_positive = scores[..., 1] * (1 - weight + weight *
                                                   (score_knn[..., 2] + 1 / (1 + IR) * score_knn[..., 0]))
                score_negative = scores[..., 0] * (1 - weight + weight *
                                                   (score_knn[..., 1] + IR / (1 + IR) * score_knn[..., 0]))
                # original contribution
                # score_knn = A_star.predict_proba(X_unlabeled[sampling])
                # constraint = np.abs(score_knn[..., 0] - score_knn[..., 1])
                # score_positive = scores[..., 1] * (np.ones(len(score_knn)) - 0.2 * constraint)
                # score_negative = scores[..., 0] * (np.ones(len(score_knn)) - 0.2 * constraint)
                pseudo_positive = score_positive.argsort()[:: -1][0: label_contribution]
                pseudo_negative = score_negative.argsort()[:: -1][0: label_contribution]

                for i in range(0, label_contribution):
                    if data_temp['Pl_train'][sampling[pseudo_positive[i]]] == -1:
                        data_temp['Pl_train'][sampling[pseudo_positive[i]]] = 1
                        data_temp['X_positive_train'] = np.concatenate((data_temp['X_positive_train'],
                                                                        [data_temp['X_unlabeled_train'][
                                                                             sampling[pseudo_positive[i]]]]))
                    if data_temp['Pl_train'][sampling[pseudo_negative[i]]] == -1:
                        data_temp['Pl_train'][sampling[pseudo_negative[i]]] = 0
                        data_temp['X_negative_train'] = np.concatenate((data_temp['X_negative_train'],
                                                                        [data_temp['X_unlabeled_train'][
                                                                             sampling[pseudo_negative[i]]]]))

    def update(self):
        for key in self.population:
            units = self.population[key]['chromosome']
            for clf in units:
                for i in range(0, len(units[clf])):
                    samples_temp = np.zeros(len(self.population[key]['data']['X_negative_train']), dtype=int)
                    samples_index = np.where(units[clf][i]['samples'] == 1)[0]
                    samples_temp[samples_index] = 1
                    num_positive = len(self.population[key]['data']['X_positive_train'])
                    num_negative = len(samples_index)

                    if num_negative < num_positive:
                        samples_index = np.where(samples_temp == 0)[0]
                        samples_index_new = random.sample(range(0, len(samples_index)), num_positive - num_negative)
                        samples_temp[samples_index[samples_index_new]] = 1
                    if num_negative > num_positive:
                        samples_index = np.where(samples_temp == 1)[0]
                        samples_index_new = random.sample(range(0, len(samples_index)), num_negative - num_positive)
                        samples_temp[samples_index[samples_index_new]] = 0

                    units[clf][i]['samples'] = deepcopy(samples_temp)

    def get_fitness(self):
        for key in self.population:
            X_validation = deepcopy(self.population[key]['data']['X_validation'])
            y_validation = deepcopy(self.population[key]['data']['y_validation'])
            units = self.population[key]['chromosome']
            for clf in units:
                for i in range(0, len(units[clf])):
                    if units[clf][i]['elite'] is True:
                        units[clf][i]['fitness'] = 0
                        continue
                    y_pred_proba = units[clf][i]['clf'].predict_proba(X_validation)
                    units[clf][i]['fitness'] = roc_auc_score(y_validation, y_pred_proba[..., 1])
                    constrain = []
                    set_i = set(np.where(units[clf][i]['samples'] == 1)[0])
                    for j in range(0, len(units[clf])):
                        if units[clf][j]['elite'] is True:
                            set_elite = set(np.where(units[clf][j]['samples'] == 1)[0])
                            constrain.append(len(set_elite & set_i) / len(set_elite | set_i))
                    if len(constrain) > 0:
                        units[clf][i]['fitness'] *= (1 - 0.25 * np.mean(constrain))

    def cross_over(self, elite=False):
        ratio_cross = 0.3
        # ratio_diverse = 0.3
        ratio_mutate = 0.1
        # if elite is True:
        #     ratio_mutate = 0.3
        # ratio_reset = 0.5

        population_new = {}
        for key in self.population:
            population_new[key] = {}
            units = self.population[key]['chromosome']
            for clf in units:
                population_new[key][clf] = []
                units[clf].sort(key=lambda x: x['fitness'], reverse=True)
                # print('%.3f' % units[clf][0]['fitness'])

                if elite is True:
                    units[clf][0]['elite'] = True

                for i in range(0, len(units[clf])):
                    population_new[key][clf].append({'samples': None, 'clf': deepcopy(self.base_classifiers_[clf]),
                                                     'fitness': 0, 'elite': False})

        for key in population_new:
            units = self.population[key]['chromosome']
            for clf in population_new[key]:
                # copy elites
                elite_index = self.population_size_
                for i in range(0, self.population_size_):
                    if units[clf][i]['elite'] is True:
                        population_new[key][clf][elite_index - 1]['samples'] = units[clf][i]['samples']
                        population_new[key][clf][elite_index - 1]['elite'] = True
                        elite_index -= 1

                for i in range(0, elite_index):
                    pair = random.sample(range(1, int(self.population_size_ / 2) + 1), 2)
                    vector_samples = deepcopy(units[clf][pair[0]]['samples'])
                    for j in range(0, len(vector_samples)):
                        if random.random() < ratio_cross:
                            vector_samples[j] = units[clf][pair[1]]['samples'][j]
                    for j in range(0, len(vector_samples)):
                        if random.random() < ratio_mutate and vector_samples[j] == units[clf][pair[1]]['samples'][j]:
                            vector_samples[j] = 0

                    population_new[key][clf][i]['samples'] = deepcopy(vector_samples)

        for key in population_new:
            units = self.population[key]['chromosome']
            for clf in population_new[key]:
                for i in range(0, self.population_size_):
                    units[clf][i]['samples'] = deepcopy(population_new[key][clf][i]['samples'])
                    units[clf][i]['elite'] = deepcopy(population_new[key][clf][i]['elite'])

    def mutate(self, progress=0.0):
        for key in self.population:
            units = self.population[key]['chromosome']
            for clf in units:
                for i in range(0, len(units[clf])):
                    if units[clf][i]['elite'] is True:
                        continue
                    num_elites = 0
                    mean_elites = np.zeros(len(units[clf][i]['samples']))
                    for j in range(0, len(units[clf])):
                        if units[clf][j]['elite'] is True:
                            num_elites += 1
                            mean_elites += units[clf][j]['samples']
                    if num_elites > 0:
                        mean_elites /= num_elites
                        for k in range(0, len(units[clf][i]['samples'])):
                            if random.random() < 0.2 * math.exp(progress) and random.random() < mean_elites[k]:
                                units[clf][i]['samples'][k] = 0


class Voting(object):
    def __init__(self):
        self.already_predicted = False
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X_train, y_train):
        self.X_train_ = deepcopy(X_train)
        self.y_train_ = deepcopy(y_train)

    def predict_proba(self, X_test):
        label_pred_proba = np.zeros((len(X_test), 2))
        for i in range(0, len(X_test)):
            label_pred_proba[i, 0] = np.mean(X_test[i, ...])
            label_pred_proba[i, 1] = 1 - label_pred_proba[i, 0]

        self.already_predicted = True

        return label_pred_proba

    def predict(self, X_test):
        label_pred_proba = self.predict_proba(X_test)
        label_pred = np.zeros(len(label_pred_proba), dtype=int)
        positive_pred = np.where(label_pred_proba[..., 1] > 0.5)
        label_pred[positive_pred] = 1

        self.already_predicted = True

        return label_pred


# Tools
def combine(data, keys=None, axis=0):  # A safe concatenate function
    if not (type(data) is tuple or type(data) is dict):
        return data

    data_temp = []

    for i in range(0, len(data)):
        if data[i] is not None and len(data[i]) > 0:
            if keys is not None and i not in keys:
                continue
            data_temp.append(deepcopy(data[i]))

    if len(data_temp) > 0:
        return np.concatenate(tuple(data_temp), axis=axis)
    else:
        return None


def multiply(mate_features):  # A safe concatenate function
    scores = None
    weight = 1 / len(mate_features)

    for clf in mate_features:
        if scores is None:
            scores = (deepcopy(mate_features[clf])) ** weight
        else:
            scores *= (deepcopy(mate_features[clf])) ** weight

    return scores
