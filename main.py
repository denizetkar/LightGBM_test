import math
import os
import pickle

import hyperopt
from hyperopt import hp, Trials
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import helper
from helper import LGBMEvaluator


def main():
    rand_seed = 11
    np.random.seed(rand_seed)
    items_data_path = os.path.join('data', 'items.txt')
    items_data = pd.read_csv(items_data_path, sep=';', header=None)
    items_data = items_data.rename(columns={0: 'itemId', 1: 'itemDesc', 2: 'item1', 3: 'item2', 4: 'item3',
                                            5: 'item4', 6: 'item5', 7: 'item6', 8: 'item7', 9: 'item8'})
    ratings_data_path = os.path.join('data', 'ratings.txt')
    ratings_data = pd.read_csv(ratings_data_path, sep=';', header=None)
    ratings_data = ratings_data.rename(columns={0: 'contextId', 1: 'itemId', 2: 'rating', 3: 'userId'})
    users_data_path = os.path.join('data', 'usersDescription.txt')
    users_data = pd.read_csv(users_data_path, sep=';', header=None)
    users_data = users_data.rename(columns={0: 'contextId', 1: 'age', 2: 'is_man', 3: 'is_woman', 50: 'userId'})
    # np.all(users_data['is_man'] ^ users_data['is_woman']) --> True
    # So either 'is_man' or 'is_woman' is enough
    users_data.drop('is_woman', axis=1, inplace=True)
    for i in range(3, 16):
        users_data = users_data.rename(columns={i + 1: 'SPC' + str(i - 2)})
    for i in range(16, 29):
        users_data = users_data.rename(columns={i + 1: 'userSpecialty' + str(i - 15)})
    for i in range(29, 39):
        users_data = users_data.rename(columns={i + 1: 'userPreference' + str(i - 28)})
    for i in range(39, 47):
        users_data = users_data.rename(columns={i + 1: 'userHighDegree' + str(i - 38)})
    for i in range(47, 49):
        users_data = users_data.rename(columns={i + 1: 'weatherSeason' + str(i - 46)})
    num_users = users_data['userId'].unique().size
    users_data['contextId'] = (users_data['contextId'] - 1) // num_users
    ratings_data['contextId'] = (ratings_data['contextId'] - 1) // num_users

    # Extract features from 'itemDesc'
    tfidf_vectorizer = TfidfVectorizer()
    item_desc_features = tfidf_vectorizer.fit_transform(items_data['itemDesc'])
    item_desc_features = pd.DataFrame(
        item_desc_features.todense(),
        columns=['itemDesc_' + str(col_name) for col_name in tfidf_vectorizer.get_feature_names()])
    items_data = pd.concat([items_data.drop('itemDesc', axis=1), item_desc_features], axis=1)

    # Merge all data tables to create training data table
    train_data = ratings_data.merge(items_data, on='itemId', how='left', suffixes=('', '_duplicate_'))
    train_data.drop([col for col in train_data.columns if col.endswith('_duplicate_')], axis=1, inplace=True)
    train_data = train_data.merge(users_data, how='left', on=['userId', 'contextId'], suffixes=('', '_duplicate_'))
    train_data.drop([col for col in train_data.columns if col.endswith('_duplicate_')], axis=1, inplace=True)
    helper.safe_del(['items_data', 'ratings_data', 'users_data', 'i', 'tfidf_vectorizer',
                     'item_desc_features'], locals())

    # get input and output tables ready
    drop_cols = ['userId']
    target_col = 'rating'
    X = train_data.drop(drop_cols + [target_col], axis=1)
    y = train_data['rating']
    del train_data
    input_cols = list(X.columns)
    input_cont_cols = [col for col in X.columns if col.startswith('itemDesc_')] + ['age']
    input_cat_cols = [col for col in input_cols if col not in input_cont_cols]

    # create a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rand_seed)
    del X, y

    prior_params = {
        "seed": rand_seed,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
        "num_threads": 12,
        # "device_type": "gpu",
        # "gpu_platform_id": 1,
        # "gpu_device_id": 0,
        "first_metric_only": True,
        "num_iterations": 2000
    }

    quantized_param_names = ['num_leaves', 'min_data_in_leaf', 'early_stopping_round',
                             'min_data_per_group', 'max_cat_threshold']
    lgbm_evaluator = LGBMEvaluator(X_train, X_test, y_train, y_test, input_cat_cols, prior_params,
                                   quantized_param_names, invert_loss=True)
    # For quantized uniform parameters: low <- (actual_low - q/2) AND high <- (actual_high + q/2)
    space = {
        "learning_rate": hp.loguniform('learning_rate', math.log(1e-4), math.log(4e-1)),
        "num_leaves": hp.quniform('num_leaves', 19.5, 100.5, 1),
        "min_data_in_leaf": hp.quniform('min_data_in_leaf', 7.5, 202.5, 5),
        "early_stopping_round": hp.qloguniform('early_stopping_round', math.log(2.5), math.log(52.5), 5),
        "lambda_l1": hp.uniform('lambda_l1', 0.0, 1.0),
        "min_data_per_group": hp.qloguniform('min_data_per_group', math.log(47.5), math.log(502.5), 5),
        "max_cat_threshold": hp.quniform('max_cat_threshold', 14, 50, 4)
    }
    trials = Trials()
    best_params = hyperopt.fmin(lgbm_evaluator, space, algo=hyperopt.tpe.suggest, max_evals=1000, trials=trials,
                                rstate=np.random.RandomState(rand_seed),
                                points_to_evaluate=[
                                    # my defaults
                                    {
                                        "learning_rate": 0.05,
                                        "num_leaves": 31,
                                        "min_data_in_leaf": 20,
                                        "early_stopping_round": 10,
                                        "lambda_l1": 0.005,
                                        "min_data_per_group": 100,
                                        "max_cat_threshold": 32
                                    },
                                    # rand_seed: 11
                                    # validation accuracy: 0.8048528652555498
                                    # validation auc: 0.8619842614803818
                                    {
                                        'early_stopping_round': 25,
                                        'lambda_l1': 0.6443540795535917,
                                        'learning_rate': 0.08445286192154607,
                                        'max_cat_threshold': 48,
                                        'min_data_in_leaf': 10,
                                        'min_data_per_group': 235,
                                        'num_leaves': 84
                                    },
                                    # rand_seed: 11
                                    # validation accuracy: 0.8069179143004647
                                    # validation auc: 0.8645277768219668
                                    {
                                        'early_stopping_round': 35,
                                        'lambda_l1': 0.43352230999559277,
                                        'learning_rate': 0.11493763952652163,
                                        'max_cat_threshold': 24,
                                        'min_data_in_leaf': 10,
                                        'min_data_per_group': 105,
                                        'num_leaves': 64
                                    },
                                    # rand_seed: 11
                                    # validation accuracy: 0.8079504388229221
                                    # validation auc: 0.8687060361099915
                                    {
                                        'early_stopping_round': 40,
                                        'lambda_l1': 0.06042506065538272,
                                        'learning_rate': 0.09047388281273572,
                                        'max_cat_threshold': 28,
                                        'min_data_in_leaf': 10,
                                        'min_data_per_group': 50,
                                        'num_leaves': 92
                                    }
                                ])
    # Save the model, parameters and loss
    model_path = os.path.join('models', 'best_model_params_loss.pickle')
    with open(model_path, 'wb') as f:
        pickle.dump((lgbm_evaluator.best_model, lgbm_evaluator.best_params, lgbm_evaluator.best_loss), f)

    pred_y_test = lgbm_evaluator.best_model.predict(X_test)
    pred_test = pred_y_test >= 0.5
    acc = accuracy_score(y_test, pred_test)
    print('accuracy is ' + str(acc))
    pass


if __name__ == '__main__':
    main()
