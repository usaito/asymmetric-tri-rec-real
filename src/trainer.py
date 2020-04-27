"""
Codes for training recommenders used in the real-world experiments
in the paper "Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback".
"""
import time
import yaml
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.python.framework import ops

from evaluate.evaluator import aoa_evaluator
from utils.preprocessor import preprocess_datasets
from models.models import MFIPS


def estimate_pscore(train: np.ndarray, train_mcar: np.ndarray,
                    val: np.ndarray, model_name: str) -> Tuple:
    """Estimate pscore."""
    if 'item' in model_name:
        pscore = np.unique(train[:, 1], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 1]]
        pscore_val = pscore[val[:, 1]]

    elif 'user' in model_name:
        pscore = np.unique(train[:, 0], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 0]]
        pscore_val = pscore[val[:, 0]]

    elif 'both' in model_name:
        user_pscore = np.unique(train[:, 0], return_counts=True)[1]
        user_pscore = user_pscore / user_pscore.max()
        item_pscore = np.unique(train[:, 1], return_counts=True)[1]
        item_pscore = item_pscore / item_pscore.max()
        pscore_train = user_pscore[train[:, 0]] * item_pscore[train[:, 1]]
        pscore_val = user_pscore[val[:, 0]] * item_pscore[val[:, 1]]

    elif 'true' in model_name:
        pscore = np.unique(train[:, 2], return_counts=True)[1] /\
            np.unique(train_mcar[:, 2], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 2] - 1]
        pscore_val = pscore[val[:, 2] - 1]

    elif 'nb' in model_name:
        pscore = np.unique(train[:, 2], return_counts=True)[1]
        pscore = pscore / pscore.max()
        pscore_train = pscore[train[:, 2] - 1]
        pscore_val = pscore[val[:, 2] - 1]

    else:  # uniform propensity
        pscore_train = np.ones(train.shape[0])
        pscore_val = np.ones(val.shape[0])

    pscore_train = np.expand_dims(pscore_train, 1)
    pscore_val = np.expand_dims(pscore_val, 1)
    return pscore_train, pscore_val


def train_mfips(sess: tf.Session, model: MFIPS, data: str,
                train: np.ndarray, val: np.ndarray, test: np.ndarray,
                max_iters: int = 500, batch_size: int = 2**9,
                model_name: str = 'mf', seed: int = 0) -> Tuple:
    """Train and evaluate the MF-IPS model."""
    train_loss_list = []
    val_loss_list = []
    test_mse_list = []
    test_mae_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # count the num of training data and estimate the propensity scores.
    num_train = train.shape[0]
    train_mcar, test = train_test_split(test, test_size=0.95, random_state=12345)
    pscore_train, pscore_val = estimate_pscore(train=train, train_mcar=train_mcar,
                                               val=val, model_name=model_name)
    labels_train = np.expand_dims(train[:, 2], 1)
    labels_val = np.expand_dims(val[:, 2], 1)
    labels_test = np.expand_dims(test[:, 2], 1)

    # start running training a recommender
    np.random.seed(12345)
    for iter_ in np.arange(max_iters):
        # mini-batch sampling
        idx = np.random.choice(np.arange(num_train), size=batch_size)
        train_batch, pscore_batch, labels_batch = train[idx], pscore_train[idx], labels_train[idx]
        # update user-item latent factors
        _, loss = sess.run([model.apply_grads, model.loss],
                           feed_dict={model.users: train_batch[:, 0], model.items: train_batch[:, 1],
                                      model.labels: labels_batch, model.scores: pscore_batch})
        train_loss_list.append(loss)
        # calculate validation loss
        val_loss = sess.run(model.weighted_mse,
                            feed_dict={model.users: val[:, 0], model.items: val[:, 1],
                                       model.labels: labels_val, model.scores: pscore_val})
        val_loss_list.append(val_loss)
        # calculate test loss
        mse_score, mae_score = sess.run([model.mse, model.mae],
                                        feed_dict={model.users: test[:, 0],
                                                   model.items: test[:, 1],
                                                   model.labels: labels_test})
        test_mse_list.append(mse_score)
        test_mae_list.append(mae_score)

    u_emb, i_emb, i_bias = sess.run([model.user_embeddings, model.item_embeddings, model.item_bias])

    sess.close()

    return (np.min(val_loss_list),
            test_mse_list[np.argmin(val_loss_list)],
            test_mae_list[np.argmin(val_loss_list)],
            u_emb, i_emb, i_bias)


def train_mfips_with_at(sess: tf.Session, model: MFIPS, mfips1: MFIPS, mfips2: MFIPS, data: str,
                        train: np.ndarray, val: np.ndarray, test: np.ndarray,
                        epsilon: float, pre_iters: int = 500, post_iters: int = 50, post_steps: int = 5,
                        batch_size: int = 2**9, model_name: str = 'naive-at', seed: int = 0) -> Tuple:
    """Train and evaluate the MF-IPS model with asymmetric tri-training."""
    train_loss_list = []
    val_loss_list = []
    test_mse_list = []
    test_mae_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # count the num of training data and estimate the propensity scores.
    num_train = train.shape[0]
    train_mcar, test = train_test_split(test, test_size=0.95, random_state=12345)
    pscore_train, pscore_val = estimate_pscore(train=train, train_mcar=train_mcar,
                                               val=val, model_name=model_name)
    labels_train = np.expand_dims(train[:, 2], 1)
    labels_val = np.expand_dims(val[:, 2], 1)
    labels_test = np.expand_dims(test[:, 2], 1)
    pscore_model = np.ones((batch_size, 1))

    # start training a recommender
    np.random.seed(12345)
    # start pre-training step
    for i in np.arange(pre_iters):
        # mini-batch sampling
        idx = np.random.choice(np.arange(num_train), size=batch_size)
        idx1 = np.random.choice(np.arange(num_train), size=batch_size)
        idx2 = np.random.choice(np.arange(num_train), size=batch_size)
        train_batch, train_batch1, train_batch2 = train[idx], train[idx1], train[idx2]
        labels_batch, labels_batch1, labels_batch2 = labels_train[idx], labels_train[idx1], labels_train[idx2]
        pscore_batch1, pscore_batch2 = pscore_train[idx1], pscore_train[idx2]
        # update user-item latent factors
        _, train_loss = sess.run([model.apply_grads, model.loss],
                                 feed_dict={model.users: train_batch[:, 0], model.items: train_batch[:, 1],
                                            model.labels: labels_batch, model.scores: pscore_model})
        _ = sess.run(mfips1.apply_grads,
                     feed_dict={mfips1.users: train_batch1[:, 0], mfips1.items: train_batch1[:, 1],
                                mfips1.labels: labels_batch1, mfips1.scores: pscore_batch1})
        _ = sess.run(mfips2.apply_grads,
                     feed_dict={mfips2.users: train_batch2[:, 0], mfips2.items: train_batch2[:, 1],
                                mfips2.labels: labels_batch2, mfips2.scores: pscore_batch2})
    # start psuedo-labeling and final prediction steps
    all_data = pd.DataFrame(np.zeros((train[:, 0].max() + 1, train[:, 1].max() + 1)))
    all_data = all_data.stack().reset_index().values[:, :2]
    for k in np.arange(post_iters):
        for j in np.arange(post_steps):
            idx = np.random.choice(np.arange(all_data.shape[0]), size=num_train * 5)
            batch_data = all_data[idx]
            # create psuedo-labeled dataset (i.e., \tilde{\mathcal{D}})
            preds1, preds2 = sess.run([mfips1.preds, mfips2.preds],
                                      feed_dict={mfips1.users: batch_data[:, 0],
                                                 mfips1.items: batch_data[:, 1],
                                                 mfips2.users: batch_data[:, 0],
                                                 mfips2.items: batch_data[:, 1]})
            idx = np.array(np.abs(preds1 - preds2) <= epsilon).flatten()
            target_users, target_items, pseudo_labels = batch_data[idx, 0], batch_data[idx, 1], preds1[idx]
            target_data = np.c_[target_users, target_items, pseudo_labels]
            # store information during the pseudo-labeleing step
            num_target = target_data.shape[0]
            # mini-batch sampling for the pseudo-labeleing step
            idx = np.random.choice(np.arange(num_target), size=batch_size)
            idx1 = np.random.choice(np.arange(num_target), size=batch_size)
            idx2 = np.random.choice(np.arange(num_target), size=batch_size)
            train_batch, train_batch1, train_batch2 = target_data[idx], target_data[idx1], target_data[idx2]
            # update user-item latent factors of the final prediction model
            _, train_loss = sess.run([model.apply_grads, model.mse],
                                     feed_dict={model.users: train_batch[:, 0],
                                                model.items: train_batch[:, 1],
                                                model.labels: np.expand_dims(train_batch[:, 2], 1),
                                                model.scores: np.ones((np.int(batch_size), 1))})
            # calculate validation loss during the psuedo-labeleing step
            val_loss = sess.run(model.weighted_mse,
                                feed_dict={model.users: val[:, 0],
                                           model.items: val[:, 1],
                                           model.scores: pscore_val,
                                           model.labels: labels_val})
            # calculate test losses during the psuedo-labeleing step
            mse_score, mae_score = sess.run([model.mse, model.mae],
                                            feed_dict={model.users: test[:, 0], model.items: test[:, 1],
                                                       model.labels: labels_test})
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            test_mse_list.append(mse_score)
            test_mae_list.append(mae_score)
            # re-update the model parameters of pre-trained models using pseudo-labeled data
            _ = sess.run(mfips1.apply_grads,
                         feed_dict={mfips1.users: train_batch1[:, 0],
                                    mfips1.items: train_batch1[:, 1],
                                    mfips1.labels: np.expand_dims(train_batch1[:, 2], 1),
                                    mfips1.scores: np.ones((batch_size, 1))})
            _ = sess.run(mfips2.apply_grads,
                         feed_dict={mfips2.users: train_batch2[:, 0],
                                    mfips2.items: train_batch2[:, 1],
                                    mfips2.labels: np.expand_dims(train_batch2[:, 2], 1),
                                    mfips2.scores: np.ones((batch_size, 1))})

    # obtain user-item embeddings
    u_emb, i_emb, i_bias = sess.run([model.user_embeddings, model.item_embeddings, model.item_bias])

    sess.close()

    return (np.min(val_loss_list),
            test_mse_list[np.argmin(val_loss_list)],
            test_mae_list[np.argmin(val_loss_list)],
            u_emb, i_emb, i_bias)


class Objective:

    def __init__(self, data: str, model_name: str = 'mf') -> None:
        """Initialize Class"""
        self.data = data
        self.model_name = model_name

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""
        train, val, test, num_users, num_items =\
            preprocess_datasets(data=self.data, seed=12345)

        # sample a set of hyperparameters.
        config = yaml.safe_load(open('../config.yaml', 'r'))
        eta = config['eta']
        max_iters = config['max_iters']
        batch_size = config['batch_size']
        pre_iters = config['pre_iters']
        post_iters = config['post_iters']
        post_steps = config['post_steps']
        dim = trial.suggest_discrete_uniform('dim', 5, 50, 5)
        lam = trial.suggest_loguniform('lam', 1e-6, 1)
        if '-at' in self.model_name:
            epsilon = trial.suggest_loguniform('epsilon', 1e-3, 1)

        ops.reset_default_graph()
        tf.set_random_seed(12345)
        sess = tf.Session()
        if '-at' not in self.model_name:
            model = MFIPS(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta)
            score, _, _, _, _, _ = train_mfips(
                sess, model=model, data=self.data, train=train, val=val, test=test,
                max_iters=max_iters, batch_size=batch_size, model_name=self.model_name)
        else:
            model = MFIPS(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta, num=0)
            model1 = MFIPS(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta, num=1)
            model2 = MFIPS(num_users=num_users, num_items=num_items, dim=dim, lam=lam, eta=eta, num=2)
            score, _, _, _, _, _ = train_mfips_with_at(
                sess, model=model, mfips1=model1, mfips2=model2, data=self.data,
                train=train, val=val, test=test, epsilon=epsilon,
                pre_iters=pre_iters, post_iters=post_iters, post_steps=post_steps,
                batch_size=batch_size, model_name=self.model_name)

        return score


class Tuner:
    """Class for tuning hyperparameter of MF models."""

    def __init__(self, data: str, model_name: str) -> None:
        """Initialize Class."""
        self.data = data
        self.model_name = model_name

    def tune(self, n_trials: int = 30) -> None:
        """Hyperparameter Tuning by TPE."""
        path = Path(f'../logs/{self.data}')
        path.mkdir(exist_ok=True, parents=True)
        path_model = Path(f'../logs/{self.data}/{self.model_name}')
        path_model.mkdir(exist_ok=True, parents=True)
        # tune hyperparameters by Optuna
        objective = Objective(data=self.data, model_name=self.model_name)
        study = optuna.create_study(sampler=TPESampler(seed=123))
        study.optimize(objective, n_trials=n_trials)
        # save tuning results
        study.trials_dataframe().to_csv(str(path_model / f'tuning_results.csv'))
        if Path('../hyper_params.yaml').exists():
            pass
        else:
            yaml.dump(dict(yahoo=dict(), coat=dict()),
                      open('../hyper_params.yaml', 'w'), default_flow_style=False)
        time.sleep(np.random.rand())
        hyper_params_dict = yaml.safe_load(open('../hyper_params.yaml', 'r'))
        hyper_params_dict[self.data][self.model_name] = study.best_params
        yaml.dump(hyper_params_dict, open('../hyper_params.yaml', 'w'), default_flow_style=False)


class Trainer:

    def __init__(self, data: str, batch_size: int = 10, eta: float = 0.01,
                 max_iters: int = 500, pre_iters: int = 150, post_iters: int = 10,
                 post_steps: int = 10, model_name: str = 'mf') -> None:
        """Initialize class."""
        self.data = data
        hyper_params = yaml.safe_load(open(f'../hyper_params.yaml', 'r'))[data][model_name]
        self.dim = hyper_params['dim']
        self.lam = hyper_params['lam']
        if '-at' in model_name:
            self.epsilon = hyper_params['epsilon']
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.pre_iters = pre_iters
        self.post_iters = post_iters
        self.post_steps = post_steps
        self.eta = eta
        self.model_name = model_name

    def run_simulations(self, num_sims: int = 5) -> None:
        """Train mf."""
        results_mse = []
        results_mae = []
        results_ndcg = []

        # start running simulations
        start = time.time()
        for seed in np.arange(num_sims):
            train, val, test, num_users, num_items =\
                preprocess_datasets(data=self.data, seed=seed)

            ops.reset_default_graph()
            tf.set_random_seed(seed)
            sess = tf.Session()
            if '-at' not in self.model_name:
                model = MFIPS(num_users=num_users, num_items=num_items,
                              dim=self.dim, lam=self.lam, eta=self.eta)
                _, mse, mae, u_emb, i_emb, i_bias = train_mfips(
                    sess, model=model, data=self.data, train=train, val=val, test=test,
                    max_iters=self.max_iters, batch_size=self.batch_size,
                    model_name=self.model_name, seed=seed)
            else:
                model = MFIPS(num_users=num_users, num_items=num_items,
                              dim=self.dim, lam=self.lam, eta=self.eta, num=0)
                model1 = MFIPS(num_users=num_users, num_items=num_items,
                               dim=self.dim, lam=self.lam, eta=self.eta, num=1)
                model2 = MFIPS(num_users=num_users, num_items=num_items,
                               dim=self.dim, lam=self.lam, eta=self.eta, num=2)
                _, mse, mae, u_emb, i_emb, i_bias = train_mfips_with_at(
                    sess, model=model, mfips1=model1, mfips2=model2, data=self.data,
                    train=train, val=val, test=test, epsilon=self.epsilon,
                    pre_iters=self.pre_iters, post_iters=self.post_iters, post_steps=self.post_steps,
                    batch_size=self.batch_size, model_name=self.model_name, seed=seed)
            results_mae.append(mae)
            results_mse.append(mse)
            ndcg = aoa_evaluator(user_embed=u_emb, item_embed=i_emb, item_bias=i_bias, test=test)
            results_ndcg.append(ndcg)
            print(f'#{seed+1} {self.model_name}: {np.round((time.time() - start) / 60, 2)} min')
        # aggregate and save the final results
        result_path = Path(f'../logs/{self.data}/{self.model_name}')
        result_path.mkdir(parents=True, exist_ok=True)
        pd.concat([pd.DataFrame(results_mae, columns=['MAE']),
                   pd.DataFrame(results_mse, columns=['MSE']),
                   pd.DataFrame(results_ndcg, columns=['nDCG@3'])], 1)\
            .to_csv(str(result_path / 'results.csv'))
