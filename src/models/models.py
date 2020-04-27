"""
Recommender models used for the real-world experiments
in the paper "Asymmetric Tri-training for Debiasing Missing-Not-At-Random Rating Feedback".
"""
from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf


class AbstractRecommender(metaclass=ABCMeta):
    """Abstract base class for evaluator class."""

    @abstractmethod
    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        raise NotImplementedError()

    @abstractmethod
    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        raise NotImplementedError()

    @abstractmethod
    def create_losses(self) -> None:
        """Create the losses."""
        raise NotImplementedError()

    @abstractmethod
    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        raise NotImplementedError()


class MFIPS(AbstractRecommender):
    """Matrix Factorization w/o Inverse Propensity Score."""

    def __init__(self, num_users: np.array, num_items: np.array,
                 dim: int, lam: float, eta: float, num: int = 1) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.eta = eta
        self.num = num

        # Build the graphs
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_ph')
        self.items = tf.placeholder(tf.int32, [None], name='item_ph')
        self.scores = tf.placeholder(tf.float32, [None, 1], name='score_ph')
        self.labels = tf.placeholder(tf.float32, [None, 1], name='label_ph')

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            self.user_embeddings = tf.get_variable(f'user_embeddings{self.num}', shape=[self.num_users, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.item_embeddings = tf.get_variable(f'item_embeddings{self.num}', shape=[self.num_items, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.user_bias = tf.Variable(tf.random_normal(shape=[self.num_users], stddev=0.01),
                                         name=f'user_bias{self.num}')
            self.item_bias = tf.Variable(tf.random_normal(shape=[self.num_items], stddev=0.01),
                                         name=f'item_bias{self.num}')
            self.global_bias = tf.get_variable(f'global_bias{self.num}', [1],
                                               initializer=tf.constant_initializer(1e-3, dtype=tf.float32))

            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            self.i_embed = tf.nn.embedding_lookup(self.item_embeddings, self.items)
            self.u_bias = tf.nn.embedding_lookup(self.user_bias, self.users)
            self.i_bias = tf.nn.embedding_lookup(self.item_bias, self.items)

        with tf.variable_scope('prediction'):
            self.preds = tf.reduce_sum(tf.multiply(self.u_embed, self.i_embed), 1)
            self.preds = tf.add(self.preds, self.u_bias)
            self.preds = tf.add(self.preds, self.i_bias)
            self.preds = tf.add(self.preds, self.global_bias)
            self.preds = tf.expand_dims(self.preds, 1)

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            local_losses = tf.square(self.labels - self.preds)
            self.weighted_mse = tf.reduce_sum(local_losses / self.scores) / tf.reduce_sum(1. / self.scores)
            self.mse = tf.reduce_mean(tf.square(self.labels - self.preds))
            self.mae = tf.reduce_mean(tf.abs(self.labels - self.preds))

            reg_embeds = tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings)
            reg_biases = tf.nn.l2_loss(self.user_bias) + tf.nn.l2_loss(self.item_bias)
            self.loss = self.weighted_mse + self.lam * (reg_embeds + reg_biases)

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            self.apply_grads = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.loss)
