""" An abstract manager for initializing and switching predefined optimizers.
"""

import collections
import itertools
import typing as t
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from util.log import exec_log as log
from util.display import ReprMixin


def _init_optimizer(
        optim_type: str,
        learning_rate: float = None,
        global_step: int = None,
        decay_steps: int = None,
        decay_rate: float = None,
        **kwargs):
    if decay_steps and decay_rate:
        learning_rate = tf.train.exponential_decay(learning_rate,
                global_step, decay_steps, decay_rate)
    init_kwargs = {'name': 'optimizer',
                   'learning_rate': learning_rate}
    log.debug('Build %s%s' % (optim_type,
            '' if isinstance(learning_rate, float) else
            ' with exponential decay'))
    return getattr(tf.train, optim_type)(learning_rate=learning_rate, **kwargs)


class OptimizationManager(ABC, ReprMixin):

    def __init__(self, loss_op, **kwargs) -> None:
        self._idx = -1
        self._minimize_kwargs = {'loss': loss_op, **kwargs}
        self.optims = []  # type: t.List[tf.Optimizer]
        self.optim_op = None

    @property
    def optim(self):
        """ The current activating optimizer. """
        return self.optims[self._idx]

    def add_optimizer(self, optim_type, **kwargs):
        self.optims += _init_optimizer(optim_type, **kwargs),
        if len(self.optims) == 1:
            self.next()

    def has_next(self):
        return self._idx < len(self.optims) - 1

    def next(self):
        """ Set the next available optimizer for training. """
        if self.has_next():
            self._idx += 1
            self.optim_op = (self.optim.minimize(**self._minimize_kwargs))
            log.info('Train model using %s Optimizer' % self.optim.get_name())

    @abstractmethod
    def update(self, step_acc, global_step) -> None:
        """ Update the step performance and check the optimizer switch
        condition. """
        pass


class NotChange(OptimizationManager):
    """ Do not change optimizer during training. """

    def update(self, step_acc, global_step) -> None:
        pass


class ProgressCheck(OptimizationManager):
    """ Switch the optimizer when average performance of the latested
    `avg_steps` does not improve upto `min_delta`.

    Arguements:
        min_delta: Progress threshold value.
        avg_steps: The number of the last steps used as the baseline.
        patience: The maximum number of consequtive update check allowed before
            a optimzer switch.
    """
    def __init__(self, min_delta: float, avg_steps: int, patience: int,
            **kwargs) -> None:
        super(ProgressCheck, self).__init__(**kwargs)
        self.min_delta = min_delta
        self.avg_steps = avg_steps
        self.patience = patience
        self._step_q = []   # type: t.List[int]
        self._prfm_q = []   # type: t.List[float]
        self._pat_cnt = 0

    def update(self, step_prfm, global_step) -> None:
        if not self.has_next():
            return
        self._step_q.insert(0, global_step)
        self._prfm_q.insert(0, step_prfm)
        while True:
            first_step = self._step_q[-1]
            if first_step > global_step - self.avg_steps:
                break
            self._step_q.pop()
            self._prfm_q.pop()

        avg_prfm = np.mean(self._prfm_q)
        if (avg_prfm + self.min_delta > step_prfm and
                global_step > self.avg_steps):
            self._pat_cnt += 1
            log.warning('Average performance at step %d-%d and current'
                        'performance are: %.4f and %.4f, patience: %d'
                        % (self._step_q[-1],
                           self._step_q[0],
                           avg_prfm,
                           step_prfm,
                           self._pat_cnt))
            if self._pat_cnt >= self.patience:
                self.next()
                self._pat_cnt = 0
        else:
            self._pat_cnt = 0

