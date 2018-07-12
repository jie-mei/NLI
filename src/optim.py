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

    def __init__(self, loss_op, clip_norm, **kwargs) -> None:
        self._idx = -1
        self._compute_kwargs = {'loss': loss_op, **kwargs}
        self.clip_norm = clip_norm
        self.optims = []  # type: t.List[tf.Optimizer]
        self.optim_op = None

    @property
    def feed_lr(self):
        """ If this optimization manager requires feeding learning rate in each
        training step. """
        return False

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
            log.info('Train model using %s Optimizer' % self.optim.get_name())
            grads_tvars = self.optim.compute_gradients(**self._compute_kwargs)
            if self.clip_norm:
                log.info('Apply global gradient clipping with norm %f' % self.clip_norm)
                grads = [grad for grad, tvar in grads_tvars]
                tvars = [tvar for grad, tvar in grads_tvars]
                grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
                grads_tvars = zip(grads, tvars)
            self.optim_op = self.optim.apply_gradients(grads_tvars)

    @abstractmethod
    def update(self, step_acc, global_step) -> None:
        """ Update the step performance and check the optimizer switch
        condition. """
        pass


class NotChange(OptimizationManager):
    """ Do not change optimizer during training. """

    def update(self, step_acc, global_step) -> None:
        pass


class LRUpdate(OptimizationManager):
    """ Update the learning rate when the validation performance does not
    improve upto `min_delta`.

    Arguements:
        min_delta: Progress threshold value.
        patience: The maximum number of consequtive update check allowed before
            a optimzer switch.
    """
    def __init__(self, loss_op, clip_norm, min_delta, patience, **kwargs) -> None:
        super(LRUpdate, self).__init__(loss_op, clip_norm, **kwargs)
        self._lr_op = []   # type: t.List[tf.Tensor]
        self._lr_val = []  # type: t.List[float]
        self.min_delta = min_delta
        if patience <= 1:
            raise ValueError('update patience should be greater than 1, got:',
                             patience)
        self.patience = patience
        self._no_update_cnt = 0
        self._max_acc = -float('inf')

    @property
    def feed_lr(self):
        return True

    def add_optimizer(self, optim_type, **kwargs):
        if self._idx >= 1:
            # TODO
            raise ValueError('Only one optimizer is permitted.')
        if 'decay_steps' in kwargs or 'decay_rate' in kwargs:
            raise ValueError('Auto-updating learning rate is not compatible'
                             'with exponential decay setup.')
        self._lr_op += tf.placeholder(tf.float32, shape=[]),
        self._lr_val += kwargs['learning_rate'],
        return super(LRUpdate, self).add_optimizer(optim_type, **kwargs)

    @property
    def lr_op(self):
        """ The current activating optimizer. """
        return self._lr_op[self._idx]

    @property
    def lr_val(self):
        """ The current activating optimizer. """
        return self._lr_val[self._idx]

    def update(self, step_acc, global_step) -> None:
        if step_acc < self._max_acc + self.min_delta:
            self._no_update_cnt += 1
            log.debug('No update for %d consequtive times.' %
                      self._no_update_cnt)
            if self._no_update_cnt >= self.patience:
                self._lr_val[self._idx] /= 2
                log.info('Half the current learning rate to : %f' %
                         self._lr_val[self._idx])
                self._no_update_cnt = 0
        else:
            self._no_update_cnt = 0
        self._max_acc = max(self._max_acc, step_acc)


class ProgressCheck(OptimizationManager):
    """ Switch the optimizer when average performance of the latest `avg_steps`
    does not improve upto `min_delta`.

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

