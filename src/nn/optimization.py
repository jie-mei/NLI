import tensorflow as tf
import collections
from util.log import exec_log as log


LearningRateManager = collections.namedtuple(
    'LearningRateManager',
    'learning_rates global_step decay_steps decay_rate')


def get_optimizer(optimizer_names: list, lr_manager: LearningRateManager, minimize_args: dict, **kwargs):
    optimizer_name_placeholder = tf.placeholder_with_default(optimizer_names[0], (), name='optimizer_name')
    if len(optimizer_names) == 1:
        optimizer = _make_optimizer(optimizer_names[0], lr_manager, **kwargs).minimize(**minimize_args)
    else:
        optimizers = list([
            _make_optimizer(opt_name, lr_manager, optimizer_id=i, **kwargs)
            for i, opt_name in enumerate(optimizer_names)])
        optimizer_ops = list([(lambda opt=opt: opt.minimize(**minimize_args)) for opt in optimizers])
        choose_optimizer = [
            (tf.equal(optimizer_name_placeholder, tf.constant(name)), opt)
            for name, opt in zip(optimizer_names, optimizer_ops)]
        optimizer = tf.case(choose_optimizer,
            default=optimizer_ops[0], exclusive=True)
    return optimizer


def _get_lr(lr_manager, optimizer_id=0):
    if lr_manager.decay_steps:
        if lr_manager.decay_rate <= 0:
            raise ValueError('decay_rate must be positive, but is: {}'.format(lr_manager.decay_rate))
        if lr_manager.global_step is None:
            raise ValueError('global_step is requried for lr decay')

        learning_rate = lr_manager.learning_rates[optimizer_id]
        tf.summary.scalar(name='lr', tensor=learning_rate)
        return tf.train.exponential_decay(
            learning_rate, lr_manager.global_step, lr_manager.decay_steps, lr_manager.decay_rate)
    else:
        return lr_manager.learning_rates[optimizer_id]


def _make_optimizer(type_name: str, lr_manager, optimizer_id=0, **kwargs):
    kwargs['name'] = "optimizer"
    kwargs['learning_rate'] = _get_lr(lr_manager, optimizer_id)
    log.debug('Model optimzation using %s' % type_name)

    return getattr(tf.train, type_name)(**kwargs)
