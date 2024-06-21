import numpy as np

def sgd(w, dw, config=None):
  
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):

    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None

    mu = config['momentum']
    lr = config['learning_rate']
    v = mu * v - lr * dw
    next_w = w + v

    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):

    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None

    config['cache']=config['decay_rate']*config['cache'] + (1-config['decay_rate'])*dw*dw
    next_w = w- config['learning_rate']*dw/(np.sqrt(config['cache'])+config['epsilon'])

    return next_w, config


def adam(w, dw, config=None):

    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None

    config['t'] += 1
    config['m'] = config['m'] * config['beta1'] + (1 - config['beta1']) * dw
    mt = config['m'] / (1 - config['beta1'] ** config['t'])
    config['v'] = config['v'] * config['beta2'] + (1 - config['beta2']) * dw * dw
    vt = config['v'] / (1 - config['beta2'] ** config['t'])
    next_w = w - config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon'])

    return next_w, config
