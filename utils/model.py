'''Model components.'''

from typing import Any
from collections.abc import Sequence

import torch.nn as nn


ACTIVATIONS = {
    'none': nn.Identity,
    'linear': nn.Identity,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'softplus':nn.Softplus,
    'swish': nn.SiLU
}


def make_activation(mode: str | None, **kwargs: Any) -> nn.Module:
    ''''Create activation function.'''

    if mode is None:
        activ = nn.Identity()
    elif isinstance(mode, str):
        activ = ACTIVATIONS[mode](**kwargs)
    else:
        raise TypeError(f'Unknown activation type: {type(mode)}')

    return activ


def make_fc_model(
    num_inputs: int,
    num_outputs: int,
    num_hidden: int | Sequence[int] | None = None,
    activation: str | None = 'tanh'
) -> nn.Sequential:
    '''
    Create FC model.

    Parameters
    ----------
    num_inputs : int
        Number of inputs.
    num_outputs : int
        Number of outputs.
    num_hidden : int, list of ints or None
        Number of hidden neurons.
    activation : str or None
        Activation function type.

    '''

    if num_hidden is None:
        num_hidden = []
    elif isinstance(num_hidden, int):
        num_hidden = [num_hidden]
    elif not isinstance(num_hidden, Sequence):
        raise TypeError(f'Unknown hidden num. type: {type(num_hidden)}')

    # collect feature numbers
    num_features = [num_inputs] + list(num_hidden) + [num_outputs]

    # assemble model layers
    layer_list = [] # type: list[nn.Module]

    for idx, (f1, f2) in enumerate(zip(num_features[:-1], num_features[1:])):

        # create FC layer
        dense = nn.Linear(f1, f2)
        layer_list.append(dense)

        # create activation
        is_not_last = (idx < len(num_features) - 2)

        if is_not_last:
            activ = make_activation(activation)
            layer_list.append(activ)

    # create sequential model
    model = nn.Sequential(*layer_list)

    return model

