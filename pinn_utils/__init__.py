'''
PINN utilities.

Modules
-------
heat : Heat equation setup.
model : Model components.
pinn : Physics-informed NN.
train : PINN training.
vis : Visualization tools.

'''

from .heat import HeatConduction1D
from .model import make_activation, make_fc_model
from .pinn import PINN
from .train import test_pinn, train_pinn
from .vis import make_colors
