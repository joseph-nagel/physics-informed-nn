'''
Some utilities.

Modules
-------
heat : Heat equation setup.
model : Model components.
pinn : Physics-informed NN.

'''

from .heat import HeatConduction1D

from .model import make_activation, make_fc_model

from .pinn import PINN

