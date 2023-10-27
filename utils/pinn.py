'''Physics-informed NN.'''

import torch
import torch.nn as nn

from .heat import HeatConduction1D
from .model import make_fc_model


class PINN(nn.Module):
    '''
    PINN model class.

    Summary
    -------
    A PINN module for the simple 1D heat transfer problem is implemented.
    It contains a feedforward NN that predicts the temperature
    as a function of both a time and a spatial coordinate.
    Moreover, it allows for computing the regression and physics loss.
    The former loss penalizes predictions that deviate from actual data,
    while the latter measures violations of the physical constraints.

    Parameters
    ----------
    num_inputs : int
        Number of inputs.
    num_output : int
        Number of outputs.
    num_hidden : int or list thereof
        Number of hidden neurons.
    activation : str
        Activation function type.
    pde_weight : float
        PDE loss weight.
    bc_weight : float
        Boundary condition loss weight.
    ic_weight : float
        Initial condition loss weight.
    reduction : {'mean', 'sum'}
        Determines the loss reduction mode.
    alpha : float
        Thermal diffusivity coefficient.
    length : float
        Length of the rod.
    maxtime : float
        End of the time interval.
    n : int
        Determines the initial temperature.

    '''

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 num_hidden=None,
                 activation='tanh',
                 pde_weight=1.0,
                 bc_weight=1.0,
                 ic_weight=1.0,
                 reduction='mean',
                 alpha=1.0,
                 length=1.0,
                 maxtime=1.0,
                 n=1):

        super().__init__()

        # set up problem
        self.setup = HeatConduction1D(
            alpha=alpha,
            length=length,
            maxtime=maxtime,
            n=n
        )

        # create NN model
        self.model = make_fc_model(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_hidden=num_hidden,
            activation=activation
        )

        # store loss weights
        pde_weight = abs(pde_weight)
        bc_weight = abs(bc_weight)
        ic_weight = abs(ic_weight)

        self.register_buffer('pde_weight', torch.as_tensor(pde_weight))
        self.register_buffer('bc_weight', torch.as_tensor(bc_weight))
        self.register_buffer('ic_weight', torch.as_tensor(ic_weight))

        # initialize criterion
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(self, t, x):
        '''Predict PDE solution.'''
        t = torch.as_tensor(t)
        x = torch.as_tensor(x)

        t = ensure_2d(t)
        x = ensure_2d(x)

        tx = torch.cat((t, x), dim=1)
        u = self.model(tx)
        return u

    def sample_collocation(self,
                           num_pde,
                           num_bc,
                           num_ic):
        '''Sample collocation points uniformly.'''

        # randomly sample points
        pde_t = torch.rand(num_pde, 1) * self.setup.maxtime
        pde_x = torch.rand(num_pde, 1) * self.setup.length
        bc_t = torch.rand(num_bc, 1) * self.setup.maxtime
        ic_x = torch.rand(num_ic, 1) * self.setup.length

        # create output dict
        out_dict = {
            'pde_t': pde_t,
            'pde_x': pde_x,
            'bc_t': bc_t,
            'ic_x': ic_x
        }

        return out_dict

    def data_loss(self, t, x, y):
        '''Compute standard regression loss.'''
        u = self(t, x) # predict solution
        loss = self.criterion(u, y) # compute loss
        return loss

    def pde_loss(self, t, x):
        '''Compute PDE-based loss.'''

        # enable grad
        require_grad(t, x, requires_grad=True)

        # predict solution
        u = self(t, x)

        # autodiff prediction
        u_t = compute_grad(u, t)

        u_x = compute_grad(u, x)
        u_xx = compute_grad(u_x, x)

        # disable grad
        require_grad(t, x, requires_grad=False)

        # compute loss
        residual = u_t - self.setup.alpha * u_xx
        loss = self.criterion(residual, torch.zeros_like(residual))

        return loss

    def bc_loss(self, t):
        '''Compute boundary condition loss.'''

        # predict solution at left/right boundary
        u1 = self(t=t, x=torch.zeros_like(t))
        u2 = self(t=t, x=torch.full_like(t, fill_value=self.setup.length))

        # get boundary condition
        bc = self.setup.boundary_condition(t)

        # compute loss
        loss1 = self.criterion(u1, bc)
        loss2 = self.criterion(u2, bc)

        loss = loss1 + loss2

        if self.criterion.reduction == 'mean':
            loss = loss / 2

        return loss

    def ic_loss(self, x):
        '''Compute initial condition loss.'''

        # predict solution at initial time
        u0 = self(t=torch.zeros_like(x), x=x)

        # get initial condition
        ic = self.setup.initial_condition(x)

        # compute loss
        loss = self.criterion(u0, ic)

        return loss

    def physics_loss(self,
                     pde_t,
                     pde_x,
                     bc_t=None,
                     ic_x=None):
        '''Compute total physics loss.'''

        if bc_t is None:
            bc_t = pde_t

        if ic_x is None:
            ic_x = pde_x

        pde_loss = self.pde_loss(pde_t, pde_x)
        bc_loss = self.bc_loss(bc_t)
        ic_loss = self.ic_loss(ic_x)

        loss = self.pde_weight * pde_loss \
             + self.bc_weight * bc_loss \
             + self.ic_weight * ic_loss

        return loss


def ensure_2d(x):
    '''Ensure an appropriate tensor shape.'''
    if x.ndim <= 1:
        x = x.view(-1, 1)

    if x.ndim == 2:
        return x
    else:
        raise ValueError('Two or less tensor dimensions expected')


def require_grad(*tensors, requires_grad=True):
    '''Enable/disable gradient.'''
    for x in tensors:
        x.requires_grad = requires_grad


def compute_grad(outputs, inputs):
    '''Compute gradients.'''
    gradients = torch.autograd.grad(
        outputs=outputs.sum(),
        inputs=inputs,
        # grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )

    if len(gradients) == 1:
        return gradients[0]
    else:
        raise RuntimeError('Unexpected gradient tuple length encountered: {}'.format(len(gradients)))

