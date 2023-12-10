'''Heat equation setup.'''

import torch
import torch.nn as nn


class HeatConduction1D(nn.Module):
    '''
    Simple 1D heat conduction problem.

    Summary
    -------
    This class represents a simple problem based on the 1D heat equation.
    It is posed in a way that allows the PDE to have an analytical solution.
    Dirichlet boundary conditions are imposed such that the
    temperature at both ends of the rod is kept at zero.
    The initial temperature along the rod is described by a sine function.
    There is no heat source (or sink) within the rod.

    Note that the heat transfer setup here is implemented as a PyTorch module,
    because this allows for a smooth integration into the PINN framework.

    Parameters
    ----------
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
                 alpha=1.0,
                 length=1.0,
                 maxtime=1.0,
                 n=1):

        super().__init__()

        alpha = abs(alpha)
        length = abs(length)
        maxtime = abs(maxtime)
        n = abs(int(n))

        self.register_buffer('alpha', torch.as_tensor(alpha))
        self.register_buffer('length', torch.as_tensor(length))
        self.register_buffer('maxtime', torch.as_tensor(maxtime))
        self.register_buffer('n', torch.as_tensor(n))

    @property
    def sqrt_lambda(self):
        return self.n * torch.pi / self.length

    @staticmethod
    def boundary_condition(t):
        '''Return zeros as boundary condition.'''
        return torch.zeros_like(t)

    def initial_condition(self, x):
        '''Evaluate the initial condition.'''
        return torch.sin(self.sqrt_lambda * x)

    def exact_solution(self, t, x):
        '''Compute the exact solution.'''
        sqrt_lambda = self.sqrt_lambda

        space_part = torch.sin(sqrt_lambda * x)
        time_part = torch.exp(-sqrt_lambda**2 * self.alpha * t)

        u = space_part * time_part
        return u

    def forward(self, t, x):
        return self.exact_solution(t=t, x=x)

