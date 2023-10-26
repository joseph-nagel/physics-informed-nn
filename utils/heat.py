'''Heat equation setup.'''

import numpy as np


class SimpleHeatConduction1D():
    '''
    Simple 1D heat conduction problem.

    Summary
    -------
    This class represents a simple problem based on the 1D heat equation.
    It is posed in a way that allows the PDE to have an analytical solution.
    Dirichlet boundary conditions are imposed such that the
    temperature at both ends of the rod is kept at zero.
    The initial temperature along the rod is described by a sine function.

    Parameters
    ----------
    alpha : float
        Thermal diffusivity coefficient.
    length : float
        Length of the rod.
    n : int
        Determines the initial temperature.

    '''

    def __init__(self,
                 alpha=1.0,
                 length=1.0,
                 n=1):

        self.alpha = abs(alpha)
        self.length = abs(length)
        self.n = int(abs(n))

    @property
    def sqrt_lambda(self):
        return self.n * np.pi / self.length

    def initial_condition(self, x):
        '''Evaluate the initial condition.'''
        return np.sin(self.sqrt_lambda * x)

    def exact_solution(self, t, x):
        '''Compute the exact solution.'''
        t = np.asarray(t)
        x = np.asarray(x)

        sqrt_lambda = self.sqrt_lambda

        space_part = np.sin(sqrt_lambda * x)
        time_part = np.exp(-sqrt_lambda**2 * self.alpha * t)

        u = space_part * time_part
        return u

    def __call__(self, t, x):
        return self.exact_solution(t=t, x=x)

