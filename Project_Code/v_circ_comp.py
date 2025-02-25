import numpy as np
import openmdao.api as om


class VCircComp(om.ExplicitComponent):
    """
    Computes the circular orbit velocity given a radius and gravitational
    parameter.
    """
    def initialize(self):
        pass

    def setup(self):
        self.add_input('r',
                       val=1.0,
                       desc='Radius from central body',
                       units='km')

        self.add_input('mu',
                       val=1.0,
                       desc='Gravitational parameter of central body',
                       units='km**3/s**2')

        self.add_output('vcirc',
                        val=1.0,
                        desc='Circular orbit velocity at given radius '
                             'and gravitational parameter',
                        units='km/s')

        self.declare_partials(of='vcirc', wrt='r')
        self.declare_partials(of='vcirc', wrt='mu')

    def compute(self, inputs, outputs):
        r = inputs['r']
        mu = inputs['mu']

        outputs['vcirc'] = np.sqrt(mu / r)

    def compute_partials(self, inputs, partials):
        r = inputs['r']
        mu = inputs['mu']
        vcirc = np.sqrt(mu / r)

        partials['vcirc', 'mu'] = 0.5 / (r * vcirc)
        partials['vcirc', 'r'] = -0.5 * mu / (vcirc * r ** 2)
