import numpy as np
import openmdao.api as om


class TransferOrbitComp(om.ExplicitComponent):
    """
     computes the velocity magnitude at periapsis and apoapsis of an orbit,
     given the radii of periapsis and apoapsis, and the gravitational
     parameter of the central body.
    """
    def initialize(self):
        pass

    def setup(self):
        self.add_input('mu',
                       val=398600.4418,
                       desc='Gravitational parameter of central body',
                       units='km**3/s**2')
        self.add_input('rp', val=7000.0, desc='periapsis radius', units='km')
        self.add_input('ra', val=42164.0, desc='apoapsis radius', units='km')

        self.add_output('vp', val=0.0, desc='periapsis velocity', units='km/s')
        self.add_output('va', val=0.0, desc='apoapsis velocity', units='km/s')

        # We're going to be lazy and ask OpenMDAO to approximate our
        # partials with finite differencing here.
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        mu = inputs['mu']
        rp = inputs['rp']
        ra = inputs['ra']

        a = (ra + rp) / 2.0
        e = (a - rp) / a
        p = a * (1.0 - e ** 2)
        h = np.sqrt(mu * p)

        outputs['vp'] = h / rp
        outputs['va'] = h / ra
