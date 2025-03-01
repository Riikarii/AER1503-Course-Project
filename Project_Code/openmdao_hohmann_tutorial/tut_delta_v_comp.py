import numpy as np
import openmdao.api as om


class TutDeltaVComp(om.ExplicitComponent):
    """
    Compute the delta-V performed given the magnitude of two velocities
    and the angle between them.
    """
    def initialize(self):
        pass

    def setup(self):
        self.add_input('v1', val=1.0, desc='Initial velocity', units='km/s')
        self.add_input('v2', val=1.0, desc='Final velocity', units='km/s')
        self.add_input('dinc', val=1.0, desc='Plane change', units='rad')

        # Note:  We're going to use trigonometric functions on dinc.  The
        # automatic unit conversion in OpenMDAO comes in handy here.

        self.add_output('delta_v', val=0.0, desc='Delta-V', units='km/s')

        self.declare_partials(of='delta_v', wrt='v1')
        self.declare_partials(of='delta_v', wrt='v2')
        self.declare_partials(of='delta_v', wrt='dinc')

    def compute(self, inputs, outputs):
        v1 = inputs['v1']
        v2 = inputs['v2']
        dinc = inputs['dinc']

        outputs['delta_v'] = np.sqrt(
            v1 ** 2 + v2 ** 2 - 2.0 * v1 * v2 * np.cos(dinc))

    def compute_partials(self, inputs, partials):
        v1 = inputs['v1']
        v2 = inputs['v2']
        dinc = inputs['dinc']

        delta_v = np.sqrt(v1 ** 2 + v2 ** 2 - 2.0 * v1 * v2 * np.cos(dinc))

        partials['delta_v', 'v1'] = 0.5 / delta_v * \
            (2 * v1 - 2 * v2 * np.cos(dinc))
        partials['delta_v', 'v2'] = 0.5 / delta_v * \
            (2 * v2 - 2 * v1 * np.cos(dinc))
        partials['delta_v', 'dinc'] = 0.5 / delta_v * \
            (2 * v1 * v2 * np.sin(dinc))
