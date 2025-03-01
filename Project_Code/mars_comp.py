import numpy as np
import openmdao.api as om


class MarsOrbitComp(om.ExplicitComponent):
    """
    Computes the circular orbital velocity about Mars (Mars-centered).
    """
    def setup(self):
        self.add_input('r_Mars', val=1.0, units='km',
                       desc="Mars parking orbit radius (from Mars center)")
        self.add_input('mu_mars', val=1.0, units='km**3/s**2',
                       desc="Mars gravitational parameter")
        self.add_output('v_Mars', val=1.0, units='km/s',
                        desc="Circular orbit speed around Mars")
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        r = inputs['r_Mars']
        mu = inputs['mu_mars']
        outputs['v_Mars'] = np.sqrt(mu / r)


class MarsInjectionMatchComp(om.ExplicitComponent):
    """
    Computes the hyperbolic excess speed at Mars arrival.
    v_inf_mars = |v_arr_helio - v_Mars_helio|,
    where v_Mars_helio = sqrt(mu_sun / r_Mars_helio).
    """
    def setup(self):
        self.add_input('v_arr_helio', val=1.0, units='km/s')
        self.add_input('r_Mars_helio', val=1.0, units='km')
        self.add_input('mu_sun', val=1.0, units='km**3/s**2')
        self.add_output('v_inf_mars', val=1.0, units='km/s',
                        desc="Hyperbolic excess speed at Mars")
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        rM = inputs['r_Mars_helio']
        mu = inputs['mu_sun']
        v_circ = np.sqrt(mu / rM)
        outputs['v_inf_mars'] = np.abs(inputs['v_arr_helio'] - v_circ)


class MarsCaptureDVComp(om.ExplicitComponent):
    """
    Computes the delta-V required for Mars orbit insertion.
    Uses:
      ΔV_capture = sqrt(v_inf_mars^2 + v_Mars^2) - v_inf_mars.
    """
    def setup(self):
        self.add_input('v_inf_mars', val=1.0, units='km/s')
        self.add_input('v_Mars', val=1.0, units='km/s')
        self.add_output('dv_capture', val=1.0, units='km/s',
                        desc="Delta-V for Mars capture burn")
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        v_inf = inputs['v_inf_mars']
        v_circ = inputs['v_Mars']
        outputs['dv_capture'] = np.sqrt(v_inf**2 + v_circ**2) - v_inf
