import numpy as np
import openmdao.api as om


class HelioTransferOrbitComp(om.ExplicitComponent):
    """
    Computes the heliocentric Hohmann transfer parameters.
    Inputs:
      - r_Earth_helio: Earth's heliocentric orbit radius (≈1.496e8 km)
      - r_Mars_helio: Mars' heliocentric orbit radius (≈2.279e8 km)
      - mu_sun: Sun's gravitational parameter (≈1.3271244e11 km^3/s^2)
    Outputs:
      - v_dep_helio: transfer ellipse speed at Earth's orbit (injection speed)
      - v_arr_helio: transfer ellipse speed at Mars' orbit (arrival speed)
      - tof_helio: one-way transfer time (s)
    """
    def setup(self):
        self.add_input('r_Earth_helio', val=1.0, units='km',
                       desc="Earth's heliocentric orbit radius")

        self.add_input('r_Mars_helio',  val=1.0, units='km',
                       desc="Mars' heliocentric orbit radius")

        self.add_input('mu_sun', val=1.0, units='km**3/s**2',
                       desc="Sun's gravitational parameter")

        self.add_output('v_dep_helio', val=1.0, units='km/s',
                        desc="Transfer speed at Earth's orbit")

        self.add_output('v_arr_helio', val=1.0, units='km/s',
                        desc="Transfer speed at Mars' orbit")

        self.add_output('tof_helio', val=1.0, units='s',
                        desc="Heliocentric time of flight (one way)")

        self.declare_partials(of='v_dep_helio', wrt='r_Earth_helio')
        self.declare_partials(of='v_dep_helio', wrt='r_Mars_helio')
        self.declare_partials(of='v_dep_helio', wrt='mu_sun')

        self.declare_partials(of='v_arr_helio', wrt='r_Earth_helio')
        self.declare_partials(of='v_arr_helio', wrt='r_Mars_helio')
        self.declare_partials(of='v_arr_helio', wrt='mu_sun')

        self.declare_partials(of='tof_helio', wrt='r_Earth_helio')
        self.declare_partials(of='tof_helio', wrt='r_Mars_helio')
        self.declare_partials(of='tof_helio', wrt='mu_sun')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        rE = inputs['r_Earth_helio']
        rM = inputs['r_Mars_helio']
        mu = inputs['mu_sun']
        a = 0.5 * (rE + rM)
        outputs['v_dep_helio'] = np.sqrt(mu*(2.0/rE - 1.0/a))
        outputs['v_arr_helio'] = np.sqrt(mu*(2.0/rM - 1.0/a))
        outputs['tof_helio'] = np.pi * np.sqrt(a**3 / mu)

    def compute_partials(self, inputs, partials):
        rE = inputs['r_Earth_helio']
        rM = inputs['r_Mars_helio']
        mu = inputs['mu_sun']
        a = 0.5 * (rE + rM)
        v_dep_helio = np.sqrt(mu*(2.0/rE - 1.0/a))
        v_arr_helio = np.sqrt(mu*(2.0/rM - 1.0/a))
        tof_helio = np.pi * np.sqrt(a**3 / mu)

        partials['v_dep_helio', 'r_Earth_helio'] = - ((2 * mu) / rE**2) / \
            v_dep_helio
        partials['v_dep_helio', 'r_Mars_helio'] = 0
        partials['v_dep_helio', 'mu_sun'] = np.sqrt((2.0/rE - 1.0/a)) / \
            v_dep_helio

        partials['v_arr_helio', 'r_Earth_helio'] = 0
        partials['v_arr_helio', 'r_Mars_helio'] = - ((2 * mu) / rM**2) / \
            v_arr_helio
        partials['v_arr_helio', 'mu_sun'] = np.sqrt((2.0/rM - 1.0/a)) / \
            v_dep_helio

        partials['tof_helio', 'r_Earth_helio'] = 0
        partials['tof_helio', 'r_Mars_helio'] = 0
        partials['tof_helio', 'mu_sun'] = - ((np.pi**2 * a**3) /
                                             (2 * mu**2)) / tof_helio


class HelioInjectionMatchComp(om.ExplicitComponent):
    """
    Computes the hyperbolic excess speed at Earth required for heliocentric
    injection.
    v_inf = |v_dep_helio - v_Earth_helio|, where
    v_Earth_helio = sqrt(mu_sun / r_Earth_helio).
    """
    def setup(self):
        self.add_input('v_dep_helio', val=1.0, units='km/s')
        self.add_input('r_Earth_helio', val=1.0, units='km')
        self.add_input('mu_sun', val=1.0, units='km**3/s**2')
        self.add_output('v_inf', val=1.0, units='km/s',
                        desc="Hyperbolic excess speed at Earth")

        self.declare_partials('*', '*', method='fd')
        # self.declare_partials(of='v_inf', wrt='v_dep_helio')
        # self.declare_partials(of='v_inf', wrt='r_Earth_helio')
        # self.declare_partials(of='v_inf', wrt='mu_sun')

    def compute(self, inputs, outputs):
        rE = inputs['r_Earth_helio']
        mu = inputs['mu_sun']
        # v_circ_helio = np.sqrt(mu / rE)
        outputs['v_inf'] = np.abs(inputs['v_dep_helio'] - np.sqrt(mu / rE))

    # def compute_partials(self, inputs, partials):
    #     rE = inputs['r_Earth_helio']
    #     mu = inputs['mu_sun']
