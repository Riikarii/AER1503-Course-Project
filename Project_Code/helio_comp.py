import numpy as np
import openmdao.api as om


class HelioTransferOrbitComp(om.ExplicitComponent):
    """
    Computes the heliocentric Hohmann transfer parameters.
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
            v_arr_helio

        partials['tof_helio', 'r_Earth_helio'] = 3 * np.pi * (2*a)**2 / \
            2**2.5 * mu * np.sqrt((2*a)**3 / mu)

        partials['tof_helio', 'r_Mars_helio'] = 3 * np.pi * (2*a)**2 / \
            2**2.5 * mu * np.sqrt((2*a)**3 / mu)

        partials['tof_helio', 'mu_sun'] = - ((np.pi**2 * a**3) /
                                             (2 * mu**2)) / tof_helio


class HelioInjectionMatchComp(om.ExplicitComponent):
    """
    Computes the hyperbolic excess speed at Earth required for heliocentric
    injection.
    """
    def setup(self):
        self.add_input('v_dep_helio', val=1.0, units='km/s',
                       desc="Transfer speed at Earth's orbit")

        self.add_input('r_Earth_helio', val=1.0, units='km',
                       desc="Earth's heliocentric orbit radius")

        self.add_input('mu_sun', val=1.0, units='km**3/s**2',
                       desc="Sun's gravitational parameter")

        self.add_output('v_inf', val=1.0, units='km/s',
                        desc="Hyperbolic excess speed at Earth")

        self.declare_partials(of='v_inf', wrt='v_dep_helio')
        self.declare_partials(of='v_inf', wrt='r_Earth_helio')
        self.declare_partials(of='v_inf', wrt='mu_sun')

    def compute(self, inputs, outputs):
        rE = inputs['r_Earth_helio']
        mu = inputs['mu_sun']
        v_dep_helio = inputs['v_dep_helio']
        outputs['v_inf'] = np.abs(v_dep_helio - np.sqrt(mu / rE))

    def compute_partials(self, inputs, partials):
        rE = inputs['r_Earth_helio']
        mu = inputs['mu_sun']
        v_dep_helio = inputs['v_dep_helio']

        partials['v_inf', 'v_dep_helio'] = (v_dep_helio - np.sqrt(mu / rE)) / \
            np.abs(v_dep_helio - np.sqrt(mu / rE))

        top1 = (mu * (np.sqrt(mu / rE) - v_dep_helio))
        bottom1 = 2 * np.abs(np.sqrt(mu / rE) - v_dep_helio) * \
            np.sqrt(mu / rE) * rE**2
        partials['v_inf', 'r_Earth_helio'] = - top1 / bottom1

        top2 = np.sqrt(mu / rE) - v_dep_helio
        bottom2 = 2 * rE * np.sqrt(mu / rE) * np.abs(top2)
        partials['v_inf', 'mu_sun'] = top2 / bottom2
