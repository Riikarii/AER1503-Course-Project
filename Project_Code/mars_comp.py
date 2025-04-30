import numpy as np
import openmdao.api as om


class MarsOrbitComp(om.ExplicitComponent):
    """
    Computes the circular orbital velocity about Mars (Mars-centered).
    """
    def setup(self):
        self.add_input('r_LMO', val=1.0, units='km',
                       desc="LMO radius from Mars center")

        self.add_input('mu_mars', val=1.0, units='km**3/s**2',
                       desc="Mars' gravitational parameter")

        self.add_output('v_LMO', val=1.0, units='km/s',
                        desc="Circular orbit speed around Mars")

        self.declare_partials(of='v_LMO', wrt='r_LMO')
        self.declare_partials(of='v_LMO', wrt='mu_mars')

    def compute(self, inputs, outputs):
        r = inputs['r_LMO']
        mu = inputs['mu_mars']
        outputs['v_LMO'] = np.sqrt(mu / r)

    def compute_partials(self, inputs, partials):
        r = inputs['r_LMO']
        mu = inputs['mu_mars']
        v_LMO = np.sqrt(mu / r)

        partials['v_LMO', 'mu_mars'] = 0.5 / (r * v_LMO)
        partials['v_LMO', 'r_LMO'] = -0.5 * mu / (v_LMO * r ** 2)


class MarsInjectionMatchComp(om.ExplicitComponent):
    """
    Computes the hyperbolic excess speed at Mars arrival.
    """
    def setup(self):
        self.add_input('v_arr_helio', val=1.0, units='km/s',
                       desc="Transfer speed at Mars' orbit")

        self.add_input('r_Mars_helio', val=1.0, units='km',
                       desc="Mars' heliocentric orbit radius")

        self.add_input('mu_sun', val=1.0, units='km**3/s**2',
                       desc="Sun's gravitational parameter")

        self.add_output('v_inf_mars', val=1.0, units='km/s',
                        desc="Hyperbolic excess speed at Mars")

        self.declare_partials(of='v_inf_mars', wrt='v_arr_helio')
        self.declare_partials(of='v_inf_mars', wrt='r_Mars_helio')
        self.declare_partials(of='v_inf_mars', wrt='mu_sun')

    def compute(self, inputs, outputs):
        rM = inputs['r_Mars_helio']
        mu = inputs['mu_sun']
        v_arr_helio = inputs['v_arr_helio']
        outputs['v_inf_mars'] = np.abs(v_arr_helio - np.sqrt(mu / rM))

    def compute_partials(self, inputs, partials):
        rM = inputs['r_Mars_helio']
        mu = inputs['mu_sun']
        v_arr_helio = inputs['v_arr_helio']

        top1 = (v_arr_helio - np.sqrt(mu / rM))
        bottom1 = np.abs(v_arr_helio - np.sqrt(mu / rM))
        partials['v_inf_mars', 'v_arr_helio'] = top1 / bottom1

        top2 = (mu * (np.sqrt(mu / rM) - v_arr_helio))
        bottom2 = 2 * np.abs(np.sqrt(mu / rM) - v_arr_helio) * \
            np.sqrt(mu / rM) * rM**2
        partials['v_inf_mars', 'r_Mars_helio'] = - top2 / bottom2

        top3 = np.sqrt(mu / rM) - v_arr_helio
        bottom3 = 2 * rM * np.sqrt(mu / rM) * np.abs(top2)
        partials['v_inf_mars', 'mu_sun'] = top3 / bottom3


class MarsCaptureDVComp(om.ExplicitComponent):
    """
    Computes the delta-V required for Mars orbit insertion.
    """
    def setup(self):
        self.add_input('v_inf_mars', val=1.0, units='km/s',
                       desc="Hyperbolic excess speed at Mars")

        self.add_input('v_LMO', val=1.0, units='km/s',
                       desc='LEO circular velocity')

        self.add_output('dv_capture', val=1.0, units='km/s',
                        desc="Delta-V for Mars capture burn")

        self.declare_partials('*', '*', method='fd')

        # Defining partials for capture delta-V makes the optimizer converge
        # without optimizing, so let OpenMDAO define partials automatically
        # using finite differencing

        # self.declare_partials(of='dv_capture', wrt='v_LMO')
        # self.declare_partials(of='dv_capture', wrt='v_inf_mars')

    def compute(self, inputs, outputs):
        v_inf = inputs['v_inf_mars']
        v_LMO = inputs['v_LMO']
        outputs['dv_capture'] = np.sqrt(v_inf**2 + v_LMO**2) - v_inf

    # def compute_partials(self, inputs, partials):
    #     v_inf = inputs['v_inf_mars']
    #     v_LMO = inputs['v_LMO']

    #     partials['dv_capture', 'v_LMO'] = v_LMO / \
    #         (np.sqrt(v_LMO**2 + v_inf**2)) - 1

    #     partials['dv_capture', 'v_inf_mars'] = v_inf / \
    #         (np.sqrt(v_LMO**2 + v_inf**2))
