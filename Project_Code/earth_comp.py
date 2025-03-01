import numpy as np
import openmdao.api as om


class EarthOrbitComp(om.ExplicitComponent):
    """
    Computes the circular orbital velocity in LEO (Earth-centered).
    """
    def setup(self):
        self.add_input('r_LEO', val=1.0, units='km',
                       desc='LEO radius from Earth center')
        self.add_input('mu_earth', val=1.0, units='km**3/s**2',
                       desc="Earth's gravitational parameter")

        self.add_output('v_LEO', val=1.0, units='km/s',
                        desc='Circular speed in LEO')

        self.declare_partials(of='v_LEO', wrt='r_LEO')
        self.declare_partials(of='v_LEO', wrt='mu_earth')

        # self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        r = inputs['r_LEO']
        mu = inputs['mu_earth']
        outputs['v_LEO'] = np.sqrt(mu / r)

    def compute_partials(self, inputs, partials):
        r = inputs['r_LEO']
        mu = inputs['mu_earth']
        v_LEO = np.sqrt(mu / r)

        partials['v_LEO', 'mu_earth'] = 0.5 / (r * v_LEO)
        partials['v_LEO', 'r_LEO'] = -0.5 * mu / (v_LEO * r ** 2)


class EarthEscapeDVComp(om.ExplicitComponent):
    """
    Computes the delta-V required to escape Earth from LEO.
    Uses the patched-conic formula:
      ΔV_escape = sqrt(v_LEO^2 + v_inf^2) - v_LEO,
    where v_inf is the required hyperbolic excess speed.
    """
    def setup(self):
        self.add_input('v_LEO', val=1.0, units='km/s',
                       desc='LEO circular velocity')

        self.add_input('v_inf', val=1.0, units='km/s',
                       desc='Hyperbolic excess speed required')

        self.add_output('dv_escape', val=1.0, units='km/s',
                        desc='Earth escape delta-V')

        self.declare_partials(of='dv_escape', wrt='v_LEO')
        self.declare_partials(of='dv_escape', wrt='v_inf')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        v_LEO = inputs['v_LEO']
        v_inf = inputs['v_inf']
        outputs['dv_escape'] = np.sqrt(v_LEO**2 + v_inf**2) - v_LEO

    def compute_partials(self, inputs, partials):
        v_LEO = inputs['v_LEO']
        v_inf = inputs['v_inf']
        # dv_esc = np.sqrt(v_LEO**2 + v_inf**2) - v_LEO

        partials['dv_escape', 'v_LEO'] = v_LEO / \
            (np.sqrt(v_LEO**2 + v_inf**2)) - 1
        partials['dv_escape', 'v_inf'] = v_inf / (np.sqrt(v_LEO**2 + v_inf**2))
