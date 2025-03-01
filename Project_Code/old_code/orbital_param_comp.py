import numpy as np
import openmdao.api as om


class OrbitalParamsComp(om.ExplicitComponent):
    """
    Provides fundamental orbital parameters for the initial and final circular
    orbits. For a simple Hohmann transfer, we'll treat the orbits as circular
    with known radii.
    """

    def setup(self):
        # Inputs: gravitational parameter mu, radius of initial orbit r1,
        # radius of final orbit r2
        self.add_input(
            "mu", val=1.0,
            units="km**3/s**2",
            desc="Gravitational Parameter of central body")
        self.add_input(
            "r", val=1.0, units="km",
            desc="Radius from central body")

        # Outputs: the circular velocities for each orbit
        self.add_output(
            "v_circ", val=0.0, units="km/s",
            desc="Circular orbit velocity given radius and mu")

        self.declare_partials(of='v_circ', wrt='r')
        self.declare_partials(of='v_circ', wrt='mu')

    def compute(self, inputs, outputs):
        mu = inputs["mu"]
        r = inputs["r"]

        outputs["v_circ"] = np.sqrt(mu / r)  # circular orbit velocity

    def compute_partials(self, inputs, partials):
        r = inputs['r']
        mu = inputs['mu']
        v_circ = np.sqrt(mu / r)

        partials['v_circ', 'mu'] = 0.5 / (r * v_circ)
        partials['v_circ', 'r'] = -0.5 * mu / (v_circ * r ** 2)
