import numpy as np
import openmdao.api as om


class DeltaVComp(om.ExplicitComponent):
    """
    Computes the two impulses for a Hohmann transfer:
      dv1 = vp - v1
      dv2 = v2 - va
    Then sums them up as total_dv.
    """

    def setup(self):
        # Inputs
        self.add_input(
            "v1", val=1.0, units="km/s",
            desc="circ. velocity initial orbit")
        self.add_input(
            "v2", val=1.0, units="km/s",
            desc="circ. velocity final orbit")
        # self.add_input(
        #     'd_inclination', val=1.0, units='rad',
        #     desc='Change in orbit inclinations')
        self.add_input(
            "vp", val=0.0, units="km/s",
            desc="periapsis velocity of transfer ellipse")
        self.add_input(
            "va", val=0.0, units="km/s",
            desc="apoapsis velocity of transfer ellipse")

        # Outputs
        # self.add_output("dv1", val=0.0, units="km/s", desc="First impulse")
        # self.add_output("dv2", val=0.0, units="km/s", desc="Second impulse")
        self.add_output(
            "delta_v", val=0.0, units="km/s", desc="Delta V")

        # Finite difference for partials (for now)
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        v1 = inputs["v1"]
        v2 = inputs["v2"]
        vp = inputs["vp"]
        va = inputs["va"]

        # The classical Hohmann formula for 2 impulses
        # outputs["dv1"] = np.abs(vp - v1)
        # outputs["dv2"] = np.abs(v2 - va)
        outputs["delta_v"] = np.abs(vp - v1) + np.abs(v2 - va)
