import numpy as np
import openmdao.api as om


class MassPropComp(om.ExplicitComponent):
    """
    Example mass-propagation for the total delta-V using rocket equation.
    m_f = m0 * exp(- dv_total / (Isp*g0)), ignoring staging or multi-impulse
    detail. Here we just show how you might compute final mass or mass
    fraction.
    """

    def initialize(self):
        self.options.declare(
            "g0", default=9.80665e-3,
            desc="standard gravity in km/s^2 if wanted")

    def setup(self):
        self.add_input("dv_total", val=0.0, units="km/s")
        self.add_input("m0", val=1000.0, units="kg", desc="Initial mass")
        self.add_input(
            "Isp", val=300.0, units="s",
            desc="constant Isp (for a toy model)")

        self.add_output(
            "m_f", val=0.0, units="kg", desc="Final mass after burn")

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        dv = inputs["dv_total"]
        m0 = inputs["m0"]
        Isp = inputs["Isp"]
        g0 = self.options["g0"]

        outputs["m_f"] = m0 * np.exp(-dv / (Isp*g0))
