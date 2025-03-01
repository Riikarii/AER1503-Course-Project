import openmdao.api as om


class TotalDeltaVComp(om.ExplicitComponent):
    """
    Sums the Earth escape and Mars capture delta-V's to get the total mission
    delta-V.
    """
    def setup(self):
        self.add_input('dv_escape', val=1.0, units='km/s')
        self.add_input('dv_capture', val=1.0, units='km/s')
        self.add_output('delta_v_total', val=1.0, units='km/s')
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        outputs['delta_v_total'] = inputs['dv_escape'] + inputs['dv_capture']
