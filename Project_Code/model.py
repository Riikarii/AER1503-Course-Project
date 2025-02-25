import openmdao.api as om
from delta_v_comp import DeltaVComp
from transfer_orbit_comp import TransferOrbitComp
from v_circ_comp import VCircComp
"""
Two instances of VCircComp are used to compute the velocity of the spacecraft
in the initial and final circular orbits.

The TransferOrbitComp is used to compute the periapsis and apoapsis velocity
of the spacecraft in the transfer orbit.

Now we can use the DeltaVComp to provide the magnitude of the
delta-V at each of the two impulses.

We use two ExecComps to provide some simple calculations.
One sums the delta-Vs of the two impulses to provide the total delta-V of
the transfer. We will use this as the objective for the optimization.

The other ExecComp sums up the inclination change at each impulse.
We will provide this to the driver as a constraint to ensure that our total
inclination change meets our requirements.

Lastly, we provide unambiguous values and units for the gravitational
parameter, the radii of the two circular orbits, and the delta-V to
be performed at each of the two impulses.

We will use the initial and final radii of the orbits, and the inclination
change at each of the two impulses as our design variables.

To run the model, we provide values for the design variables
and invoke run_model.

To find the optimal solution for the model, we invoke run_driver,
where we have defined the driver of the problem to be ScipyOptimizeDriver.
"""

if __name__ == '__main__':

    prob = om.Problem()

    model = prob.model

    model.add_subsystem('leo', subsys=VCircComp(),
                        promotes_inputs=[('r', 'r1'), 'mu'])
    model.add_subsystem('geo', subsys=VCircComp(),
                        promotes_inputs=[('r', 'r2'), 'mu'])

    model.add_subsystem('transfer', subsys=TransferOrbitComp(),
                        promotes_inputs=[('rp', 'r1'), ('ra', 'r2'), 'mu'])

    model.add_subsystem('dv1', subsys=DeltaVComp(),
                        promotes_inputs=[('dinc', 'dinc1')])

    model.connect('leo.vcirc', 'dv1.v1')
    model.connect('transfer.vp', 'dv1.v2')

    model.add_subsystem('dv2', subsys=DeltaVComp(),
                        promotes_inputs=[('dinc', 'dinc2')])

    model.connect('transfer.va', 'dv2.v1')
    model.connect('geo.vcirc', 'dv2.v2')

    model.add_subsystem('dv_total',
                        subsys=om.ExecComp('delta_v=dv1+dv2',
                                           delta_v={'units': 'km/s'},
                                           dv1={'units': 'km/s'},
                                           dv2={'units': 'km/s'}),
                        promotes=['delta_v'])

    model.connect('dv1.delta_v', 'dv_total.dv1')
    model.connect('dv2.delta_v', 'dv_total.dv2')

    model.add_subsystem('dinc_total',
                        subsys=om.ExecComp('dinc=dinc1+dinc2',
                                           dinc={'units': 'deg'},
                                           dinc1={'units': 'deg'},
                                           dinc2={'units': 'deg'}),
                        promotes=['dinc', 'dinc1', 'dinc2'])

    prob.driver = om.ScipyOptimizeDriver()

    model.add_design_var('dinc1', lower=0, upper=28.5)
    model.add_design_var('dinc2', lower=0, upper=28.5)
    model.add_constraint('dinc', lower=28.5, upper=28.5, scaler=1.0)
    model.add_objective('delta_v', scaler=1.0)

    # set defaults for our promoted variables to remove ambiguities in value
    # and/or units
    model.set_input_defaults('r1', val=6778.0)
    model.set_input_defaults('r2', val=42164.0)
    model.set_input_defaults('mu', val=398600.4418)
    model.set_input_defaults('dinc1', val=0., units='deg')
    model.set_input_defaults('dinc2', val=28.5, units='deg')

    # Setup the problem
    prob.setup()

    # Execute the model with the given inputs
    prob.run_model()

    print(f"Delta-V (km/s): {prob['delta_v'][0]}")
    print("Inclination change split (deg): ",
          prob['dinc1'][0], prob['dinc2'][0], "\n")

    prob.run_driver()

    print(f"\nOptimized Delta-V (km/s): {prob['delta_v'][0]}")
    print("Inclination change split (deg): ",
          prob['dinc1'][0], prob['dinc2'][0])
