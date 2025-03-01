import openmdao.api as om
from delta_v_comp import DeltaVComp
# from mass_prop_comp import MassPropComp
from transfer_orbit_comp import TransferOrbitComp
from orbital_param_comp import OrbitalParamsComp
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

if __name__ == "__main__":

    # Create the OpenMDAO problem
    prob = om.Problem()

    model = prob.model

    # Subsystems for Earth and Mars orbits
    model.add_subsystem('earth_orbit', subsys=OrbitalParamsComp(),
                        promotes_inputs=[('r', 'r1'), 'mu'])
    model.add_subsystem('mars_orbit', subsys=OrbitalParamsComp(),
                        promotes_inputs=[('r', 'r2'), 'mu'])

    model.add_subsystem('transfer_orbit', subsys=TransferOrbitComp(),
                        promotes_inputs=[('rp', 'r1'), ('ra', 'r2'), 'mu_sun'])

    # First impulse
    model.add_subsystem('dv1', subsys=DeltaVComp())
    model.connect('earth_orbit.v_circ', 'dv1.v1')  # initial orbit velocity
    model.connect('transfer_orbit.vp', 'dv1.v2')  # periapsis velocity of transfer

    # Second impulse
    model.add_subsystem('dv2', subsys=DeltaVComp())
    model.connect('transfer_orbit.va', 'dv2.v1')   # apoapsis velocity of transfer
    model.connect('mars_orbit.v_circ', 'dv2.v2')   # final orbit velocity

    # Summation of impulses => total delta-V
    model.add_subsystem('dv_total',
                        subsys=om.ExecComp('delta_v = dv1 + dv2',
                                           delta_v={'units': 'km/s'},
                                           dv1={'units': 'km/s'},
                                           dv2={'units': 'km/s'}),
                        promotes=['delta_v'])
    model.connect('dv1.delta_v', 'dv_total.dv1')
    model.connect('dv2.delta_v', 'dv_total.dv2')

    # # Add an independent variable component to set r1, r2, mu, etc.
    # indeps = model.add_subsystem(
    #     "indeps", om.IndepVarComp(), promotes=["*"])
    # indeps.add_output(
    #     "mu", 1.32712440018e11,
    #     units="km**3/s**2")  # e.g. near-sun, or use Earth->Mars
    # indeps.add_output(
    #     "r1", 1.0e8, units="km")   # some nominal orbit radius
    # indeps.add_output(
    #     "r2", 2.279e8,
    #     units="km")  # bigger orbit radius (like an approximation to Mars)
    # indeps.add_output("m0", 1000.0, units="kg")
    # indeps.add_output("Isp", 320.0, units="s")

    # # Add subsystems
    # model.add_subsystem("orbits", OrbitalParamsComp(), promotes=["*"])
    # model.add_subsystem("transfer", TransferOrbitComp(), promotes=["*"])
    # model.add_subsystem("deltaV", DeltaVComp(), promotes=["*"])
    # model.add_subsystem("massprop", MassPropComp(), promotes=["*"])

    prob.driver = om.ScipyOptimizeDriver()

    # For demonstration, let’s treat r1 (Earth orbit radius) and r2 (Mars orbit radius)
    # as design variables. We'll try to find the combination that yields minimal total Delta-V.
    # In reality, you'd often fix Earth radius and Mars radius, but we do this
    # to mimic the "optimize" pattern from the original example.
    model.add_design_var('r1', lower=6.0e3,  upper=7.5e3)   # e.g. variation around LEO altitude
    model.add_design_var('r2', lower=3.4e3,  upper=6.4e3)   # e.g. variation around Mars orbit
    model.add_objective('delta_v')
    # model.add_objective('tof')

    # -----------------------
    # Input defaults
    # -----------------------
    # Use solar mu: ~1.32712440018e11 km^3/s^2, or approximate as needed
    model.set_input_defaults('mu_sun', val=1.327e11, units='km**3/s**2')
    model.set_input_defaults('mu_earth', val=3.968e5, units='km**3/s**2')
    model.set_input_defaults('mu_mars', val=4.283e4, units='km**3/s**2')

    # Some nominal guesses
    model.set_input_defaults('r1', val=6.78e3,  units='km')   # ~ LEO radius from Earth center
    model.set_input_defaults('r2', val=4.4e3,  units='km')   # ~ Radius of orbit around Mars

    # Setup the problem
    prob.setup()

    # Execute the model
    prob.run_model()

    # Print some results
    # print("=== Non Optimal Results ===")
    print(f"\n\nInitial (Earth) orbit radius (r1) = {prob['r1'][0]:.3g} km")
    print(f"Final (Mars) orbit radius (r2)  = {prob['r2'][0]:.3g} km")
    print('Delta-V (km/s):', prob['delta_v'][0])
    # print('Time of Flight (days):', prob['tof'][0]/86400)
    # print(f"Circular velocity v1    = {prob['v_circ'][0]:.5g} km/s")
    # print(f"Circular velocity v2    = {prob['v_circ'][0]:.5g} km/s")
    # print(f"Transfer peri. velocity = {prob['vp'][0]:.5g} km/s")
    # print(f"Transfer apo. velocity  = {prob['va'][0]:.5g} km/s")
    # print(f"Time of flight (TOF)    \
    #     = {prob['tof'][0]:.5g} s  (~{prob['tof'][0]/86400:.2f} days)")
    # # print(f"dv1                     = {prob['dv1'][0]:.5g} km/s")
    # # print(f"dv2                     = {prob['dv2'][0]:.5g} km/s")
    # print(f"dv_total                = {prob['dv_total'][0]:.5g} km/s")
    # print(f"Initial mass            = {prob['m0'][0]:.5g} kg")
    # print(f"Final mass              = {prob['m_f'][0]:.5g} kg")

    prob.run_driver()

    # # Print some results
    # print("\n\n=== Optimal Results ===")
    print(f"\nInitial orbit radius (r1) = {prob['r1'][0]:.3g} km")
    print(f"Final orbit radius (r2)  = {prob['r2'][0]:.3g} km")
    print('Optimized Delta-V (km/s):', prob['delta_v'][0])
    # print('Optimized Time of Flight (days):', prob['tof'][0]/86400)
    # print(f"Circular velocity v1    = {prob['v_circ'][0]:.5g} km/s")
    # print(f"Circular velocity v2    = {prob['v_circ'][0]:.5g} km/s")
    # print(f"Transfer peri. velocity = {prob['vp'][0]:.5g} km/s")
    # print(f"Transfer apo. velocity  = {prob['va'][0]:.5g} km/s")
    # print(f"Time of flight (TOF)    \
    #     = {prob['tof'][0]:.5g} s  (~{prob['tof'][0]/86400:.2f} days)")
    # # print(f"dv1                     = {prob['dv1'][0]:.5g} km/s")
    # # print(f"dv2                     = {prob['dv2'][0]:.5g} km/s")
    # print(f"dv_total                = {prob['dv_total'][0]:.5g} km/s")
    # print(f"Initial mass            = {prob['m0'][0]:.5g} kg")
    # print(f"Final mass              = {prob['m_f'][0]:.5g} kg")
