import openmdao.api as om
from earth_comp import EarthOrbitComp, EarthEscapeDVComp
from helio_comp import HelioTransferOrbitComp, HelioInjectionMatchComp
from mars_comp import MarsOrbitComp, MarsInjectionMatchComp, MarsCaptureDVComp
from total_delta_v_comp import TotalDeltaVComp

if __name__ == "__main__":
    prob = om.Problem()
    model = prob.model

    # Independent variable component for constants
    indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    # Earth constants
    indeps.add_output('mu_earth', val=3.986004418e5, units='km**3/s**2')
    # Sun constant
    indeps.add_output('mu_sun', val=1.32712440018e11, units='km**3/s**2')
    # Mars constant
    indeps.add_output('mu_mars', val=4.282837e4, units='km**3/s**2')

    # Heliocentric orbital radii
    indeps.add_output('r_Earth_helio', val=1.496e8, units='km')
    indeps.add_output('r_Mars_helio',  val=2.279e8, units='km')

    # Design variables:
    # LEO radius (Earth-centered) – typically near 6778 km.
    indeps.add_output('r_LEO', val=6778.0, units='km')
    # Mars parking orbit radius (Mars-centered) – choose, e.g., 4000 km.
    indeps.add_output('r_Mars', val=3580.0, units='km')

    # ---------------------
    # Earth Phase
    # ---------------------
    model.add_subsystem('earth_orbit', EarthOrbitComp(),
                        promotes_inputs=['r_LEO', 'mu_earth'],
                        promotes_outputs=['v_LEO'])
#     model.add_subsystem('earth_escape', EarthEscapeDVComp(),
#                         promotes_inputs=['v_LEO'])
    # v_inf will be connected below

    # ---------------------
    # Heliocentric Phase
    # ---------------------
    model.add_subsystem('heliotransfer', HelioTransferOrbitComp(),
                        promotes_inputs=['r_Earth_helio', 'r_Mars_helio',
                                         'mu_sun'],
                        promotes_outputs=['v_dep_helio', 'v_arr_helio',
                                          'tof_helio'])
    model.add_subsystem('helio_injection', HelioInjectionMatchComp(),
                        promotes_inputs=['r_Earth_helio', 'mu_sun'])

    model.add_subsystem('earth_escape', EarthEscapeDVComp(),
                        promotes_inputs=['v_LEO'])

    # Connect the heliocentric departure speed to the injection matching
    # component
    model.connect('v_dep_helio', 'helio_injection.v_dep_helio')
    # Connect the computed v_inf to Earth escape component
    model.connect('helio_injection.v_inf', 'earth_escape.v_inf')

    # ---------------------
    # Mars Phase
    # ---------------------
    model.add_subsystem('mars_orbit', MarsOrbitComp(),
                        promotes_inputs=['r_Mars', 'mu_mars'],
                        promotes_outputs=['v_Mars'])
    model.add_subsystem('mars_injection', MarsInjectionMatchComp(),
                        promotes_inputs=['r_Mars_helio', 'mu_sun'],
                        promotes_outputs=['v_inf_mars'])
    model.connect('v_arr_helio', 'mars_injection.v_arr_helio')
    model.add_subsystem('mars_capture', MarsCaptureDVComp(),
                        promotes_inputs=['v_Mars'])
    model.connect('v_inf_mars', 'mars_capture.v_inf_mars')

    # ---------------------
    # Total Mission ΔV
    # ---------------------
    model.add_subsystem('total_dv', TotalDeltaVComp(),
                        promotes_outputs=['delta_v_total'])
    model.connect('earth_escape.dv_escape', 'total_dv.dv_escape')
    model.connect('mars_capture.dv_capture', 'total_dv.dv_capture')

    # ---------------------
    # Optimization Setup
    # ---------------------
    # We will allow slight variation in LEO altitude and in the Mars parking
    # orbit.
    model.add_design_var('r_LEO', lower=6538.0, upper=8378.0)  # def of LEO
    model.add_design_var('r_Mars', lower=3490.0, upper=3590.0)  # def of LMO
    model.add_objective('delta_v_total')

    # Set the driver
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['tol'] = 1e-13  # machine epsilon accuracy
    # prob.driver.options['maxiter'] = 5

    prob.setup()
    prob.run_model()

    print("\n--- Non-optimized Results ---")
    print("LEO radius (r_LEO):", prob['r_LEO'][0], "km")
    print("Mars parking orbit (r_Mars):", prob['r_Mars'][0], "km")
    print("LEO velocity (v_LEO):", prob['v_LEO'][0], "km/s")
    print("Heliocentric injection speed (v_dep_helio):",
          prob['v_dep_helio'][0], "km/s")
    print("Hyperbolic excess at Earth (v_inf):",
          prob['helio_injection.v_inf'][0], "km/s")
    print("Earth escape ΔV (dv_escape):",
          prob['earth_escape.dv_escape'][0], "km/s")
    print("Heliocentric arrival speed (v_arr_helio):",
          prob['v_arr_helio'][0], "km/s")
    print("Hyperbolic excess at Mars (v_inf_mars):",
          prob['v_inf_mars'][0], "km/s")
    print("Mars circular speed (v_Mars):", prob['v_Mars'][0], "km/s")
    print("Mars capture ΔV (dv_capture):",
          prob['mars_capture.dv_capture'][0], "km/s")
    print("Total mission ΔV (delta_v_total):",
          prob['delta_v_total'][0], "km/s")
    print("Heliocentric time of flight (tof_helio):",
          prob['tof_helio'][0]/86400, "days\n\n")

    prob.run_driver()

    print("\n\n--- Optimized Results ---")
    print("Optimized LEO radius (r_LEO):", prob['r_LEO'][0], "km")
    print("Optimized Mars parking orbit (r_Mars):", prob['r_Mars'][0], "km")
    print("Optimized Total mission ΔV (delta_v_total):",
          prob['delta_v_total'][0], "km/s")
