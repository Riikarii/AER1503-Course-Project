import openmdao.api as om
import matplotlib.pyplot as plt
from earth_comp import EarthOrbitComp, EarthEscapeDVComp
from helio_comp import HelioTransferOrbitComp, HelioInjectionMatchComp
from mars_comp import MarsOrbitComp, MarsInjectionMatchComp, MarsCaptureDVComp
from total_delta_v_comp import TotalDeltaVComp


if __name__ == "__main__":

    # Loop to track multiple trials
    iterations = {'T1': [], 'T2': [], 'T3': []}
    objective_values = {'T1': [], 'T2': [], 'T3': []}
    design_var1 = {'T1': [], 'T2': [], 'T3': []}
    design_var2 = {'T1': [], 'T2': [], 'T3': []}

    r_LEO_vals = [6538, 7458, 8378]
    r_LMO_vals = [3490, 3540, 3590]

    for i, trial in enumerate(iterations.keys()):
        prob = om.Problem()
        model = prob.model

        # Independent variable component for constants
        indeps = model.add_subsystem('indeps', om.IndepVarComp(),
                                     promotes=['*'])
        # Constants
        indeps.add_output('mu_earth', val=3.986004418e5, units='km**3/s**2')
        indeps.add_output('mu_sun', val=1.32712440018e11, units='km**3/s**2')
        indeps.add_output('mu_mars', val=4.282837e4, units='km**3/s**2')
        indeps.add_output('r_Earth_helio', val=1.496e8, units='km')
        indeps.add_output('r_Mars_helio',  val=2.279e8, units='km')

        # LEO radius design variable
        indeps.add_output('r_LEO', val=float(r_LEO_vals[i]), units='km')

        # LMO radius design variable
        indeps.add_output('r_LMO', val=float(r_LMO_vals[i]), units='km')

        # Earth Phase
        model.add_subsystem('earth_orbit', EarthOrbitComp(),
                            promotes_inputs=['r_LEO', 'mu_earth'],
                            promotes_outputs=['v_LEO'])

        # Heliocentric Phase
        model.add_subsystem('heliotransfer', HelioTransferOrbitComp(),
                            promotes_inputs=['r_Earth_helio', 'r_Mars_helio',
                                             'mu_sun'],
                            promotes_outputs=['v_dep_helio', 'v_arr_helio',
                                              'tof_helio'])
        model.add_subsystem('helio_injection', HelioInjectionMatchComp(),
                            promotes_inputs=['r_Earth_helio', 'mu_sun'])
        model.add_subsystem('earth_escape', EarthEscapeDVComp(),
                            promotes_inputs=['v_LEO'])

        model.connect('v_dep_helio', 'helio_injection.v_dep_helio')
        model.connect('helio_injection.v_inf', 'earth_escape.v_inf')

        # Mars Phase
        model.add_subsystem('mars_orbit', MarsOrbitComp(),
                            promotes_inputs=['r_LMO', 'mu_mars'],
                            promotes_outputs=['v_LMO'])
        model.add_subsystem('mars_injection', MarsInjectionMatchComp(),
                            promotes_inputs=['r_Mars_helio', 'mu_sun'],
                            promotes_outputs=['v_inf_mars'])
        model.connect('v_arr_helio', 'mars_injection.v_arr_helio')
        model.add_subsystem('mars_capture', MarsCaptureDVComp(),
                            promotes_inputs=['v_LMO'])
        model.connect('v_inf_mars', 'mars_capture.v_inf_mars')

        # Total mission delta-V
        model.add_subsystem('total_dv', TotalDeltaVComp(),
                            promotes_outputs=['delta_v_total'])
        model.connect('earth_escape.dv_escape', 'total_dv.dv_escape')
        model.connect('mars_capture.dv_capture', 'total_dv.dv_capture')

        # Optimization Setup
        model.add_design_var('r_LEO', lower=6538.0, upper=8378.0)  # def of LEO
        model.add_design_var('r_LMO', lower=3490.0, upper=3590.0)  # def of LMO

        # Objective to be optimized
        model.add_objective('delta_v_total')

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['tol'] = 1e-13  # machine epsilon accuracy

        # Code to record convergence metrics
        recorder = om.SqliteRecorder('./optimization_convergence.sqlite')
        prob.driver.add_recorder(recorder)
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_objectives'] = True
        prob.setup()
        prob.run_model()

        print("\n--- Non-optimized Results ---")
        print("LEO radius (r_LEO):", prob['r_LEO'][0], "km")
        print("Mars parking orbit (r_LMO):", prob['r_LMO'][0], "km")
        print("LEO velocity (v_LEO):", prob['v_LEO'][0], "km/s")
        print("Heliocentric injection speed (v_dep_helio):",
              prob['v_dep_helio'][0], "km/s")
        print("Hyperbolic excess at Earth (v_inf):",
              prob['helio_injection.v_inf'][0], "km/s")
        print("Earth escape delta-V (dv_escape):",
              prob['earth_escape.dv_escape'][0], "km/s")
        print("Heliocentric arrival speed (v_arr_helio):",
              prob['v_arr_helio'][0], "km/s")
        print("Hyperbolic excess at Mars (v_inf_mars):",
              prob['v_inf_mars'][0], "km/s")
        print("Mars circular speed (v_LMO):", prob['v_LMO'][0], "km/s")
        print("Mars capture delta-V (dv_capture):",
              prob['mars_capture.dv_capture'][0], "km/s")
        print("Total mission delta-V (delta_v_total):",
              prob['delta_v_total'][0], "km/s")
        print("Heliocentric time of flight (tof_helio):",
              prob['tof_helio'][0]/86400, "days\n\n")

        prob.run_driver()

        print("\n\n--- Optimized Results ---")
        print("Optimized LEO radius (r_LEO):", prob['r_LEO'][0], "km")
        print("Optimized Mars parking orbit (r_LMO):", prob['r_LMO'][0], "km")
        print("Optimized Total mission delta-V (delta_v_total):",
              prob['delta_v_total'][0], "km/s")

        prob.cleanup()

        # Save trial data
        cr = om.CaseReader('optimization_convergence.sqlite')
        driver_cases = cr.list_cases('driver', recurse=False)
        for j, case_id in enumerate(driver_cases):
            case = cr.get_case(case_id)
            iterations[trial].append(j)
            objective_values[trial].append(
                case.get_objectives()['delta_v_total'])
            design_var1[trial].append(case.get_design_vars()['r_LEO'])
            design_var2[trial].append(case.get_design_vars()['r_LMO'])

    plt.figure(1, figsize=(10, 6))

    # Plot Objective Function Convergence
    plt.plot(iterations['T1'], objective_values['T1'], marker='o', color='red',
             label='Trial 1')
    plt.plot(iterations['T2'], objective_values['T2'], marker='o',
             color='blue',
             label='Trial 2')
    plt.plot(iterations['T3'], objective_values['T3'], marker='o',
             color='green',
             label='Trial 3')

    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value (km/s)')
    plt.title('Objective Function Convergence vs. Iteration')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("obj_converge.png")

    plt.figure(2, figsize=(10, 6))

    # Plot Design Variable Convergence
    # r_LEO
    plt.plot(iterations['T1'], design_var1['T1'], marker='o', color='red',
             label=r'Trial 1 $r_{LEO}$')
    plt.plot(iterations['T2'], design_var1['T2'], marker='o', color='blue',
             label=r'Trial 2 $r_{LEO}$')
    plt.plot(iterations['T3'], design_var1['T3'], marker='o', color='green',
             label=r'Trial 3 $r_{LEO}$')

    # r_LMO
    plt.plot(iterations['T1'], design_var2['T1'], marker='o', color='darkred',
             label=r'Trial 1 $r_{LMO}$')
    plt.plot(iterations['T2'], design_var2['T2'], marker='o', color='darkblue',
             label=r'Trial 2 $r_{LMO}$')
    plt.plot(iterations['T3'], design_var2['T3'], marker='o',
             color='darkgreen', label=r'Trial 3 $r_{LMO}$')

    plt.xlabel('Iteration')
    plt.ylabel('Design Variable Value (km)')
    plt.title('Design Variable Convergence vs. Iteration')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("design_var_converge.png")
