import numpy as np
import openmdao.api as om


class TransferOrbitComp(om.ExplicitComponent):
    """
    Computes key parameters of a heliocentric Hohmann transfer from Earth's
    orbit to Mars' orbit.

    We assume:
      - r_earth (perihelion) and r_mars (aphelion) are circular orbits around the Sun.
      - mu is the gravitational parameter of the Sun (~1.32712440018e11 km^3/s^2).
      - No inclination change (pure coplanar Hohmann).
      - Outputs the velocity at Earth's distance (v_dep) on the transfer ellipse,
        the velocity at Mars' distance (v_arr) on the transfer ellipse,
        and the one-way time of flight (tof).
    """

    def setup(self):
        # Inputs: same mu, plus r1, r2
        self.add_input('mu_sun',
                       val=1.327e11,
                       desc='Gravitational parameter of the Sun',
                       units='km**3/s**2')

        # Earth's orbital radius around the Sun (perihelion of transfer)
        self.add_input('rp', val=1.496e8, units='km',
                       desc='Nominal heliocentric distance for Earth')

        # # fixed circular orbit around Earth at a radius of 7000 km
        # self.add_input("rp", val=7000.0, units="km")

        # Mars' orbital radius around the Sun (aphelion of transfer)
        self.add_input('ra', val=2.279e8, units='km',
                       desc='Nominal heliocentric distance for Mars')

        # # fixed circular orbit around Mars at a radius of 4400 km
        # self.add_input('ra', val=225004400, desc='apoapsis radius', units='km')

        # Outputs: velocity at periapsis (vp), velocity at apoapsis (va)
        # time_of_flight
        # velocity at Earth distance on the transfer ellipse
        self.add_output('vp', val=0.0, units='km/s',
                        desc='Transfer-ellipse velocity at Earth distance')
        # self.add_output("vp", val=0.0, units="km/s")

        # velocity at Mars distance on the transfer ellipse
        self.add_output('va', val=0.0, units='km/s',
                        desc='Transfer-ellipse velocity at Mars distance')
        # self.add_output("va", val=0.0, units="km/s")

        self.add_output('tof', val=0.0, units='s',
                        desc='Hohmann time of flight (half the ellipse)')

        # Finite difference partials (for now)
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        mu_sun = inputs["mu_sun"]
        rp = inputs["rp"]
        ra = inputs["ra"]

        # Semi-major axis
        a = 0.5 * (rp + ra)

        outputs['vp'] = np.sqrt(mu_sun * (2.0/rp - 1.0/a))
        outputs['va'] = np.sqrt(mu_sun * (2.0/ra - 1.0/a))
        outputs['tof'] = np.pi * np.sqrt(a**3 / mu_sun)
        # # Eccentricity
        # e = (ra - rp) / (ra + rp)

        # p = a * (1.0 - e**2)
        # h = np.sqrt(mu_sun * p)

        # # Periapsis velocity of transfer ellipse
        # # v_p = sqrt(mu * (2/r1 - 1/a))
        # outputs["vp"] = h / rp

        # # Apoapsis velocity of transfer ellipse
        # # v_a = sqrt(mu * (2/r2 - 1/a))
        # outputs["va"] = h / ra

        # # Hohmann time of flight = pi * sqrt(a^3 / mu)
        # # (One-way transfer)
        # outputs["tof"] = np.pi * np.sqrt(a**3 / mu)
