import numpy as np
from numpy.typing import NDArray
from params import Simulation, Rotation


def position_sine(time, reduced_frequency, amplitude=0.5):
    """
    Both `t` and `rate` must be nondimensionalized by the freestream vel.
    `static_aoa` must be in [rad]
    """
    strouhal = reduced_frequency * amplitude / np.pi
    velocity = 2.0 * np.pi * strouhal * np.cos(2.0 * reduced_frequency * time)
    aoa = np.arctan(-velocity)

    return aoa


def position_cosine(time, reduced_frequency, amplitude=0.5):
    """
    Both `t` and `rate` must be nondimensionalized by the freestream vel.
    `static_aoa` must be in [rad]
    """
    strouhal = reduced_frequency * amplitude / np.pi
    velocity = -2.0 * np.pi * strouhal * np.sin(2.0 * reduced_frequency * time)
    aoa = np.arctan(-velocity)

    return aoa


def plunging_cte_rate_smooth(time, rate, mach):
    """
    Both `t` and `rate` must be nondimensionalized by the freestream vel.
    `rate` must be in [rad/s]
    """
    factor = 9.2 * mach
    rate = rate * mach
    time = time / mach
    dhdt = -np.tan(rate * time) * (1.0 - np.exp(-factor * time))
    aoa = np.arctan(-dhdt)

    return aoa


def pitching_cte_rate_smooth(time, rate, mach):
    """
    Both `t` and `rate` must be nondimensionalized by the freestream vel.
    `rate` must be in [rad/s]
    """
    factor = 9.2 * mach
    rate = rate * mach
    time = time / mach
    aoa = rate * (time + (1.0 / factor) * (np.exp(-factor * time) - 1.0))

    return aoa


def pitching_cte_rate_smooth_no_mach(time, rate, mach):
    """
    Both `t` and `rate` must be nondimensionalized by the freestream vel.
    `rate` must be in [rad/s]
    """
    factor = 9.2
    rate = rate * mach
    time = time / mach
    aoa = rate * (time + (1.0 / factor) * (np.exp(-factor * time) - 1.0))

    return aoa


def rotate_velocity(
    velocity_x: NDArray,
    velocity_y: NDArray,
    simulation: Simulation,
    time: float,
):
    rate_or_freq = simulation.rotate.rate_or_frequency

    if simulation.rotate.motion_type == "position_sine":
        aoa = position_sine(time, rate_or_freq)
    elif simulation.rotate.motion_type == "position_cosine":
        aoa = position_cosine(time, rate_or_freq)
    elif simulation.rotate.motion_type == "pitch_ramp":
        aoa = pitching_cte_rate_smooth(time, rate_or_freq, simulation.mach)
    elif simulation.rotate.motion_type == "pitch_ramp_no_mach":
        aoa = pitching_cte_rate_smooth_no_mach(time, rate_or_freq, simulation.mach)
    elif simulation.rotate.motion_type == "plunge_ramp":
        aoa = plunging_cte_rate_smooth(time, rate_or_freq, simulation.mach)
    else:
        raise ValueError("Invalid motion type")

    rotated_velocity_x = velocity_x * np.cos(aoa) - velocity_y * np.sin(aoa)
    rotated_velocity_y = velocity_x * np.sin(aoa) + velocity_y * np.cos(aoa)

    return rotated_velocity_x, rotated_velocity_y


def _test():
    simulation = Simulation(
        path="dummy_path",
        output_folder_name="dummy_name",
        mach=0.1,
        reynolds=6e4,
        rotate=Rotation(apply=True, motion_type="pitch_ramp", rate_or_frequency=0.10),
    )

    velocity_x = 1.0
    velocity_y = 0.0

    time = 7.0

    rotated_vel_x, rotated_vel_y = rotate_velocity(
        velocity_x, velocity_y, simulation, time
    )

    print(rotated_vel_x)
    print(rotated_vel_y)


if __name__ == "__main__":
    _test()
