import os
import numpy as np
from numpy.typing import NDArray

from metrics import Metrics
from file_searching import search_for_files
from read_flow_cgns import read_flow_in_cgns
from read_grid_cgns import read_grid_in_cgns
from airfoil_sides import get_airfoil_suction_and_pressure_side_indices
from flow_forces import compute_wall_pressure_forces, compute_wall_viscous_forces
from aerodynamic_coefficients import (
    compute_lift_and_drag_coefficients,
    compute_pitch_moment_coefficient,
    compute_pressure_coefficient,
    compute_skin_friction_coefficient,
    compute_circulation,
)
from params import Simulation, SIMULATIONS, STATIC_AOA_IN_DEG, PIVOT_POINT
from rotate_velocity import rotate_velocity


GAMMA = 1.4
REFERENCE_DENSITY = 1.0
STATIC_PRESSURE = 1.0 / GAMMA

FLOW_FILE_PATTERN = "qout2Davg*.cgns"
BASE_PATH_TO_SAVE_DATA = (
    "/home/miotto/Desktop/CNN_PyTorch_coeffs_results/trained_models/dataset"
)


def main(simulation: Simulation, output_path: str):
    path = simulation.path
    qout_files = search_for_files(path, pattern=FLOW_FILE_PATTERN)
    grid_file = search_for_files(path, pattern="grid2D.cgns")[0]

    x, y = read_grid_in_cgns(grid_file)

    metrics = Metrics(x, y)

    suction_side_indices, _ = get_airfoil_suction_and_pressure_side_indices(
        x[:, 0], y[:, 0], STATIC_AOA_IN_DEG, metrics
    )

    straight_airfoil_x = (
        np.cos(np.deg2rad(STATIC_AOA_IN_DEG)) * x
        - np.sin(np.deg2rad(STATIC_AOA_IN_DEG)) * y
    )

    x_coord_suction_side = straight_airfoil_x[suction_side_indices, 0]

    # straight_airfoil_y = (
    #     np.sin(np.deg2rad(STATIC_AOA_IN_DEG)) * x
    #     + np.cos(np.deg2rad(STATIC_AOA_IN_DEG)) * y
    # )

    # import matplotlib.pyplot as plt

    # plt.plot(x[:, 0], y[:, 0], "-o", color="b")
    # plt.plot(straight_airfoil_x[:, 0], straight_airfoil_y[:, 0], "-o", color="r")
    # plt.show()

    reference_viscosity = 1.0 / simulation.reynolds

    lift = []
    drag = []
    pitch_moment = []
    timestamps = []
    circulation = []
    wall_pressure_distribution = []
    skin_friction_distribution = []

    for index_qout, qout_file in enumerate(qout_files):
        print(
            f"{index_qout + 1}/{len(qout_files)} -- {os.path.split(output_path)[1]} / {os.path.split(qout_file)[1]}"
        )

        q_vector, time = read_flow_in_cgns(qout_file)

        velocity_x = q_vector[1] / q_vector[0]
        velocity_y = q_vector[2] / q_vector[0]
        pressure = q_vector[3]

        if simulation.rotate.apply:
            velocity_x, velocity_y = rotate_velocity(
                velocity_x, velocity_y, simulation, time
            )

        wall_pressure_forces = compute_wall_pressure_forces(pressure, metrics)
        wall_viscous_forces = compute_wall_viscous_forces(
            velocity_x, velocity_y, reference_viscosity, metrics
        )

        wall_pressure_forces = np.array(wall_pressure_forces, dtype=NDArray)
        wall_viscous_forces = np.array(wall_viscous_forces, dtype=NDArray)

        instantaneous_lift, instantaneous_drag = compute_lift_and_drag_coefficients(
            wall_pressure_forces,
            wall_viscous_forces,
            REFERENCE_DENSITY,
            simulation.mach,
        )

        instantaneous_pitch_moment = compute_pitch_moment_coefficient(
            wall_pressure_forces,
            wall_viscous_forces,
            REFERENCE_DENSITY,
            simulation.mach,
            metrics,
            PIVOT_POINT,
        )

        instantaneous_circulation = compute_circulation(
            velocity_x, velocity_y, 330, metrics
        )

        instantaneous_wall_pressure_distribution = compute_pressure_coefficient(
            pressure[:, 0], STATIC_PRESSURE, REFERENCE_DENSITY, simulation.mach
        )[suction_side_indices]

        instantaneous_skin_friction_distribution = compute_skin_friction_coefficient(
            velocity_x,
            velocity_y,
            REFERENCE_DENSITY,
            reference_viscosity,
            simulation.mach,
            metrics,
        )[suction_side_indices]

        lift.append(instantaneous_lift)
        drag.append(instantaneous_drag)
        pitch_moment.append(instantaneous_pitch_moment)
        timestamps.append(time)
        circulation.append(instantaneous_circulation)
        wall_pressure_distribution.append(instantaneous_wall_pressure_distribution)
        skin_friction_distribution.append(instantaneous_skin_friction_distribution)

    lift = np.array(lift)
    drag = np.array(drag)
    pitch_moment = np.array(pitch_moment)
    timestamps = np.array(timestamps)
    circulation = np.array(circulation)
    wall_pressure_distribution = np.array(wall_pressure_distribution)
    skin_friction_distribution = np.array(skin_friction_distribution)

    # Create timestamps based on simulation timestep
    # start_time = 0
    # timestamps = start_time + np.arange(len(qout_files)) * SIMULATION_DT

    lift.tofile(os.path.join(output_path, "coeff_lift.npy"))
    drag.tofile(os.path.join(output_path, "coeff_drag.npy"))
    pitch_moment.tofile(os.path.join(output_path, "coeff_moment.npy"))
    timestamps.tofile(os.path.join(output_path, "times.npy"))
    circulation.tofile(os.path.join(output_path, "circulation.npy"))
    wall_pressure_distribution.tofile(
        os.path.join(output_path, "wall_pressure_distribution.npy")
    )
    skin_friction_distribution.tofile(
        os.path.join(output_path, "skin_friction_distribution.npy")
    )
    x_coord_suction_side.tofile(os.path.join(output_path, "x_coord_suction_side.npy"))


def loop_through_simulations():
    for simulation in SIMULATIONS:

        output_path = os.path.join(
            BASE_PATH_TO_SAVE_DATA, simulation.output_folder_name
        )

        folder_exist = os.path.exists(output_path)
        if not folder_exist:
            os.makedirs(output_path)

        main(simulation, output_path)


if __name__ == "__main__":
    loop_through_simulations()
