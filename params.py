from typing import Literal
from dataclasses import dataclass
import numpy as np


PIVOT_POINT = [0.25 * np.cos(np.deg2rad(8)), -0.25 * np.sin(np.deg2rad(8))]
STATIC_AOA_IN_DEG = 8


@dataclass
class Rotation:
    apply: bool
    motion_type: Literal[
        "position_sine",
        "position_cosine",
        "pitch_ramp",
        "pitch_ramp_no_mach",
        "plunge_ramp",
    ]
    rate_or_frequency: float


@dataclass
class Simulation:
    path: str
    output_folder_name: str
    mach: float
    reynolds: float
    rotate: Rotation


simulation1 = Simulation(
    path="/media/miotto/Seagate Backup Plus Drive/SD7003/Brener_M01_k025_span04/output",
    output_folder_name="M01_Re60k_span04_g480_heave_periodic_k025",
    reynolds=6e4,
    mach=0.1,
    rotate=Rotation(
        apply=True,
        motion_type="position_sine",
        rate_or_frequency=0.25,
    ),
)

simulation2 = Simulation(
    path="/media/miotto/3B712DB11C683E49/SD7003/Brener_M02_k025_span04/proc/output",
    output_folder_name="M02_Re60k_span04_g480_heave_periodic_k025",
    reynolds=6e4,
    mach=0.2,
    rotate=Rotation(
        apply=True,
        motion_type="position_cosine",
        rate_or_frequency=0.25,
    ),
)

simulation3 = Simulation(
    path="/media/miotto/3B712DB11C683E49/SD7003/Brener_M04_k025_span04/proc/output",
    output_folder_name="M04_Re60k_span04_g480_heave_periodic_k025",
    reynolds=6e4,
    mach=0.4,
    rotate=Rotation(
        apply=True,
        motion_type="position_sine",
        rate_or_frequency=0.25,
    ),
)

simulation4 = Simulation(
    path="/media/miotto/Backup Plus/SD7003/Brener_M01_k050_span04/output",
    output_folder_name="M01_Re60k_span04_g480_heave_periodic_k050",
    reynolds=6e4,
    mach=0.1,
    rotate=Rotation(
        apply=True,
        motion_type="position_sine",
        rate_or_frequency=0.5,
    ),
)

simulation5 = Simulation(
    path="/media/miotto/Backup Plus/SD7003/Brener_M04_k050_span04/proc/output",
    output_folder_name="M04_Re60k_span04_g480_heave_periodic_k050",
    reynolds=6e4,
    mach=0.4,
    rotate=Rotation(
        apply=True,
        motion_type="position_sine",
        rate_or_frequency=0.5,
    ),
)

simulation6 = Simulation(
    path="/media/miotto/3B712DB11C683E49/SD7003/M01_Re60k_span01_g480/proc_pitch_ramp_k005/output",
    output_folder_name="M01_Re60k_span01_g480_pitch_ramp_k005",
    reynolds=6e4,
    mach=0.1,
    rotate=Rotation(
        apply=False,
        motion_type="pitch_ramp",
        rate_or_frequency=0.05,
    ),
)

simulation7 = Simulation(
    path="/media/miotto/Seagate Backup Plus Drive/SD7003/M01_Re60k_span01_g480/proc_pitch_ramp_k010/output",
    output_folder_name="M01_Re60k_span01_g480_pitch_ramp_k010",
    reynolds=6e4,
    mach=0.1,
    rotate=Rotation(
        apply=False,
        motion_type="pitch_ramp",
        rate_or_frequency=0.10,
    ),
)

simulation8 = Simulation(
    path="/media/miotto/Seagate Backup Plus Drive/SD7003/M01_Re60k_span01_g480/proc_heave_ramp_k005/output",
    output_folder_name="M01_Re60k_span01_g480_heave_ramp_k005",
    reynolds=6e4,
    mach=0.1,
    rotate=Rotation(
        apply=True,
        motion_type="plunge_ramp",
        rate_or_frequency=0.05,
    ),
)

simulation9 = Simulation(
    path="/media/miotto/Seagate Backup Plus Drive/SD7003/M01_Re60k_span01_g480/proc_heave_ramp_k010/output",
    output_folder_name="M01_Re60k_span01_g480_heave_ramp_k010",
    reynolds=6e4,
    mach=0.1,
    rotate=Rotation(
        apply=True,
        motion_type="plunge_ramp",
        rate_or_frequency=0.10,
    ),
)

simulation10 = Simulation(
    path="/media/miotto/3B712DB11C683E49/SD7003/M02_Re60k_span01_g480/proc_heave_ramp_k005/output",
    output_folder_name="M02_Re60k_span01_g480_heave_ramp_k005",
    reynolds=6e4,
    mach=0.2,
    rotate=Rotation(
        apply=True,
        motion_type="plunge_ramp",
        rate_or_frequency=0.05,
    ),
)

simulation11 = Simulation(
    path="/media/miotto/Backup Plus1/SD7003/M02_Re60k_span01_g480/proc_heave_ramp_k010/output",
    output_folder_name="M02_Re60k_span01_g480_heave_ramp_k010",
    reynolds=6e4,
    mach=0.2,
    rotate=Rotation(
        apply=True,
        motion_type="plunge_ramp",
        rate_or_frequency=0.10,
    ),
)

simulation12 = Simulation(
    path="/media/miotto/Backup Plus1/SD7003/M02_Re60k_span01_g480/proc_pitch_ramp_k010/output",
    output_folder_name="M02_Re60k_span01_g480_pitch_ramp_k010",
    reynolds=6e4,
    mach=0.2,
    rotate=Rotation(
        apply=False,
        motion_type="pitch_ramp_no_mach",
        rate_or_frequency=0.10,
    ),
)

simulation13 = Simulation(
    path="/media/miotto/3B712DB11C683E49/SD7003/M04_Re60k_span01_g480/proc_pitch_ramp_k005/output",
    output_folder_name="M04_Re60k_span01_g480_pitch_ramp_k005",
    reynolds=6e4,
    mach=0.4,
    rotate=Rotation(
        apply=False,
        motion_type="pitch_ramp",
        rate_or_frequency=0.05,
    ),
)

simulation14 = Simulation(
    path="/media/miotto/Seagate Backup Plus Drive/SD7003/M04_Re60k_span01_g480/proc_pitch_ramp_k010/output",
    output_folder_name="M04_Re60k_span01_g480_pitch_ramp_k010",
    reynolds=6e4,
    mach=0.4,
    rotate=Rotation(
        apply=False,
        motion_type="pitch_ramp",
        rate_or_frequency=0.10,
    ),
)

simulation15 = Simulation(
    path="/media/miotto/3B712DB11C683E49/SD7003/M04_Re60k_span01_g480/proc_heave_ramp_k005/output",
    output_folder_name="M04_Re60k_span01_g480_heave_ramp_k005",
    reynolds=6e4,
    mach=0.4,
    rotate=Rotation(
        apply=True,
        motion_type="plunge_ramp",
        rate_or_frequency=0.05,
    ),
)

simulation16 = Simulation(
    path="/media/miotto/Seagate Backup Plus Drive/SD7003/M04_Re60k_span01_g480/proc_heave_ramp_k010/output",
    output_folder_name="M04_Re60k_span01_g480_heave_ramp_k010",
    reynolds=6e4,
    mach=0.4,
    rotate=Rotation(
        apply=True,
        motion_type="plunge_ramp",
        rate_or_frequency=0.10,
    ),
)

simulation17 = Simulation(
    path="/media/miotto/Backup Plus/SD7003/M01_Re200k_g720/proc_pitch_ramp_k005/output",
    output_folder_name="M01_Re200k_span01_g720_pitch_ramp_k005",
    reynolds=2e5,
    mach=0.1,
    rotate=Rotation(
        apply=False,
        motion_type="pitch_ramp",
        rate_or_frequency=0.05,
    ),
)

simulation18 = Simulation(
    path="/media/miotto/Backup Plus/SD7003/M01_Re200k_g720/proc_pitch_ramp_k010/output",
    output_folder_name="M01_Re200k_span01_g720_pitch_ramp_k010",
    reynolds=2e5,
    mach=0.1,
    rotate=Rotation(
        apply=False,
        motion_type="pitch_ramp",
        rate_or_frequency=0.10,
    ),
)

simulation19 = Simulation(
    path="/media/miotto/3B712DB11C683E49/SD7003/M01_Re200k_span01_g720/proc_heave_ramp_k005/output",
    output_folder_name="M01_Re200k_span01_g720_heave_ramp_k005",
    reynolds=2e5,
    mach=0.1,
    rotate=Rotation(
        apply=True,
        motion_type="plunge_ramp",
        rate_or_frequency=0.05,
    ),
)

simulation20 = Simulation(
    path="/media/miotto/Backup Plus/SD7003/M01_Re200k_g720/proc_heave_ramp_k010/output",
    output_folder_name="M01_Re200k_span01_g720_heave_ramp_k010",
    reynolds=2e5,
    mach=0.1,
    rotate=Rotation(
        apply=True,
        motion_type="plunge_ramp",
        rate_or_frequency=0.10,
    ),
)

SIMULATIONS = [
    # simulation1,
    simulation2,
    # simulation3,
    # simulation4,
    # simulation5,
    # simulation6,
    # simulation7,
    # simulation8,
    # simulation9,
    # simulation10,
    # simulation11,
    # simulation12,
    # simulation13,
    # simulation14,
    # simulation15,
    # simulation16,
    # simulation17,
    # simulation18,
    # simulation19,
    # simulation20,
]
