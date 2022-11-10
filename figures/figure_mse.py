"""
For each simulation (case), I inferred the model's performance trained on different
datasets, from 1 to 5. This script plots the Mean Squared Error of the network
predictions (for all datasets) for a few selected cases.
"""
from dataclasses import dataclass
from typing import List, Union
import numpy as np
from numpy.typing import NDArray
import matplotlib
import matplotlib.pyplot as plt
import texfig


INDEX_OF_SELECTED_CASES = [1, 5, 8, 10, 13, 17, 18, 19, 20]
TRUNCATE_INDEX_CASES = [None, None, None, None, None, 695, None, 326, None]

INDEX_OF_SELECTED_CASES = [1, 3, 5, 8, 10, 13, 15, 17, 19]
TRUNCATE_INDEX_CASES = [None, None, None, None, None, None, None, 1390, None]

INDEX_OF_DATASETS = [1, 2, 3, 4, 5]
DATASET_COLORS = ["#0000FF", "#00FFFF", "#008080", "#808080", "#000000"]

NROWS = 3
NCOLS = 3

FIGURE_RATIO = 0.6
Y_MAX, Y_MIN = [10, 10 ** (-5)]


@dataclass
class SimulationCaseInfo:
    simulation_case: int
    truncation_index: int
    dataset_indices: List[int]
    dataset_color: List[str]
    mean_squared_errors: List[NDArray]


def read_mean_squared_error_data() -> List[SimulationCaseInfo]:

    mean_squared_error_data = []

    for i, case in enumerate(INDEX_OF_SELECTED_CASES):
        info = SimulationCaseInfo(
            simulation_case=case,
            truncation_index=TRUNCATE_INDEX_CASES[i],
            dataset_indices=[],
            dataset_color=[],
            mean_squared_errors=[],
        )
        for j, dataset in enumerate(INDEX_OF_DATASETS):
            mse = np.fromfile(f"./outputs/mse_DS{dataset}_{case}.npy", dtype=np.float32)
            info.dataset_indices.append(dataset)
            info.dataset_color.append(DATASET_COLORS[j])
            info.mean_squared_errors.append(mse)

        mean_squared_error_data.append(info)

    return mean_squared_error_data


def make_single_plot(ax: plt.Axes, y, color, label: str, truncate_at: Union[int, None]):
    (reference,) = ax.plot(
        y[:truncate_at],
        color=color,
        label=label,
    )
    ax.set_ylim(Y_MIN, Y_MAX)

    return reference


def make_subplot(ax: plt.Axes, case: SimulationCaseInfo):
    for i, dataset_index in enumerate(case.dataset_indices):
        reference = make_single_plot(
            ax,
            case.mean_squared_errors[i],
            color=case.dataset_color[i],
            truncate_at=case.truncation_index,
            label=dataset_index,
        )
        ax.text(
            0.5,
            1.05,
            f"Simulation \# {case.simulation_case}",
            size=10,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            zorder=20,
        )

    return reference


def make_fancy_axis(
    ax: plt.Axes, should_add_x_label: bool, should_add_y_label: bool
) -> None:
    if should_add_x_label:
        ax.set_xlabel("Simulation time", rotation="horizontal", labelpad=1.0)
    if should_add_y_label:
        ax.set_ylabel("MSE", rotation="horizontal", labelpad=1.0)
        ax.yaxis.set_label_coords(0.05, 1.1, transform=ax.transAxes)
    ax.axes.get_xaxis().set_ticklabels([])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
    ax.yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(numticks=99, base=10.0, subs="auto")
    )
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([])

    texfig.make_fancy_axis(ax)


def main() -> None:

    mean_squared_error_data = read_mean_squared_error_data()

    assert len(mean_squared_error_data) == NROWS * NCOLS

    plt.close("all")
    fig, ax = texfig.subplots(ratio=FIGURE_RATIO, nrows=NROWS, ncols=NCOLS)

    index = 0
    for row in range(NROWS):
        for col in range(NCOLS):
            case_dto = mean_squared_error_data[index]
            should_add_x_label = row == NROWS - 1
            should_add_y_label = row == 0

            make_subplot(ax[row, col], case_dto)
            make_fancy_axis(ax[row, col], should_add_x_label, should_add_y_label)

            index += 1

    fig.subplots_adjust(
        left=0.03, right=0.99, top=0.92, bottom=0.2, hspace=0.25, wspace=0.25
    )

    texfig.savefig("mse", dpi=1000, bbox_inches="tight", pad_inches=0)
    plt.close("all")


if __name__ == "__main__":
    main()
