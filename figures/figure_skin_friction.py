from copy import copy
import numpy as np
from numpy.typing import NDArray
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import texfig

from create_dataframe import NUM_INTERP_POINTS

VMIN = -0.15
VMAX = 0.15
ASPECT_RATIO = 0.5


@matplotlib.ticker.FuncFormatter
def remove_pointless_zeros_major_formatter(x, pos):
    label = "{:.2f}".format(0 if round(x, 2) == 0 else x).rstrip("0").rstrip(".")
    return label


def make_single_subplot(
    ax: plt.Axes,
    x: NDArray,
    y: NDArray,
    z: NDArray,
    levels: NDArray,
    norm: matplotlib.colors.TwoSlopeNorm,
    cmap: matplotlib.colors.LinearSegmentedColormap,
):
    reference = ax.contourf(
        x, y, z, levels=levels, norm=norm, cmap=cmap, extend="both", zorder=-20
    )
    ax.set_rasterization_zorder(-10)
    ax.set_xlim(0.0, 1.0)
    return reference


def make_fancy_axis(axes: plt.Axes):
    for ax in axes:
        ax.set_xlabel(r"$x/c$", labelpad=1.0)
        ax.xaxis.set_tick_params(width=0.5, direction="in")
        ax.yaxis.set_tick_params(width=0.5, direction="in")
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(remove_pointless_zeros_major_formatter)
        )

        ax2 = copy(ax)
        ax2 = ax.secondary_yaxis("right")
        ax2.tick_params(
            axis="y", direction="in", which="both", width=0.5, labelright=False
        )

    axes[0].set_ylabel("Simulation time", labelpad=3.5)
    axes[1].axes.get_yaxis().set_ticklabels([])


def add_colorbar(
    fig: matplotlib.figure.Figure,
    ax: plt.Axes,
    figure_reference: matplotlib.contour.QuadContourSet,
    vmin: float,
    vmax: float,
):
    axcb = inset_axes(
        ax,
        width="2%",  # width = % of parent_bbox width
        height="100%",  # height = % of parent_bbox width
        loc="upper right",
        bbox_to_anchor=(0.06, 0.07, 1, 0.8),  # (x, y, width, height)
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    axcb.tick_params(direction="in", width=0.5)
    cbar = fig.colorbar(
        figure_reference, cax=axcb, ticks=[vmin, 0.00, vmax], extendfrac=0
    )
    cbar.set_label(
        r"$C_f$",
        rotation="horizontal",
        labelpad=7,
        horizontalalignment="center",
        verticalalignment="center",
    )
    axcb.yaxis.set_label_coords(1, 1.1)
    # Remove pointless zeros from ticks
    axcb.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(remove_pointless_zeros_major_formatter)
    )


def make_figure(
    true_labels: NDArray,
    x_coord_suction_side_original: NDArray,
    predicted_labels: NDArray,
    x_coord_suction_side_interpolated: NDArray,
    times: NDArray,
) -> None:
    plt.close("all")
    fig, axes = texfig.subplots(ratio=ASPECT_RATIO, nrows=1, ncols=2)

    x_grid_original, y_grid_original = np.meshgrid(x_coord_suction_side_original, times)
    x_grid_interpolated, y_grid_interpolated = np.meshgrid(
        x_coord_suction_side_interpolated, times
    )

    levels = np.linspace(VMIN, VMAX, 400)
    norm = matplotlib.colors.TwoSlopeNorm(vmax=VMAX, vmin=VMIN, vcenter=0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "my_cmap", [(0, "royalblue"), (0.5, "white"), (1, "firebrick")], N=256
    )

    make_single_subplot(
        axes[0], x_grid_original, y_grid_original, true_labels, levels, norm, cmap
    )
    ref = make_single_subplot(
        axes[1],
        x_grid_interpolated,
        y_grid_interpolated,
        predicted_labels,
        levels,
        norm,
        cmap,
    )

    add_colorbar(fig, axes[1], ref, VMIN, VMAX)

    fig.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.2, wspace=0.1)

    make_fancy_axis(axes)

    output_filename = "Fig_Cf"
    texfig.savefig(output_filename, dpi=1000, bbox_inches="tight", pad_inches=0)

    plt.close("all")


def main() -> None:
    x_coord_suction_side_original = np.fromfile(
        "./trained_models/dataset/M02_Re60k_span01_g480_heave_ramp_k005/x_coord_suction_side.npy",
        dtype=np.float64,
    )

    x_coord_suction_side_interpolated = np.linspace(0.0, 1.0, num=NUM_INTERP_POINTS)
    # remove first point
    x_coord_suction_side_interpolated = x_coord_suction_side_interpolated[1:]

    predictions = np.fromfile(
        "./outputs/predictions_DS1_10_uv_skin_friction.npy",
        dtype=np.float32,
    ).reshape([-1, len(x_coord_suction_side_interpolated)])

    targets = np.fromfile(
        "./trained_models/dataset/M02_Re60k_span01_g480_heave_ramp_k005/skin_friction_distribution.npy",
        dtype=np.float64,
    ).reshape([-1, len(x_coord_suction_side_original)])

    times = np.fromfile("./outputs/times_10.npy", dtype=np.float64)

    make_figure(
        targets,
        x_coord_suction_side_original,
        predictions,
        x_coord_suction_side_interpolated,
        times,
    )


if __name__ == "__main__":
    main()
