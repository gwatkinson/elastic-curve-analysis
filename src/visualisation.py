import matplotlib.pyplot as plt


def plot_curve(curve, color="b", ax=None, **kwargs):
    """Plot a discrete curve."""
    ax = ax or plt.gca()
    ccurve = list(curve.copy())
    ccurve.append(curve[0])
    ax.plot(curve[:, 0], curve[:, 1], color=color, **kwargs)
    ax.axis("equal")
    ax.axis("off")
    return ax


def plot_curve_with_info(plant_dict, color="b"):
    """Plot a discrete curve."""
    curve = plant_dict["curve"]
    type_of_plant = plant_dict["type"]
    plt.plot(curve[:, 0], curve[:, 1], color=color)
    plt.title(f"Plant type: {type_of_plant}")
    plt.axis("equal")
    plt.axis("off")


def plot_sample_with_info(
    ds, indexes, n_rows=None, n_cols=None, figsize=(20, 5), show_id=True, color="b"
):
    """Plot a discrete curve."""
    plant_dicts = [ds[i] for i in indexes]
    n_cols = n_cols or len(plant_dicts)
    n_rows = n_rows or len(plant_dicts) // n_cols + 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, plant_dict in enumerate(plant_dicts):
        row, col = divmod(i, n_cols)
        curve = plant_dict["curve"]
        type_of_plant = plant_dict["type"]
        ax = axs[row, col]
        ax.plot(curve[:, 0], curve[:, 1], color=color)
        ax.axis("off")
        ax.axis("equal")
        index_info = f"(Index: {indexes[i]})" if show_id else None
        ax.set_title(f"Plant type: {type_of_plant} {index_info}")

    plt.axis("off")
    plt.axis("equal")
    fig.tight_layout()
    return fig


def plots_geodesics(geod_points, out_files):
    fig = plt.figure(figsize=(10, 2))
    plt.title("Geodesic between two cells")
    plt.axis("off")
    n_times = len(geod_points)
    for i, curve in enumerate(geod_points):
        fig.add_subplot(2, n_times // 2, i + 1)
        if i == 0:
            plt.plot(curve[:, 0], curve[:, 1], "b")
        elif i == n_times - 1:
            plt.plot(curve[:, 0], curve[:, 1], "r")
        else:
            plt.plot(curve[:, 0], curve[:, 1], "orange")
        plt.axis("equal")
        plt.axis("off")
    for file in out_files:
        plt.savefig(file)
    return fig


def plot_overlayed_geodesics(geod_points, out_files):
    fig = plt.figure(figsize=(12, 12))
    sub_geod_points = geod_points[::2]
    for i in range(1, len(sub_geod_points) - 1):
        plt.plot(
            sub_geod_points[i, :, 0], sub_geod_points[i, :, 1], "o-", color="lightgrey"
        )
    plt.plot(
        sub_geod_points[0, :, 0], sub_geod_points[0, :, 1], "o-b", label="Start Cell"
    )
    plt.plot(
        sub_geod_points[-1, :, 0], sub_geod_points[-1, :, 1], "o-r", label="End Cell"
    )

    # plt.title("Geodesic for the Square Root Velocity metric")
    # plt.legend()

    for file in out_files:
        plt.savefig(file)

    return fig
