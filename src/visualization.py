
# ==========================================================
# Imports
# ==========================================================

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.ticker import PercentFormatter

# ==========================================================
# Visual Identity
# ==========================================================
# Centralized visual configuration for all project figures.
# This module defines the project's visual language and should
# not contain any plotting logic.
# ==========================================================

# ==========================================================
# General Theme
# ==========================================================

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font="DejaVu Sans",
    font_scale=1.0
)

# ==========================================================
# Color Palette
# ==========================================================

COLORS = {

    # ------------------------------------------------------
    # Primary colors
    # ------------------------------------------------------

    "primary": "#1F4E79",        # Navy Blue
    "secondary": "#5B84B1",      # Steel Blue

    # ------------------------------------------------------
    # Neutral colors
    # ------------------------------------------------------

    "background": "#FFFFFF",
    "gray": "#C7CDD4",
    "dark_gray": "#4D4D4D",
    "grid": "#E8E8E8",

    # ------------------------------------------------------
    # Semantic colors
    # ------------------------------------------------------

    "success": "#2E8B57",
    "warning": "#F4A259",
    "danger": "#C0392B",

    # ------------------------------------------------------
    # Credit Risk colors
    # ------------------------------------------------------

    "low_risk": "#D6EAF8",
    "medium_risk": "#5DADE2",
    "high_risk": "#154360"
}

# ==========================================================
# Sequential Palettes
# ==========================================================

RISK_PALETTE = sns.color_palette(
    [
        "#D6EAF8",
        "#AED6F1",
        "#5DADE2",
        "#2874A6",
        "#154360"
    ]
)

BLUE_PALETTE = sns.color_palette(
    "Blues",
    6
)

GRAY_PALETTE = sns.color_palette(
    "Greys",
    6
)

# ==========================================================
# Matplotlib Defaults
# ==========================================================

mpl.rcParams.update({

    # ------------------------------------------------------
    # Figure
    # ------------------------------------------------------

    "figure.figsize": (8, 5),
    "figure.dpi": 140,
    "figure.facecolor": COLORS["background"],

    # ------------------------------------------------------
    # Axes
    # ------------------------------------------------------

    "axes.facecolor": COLORS["background"],
    "axes.edgecolor": COLORS["gray"],
    "axes.labelcolor": COLORS["dark_gray"],

    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,

    # ------------------------------------------------------
    # Grid
    # ------------------------------------------------------

    "grid.color": COLORS["grid"],
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.45,

    # ------------------------------------------------------
    # Ticks
    # ------------------------------------------------------

    "xtick.color": COLORS["dark_gray"],
    "ytick.color": COLORS["dark_gray"],

    # ------------------------------------------------------
    # Legend
    # ------------------------------------------------------

    "legend.frameon": False,
    "legend.fontsize": 10,

    # ------------------------------------------------------
    # Saving
    # ------------------------------------------------------

    "savefig.bbox": "tight",
    "savefig.facecolor": COLORS["background"]
})

# ==========================================================
# Default Seaborn Palette
# ==========================================================

sns.set_palette([
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["success"],
    COLORS["warning"],
    COLORS["danger"]
])

# ==========================================================
# Default Figure Sizes
# ==========================================================

FIGURE_SIZE = {
    "small": (6, 4),
    "medium": (8, 5),
    "wide": (12, 5),
    "dashboard": (14, 8)
}

# ==========================================================
# Default Typography
# ==========================================================

FONT = {
    "title": 14,
    "subtitle": 12,
    "label": 11,
    "tick": 10,
    "annotation": 9
}

# ==========================================================
# Figure Factory
# ==========================================================

def create_figure(
    n_rows=1,
    n_cols=1,
    figsize=None,
    title=None,
    sharex=False,
    sharey=False
):
    """
    Create a standardized matplotlib figure.

    Parameters
    ----------
    n_rows : int
        Number of subplot rows.

    n_cols : int
        Number of subplot columns.

    figsize : tuple, optional
        Figure size. If None, it is computed automatically.

    title : str, optional
        Figure title.

    sharex : bool

    sharey : bool

    Returns
    -------
    fig : matplotlib.figure.Figure

    axes : ndarray
        Always returned as a 2D numpy array.
    """

    if figsize is None:

        figsize = (
            7 * n_cols,
            4 * n_rows
        )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey
    )

    if n_rows * n_cols == 1:

        axes = np.array([[axes]])

    elif n_rows == 1:

        axes = np.array([axes])

    elif n_cols == 1:

        axes = axes.reshape(-1, 1)

    if title:

        fig.suptitle(
            title,
            fontsize=FONT["title"] + 2,
            fontweight="bold",
            color=COLORS["dark_gray"],
            y=0.99
        )

    return fig, axes

# ==========================================================
# Axis Formatter
# ==========================================================

def format_axes(
    ax,
    title=None,
    xlabel=None,
    ylabel=None,
    rotate_xticks=0,
    grid_axis="y"
):
    """
    Apply the project's visual style to an axis.
    """

    ax.set_axisbelow(True)

    ax.grid(
        axis=grid_axis,
        linestyle="--",
        linewidth=.8,
        alpha=.35
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(COLORS["gray"])
    ax.spines["bottom"].set_color(COLORS["gray"])

    if title is not None:

        ax.set_title(
            title,
            fontsize=FONT["subtitle"],
            fontweight="bold"
        )

    if xlabel is not None:

        ax.set_xlabel(
            xlabel,
            fontsize=FONT["label"]
        )

    if ylabel is not None:

        ax.set_ylabel(
            ylabel,
            fontsize=FONT["label"]
        )

    ax.tick_params(
        axis="x",
        rotation=rotate_xticks,
        labelsize=FONT["tick"]
    )

    ax.tick_params(
        axis="y",
        labelsize=FONT["tick"]
    )

    # ==========================================================
# Axis Formatter
# ==========================================================

def format_axes(
    ax,
    title=None,
    xlabel=None,
    ylabel=None,
    rotate_xticks=0,
    grid_axis="y"
):
    """
    Apply the project's visual style to an axis.
    """

    ax.set_axisbelow(True)

    ax.grid(
        axis=grid_axis,
        linestyle="--",
        linewidth=.8,
        alpha=.35
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(COLORS["gray"])
    ax.spines["bottom"].set_color(COLORS["gray"])

    if title is not None:

        ax.set_title(
            title,
            fontsize=FONT["subtitle"],
            fontweight="bold"
        )

    if xlabel is not None:

        ax.set_xlabel(
            xlabel,
            fontsize=FONT["label"]
        )

    if ylabel is not None:

        ax.set_ylabel(
            ylabel,
            fontsize=FONT["label"]
        )

    ax.tick_params(
        axis="x",
        rotation=rotate_xticks,
        labelsize=FONT["tick"]
    )

    ax.tick_params(
        axis="y",
        labelsize=FONT["tick"]
    )


# ==========================================================
# Annotate Bars
# ==========================================================

def annotate_bars(
    ax,
    fmt="{:,.0f}",
    fontsize=None,
    offset=3
):
    """
    Annotate vertical bar charts with their values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    fmt : str, default="{:,.0f}"
        String format applied to each value.

    fontsize : int, optional

    offset : int, default=3
        Vertical offset in points.
    """

    if fontsize is None:
        fontsize = FONT["annotation"]

    for patch in ax.patches:

        value = patch.get_height()

        if pd.isna(value):
            continue

        x = patch.get_x() + patch.get_width() / 2

        ax.annotate(
            text=fmt.format(value),
            xy=(x, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=COLORS["dark_gray"],
            fontweight="semibold"
        )

# ==========================================================
# Percentage Formatter
# ==========================================================

from matplotlib.ticker import PercentFormatter


def format_percentage_axis(
    ax,
    axis="y",
    decimals=0
):
    """
    Format an axis as percentages.

    Parameters
    ----------
    axis : {"x", "y"}

    decimals : int
        Number of decimal places.
    """

    formatter = PercentFormatter(
        xmax=1,
        decimals=decimals
    )

    if axis == "y":

        ax.yaxis.set_major_formatter(formatter)

    else:

        ax.xaxis.set_major_formatter(formatter)

# ==========================================================
# Zero Inflation Detection
# ==========================================================

def detect_zero_inflation(
    series,
    threshold=0.60
):
    """
    Detect whether a numerical variable is zero-inflated.

    Parameters
    ----------
    series : pandas.Series

    threshold : float, default=0.60

    Returns
    -------
    bool
    """

    series = series.dropna()

    if series.empty:
        return False

    if not pd.api.types.is_numeric_dtype(series):
        return False

    zero_ratio = (series == 0).mean()

    return zero_ratio >= threshold

# ==========================================================
# Quantile Labels
# ==========================================================

def build_quantile_labels(
    intervals,
    decimals=1
):
    """
    Build readable labels from IntervalIndex.

    Example
    -------
    Q1
    [0.0 – 15.4]
    """

    labels = []

    for i, interval in enumerate(intervals, start=1):

        left = round(interval.left, decimals)
        right = round(interval.right, decimals)

        labels.append(
            f"Q{i}\n[{left} – {right}]"
        )

    return labels

# ==========================================================
# Numeric Grouping
# ==========================================================

def prepare_numeric_groups(
    data,
    variable,
    bins=10,
    keep_missing=False,
    zero_threshold=0.60
):
    """
    Prepare a numerical variable for grouped analysis.

    Returns
    -------
    DataFrame
    """

    df = data[[variable]].copy()

    if keep_missing:

        df[variable] = (
            df[variable]
            .astype(object)
            .fillna("Missing")
        )

    else:

        df = df.dropna()

    if (
        pd.api.types.is_numeric_dtype(df[variable])
        and
        not detect_zero_inflation(
            df[variable],
            zero_threshold
        )
    ):

        groups = pd.qcut(
            df[variable],
            q=bins,
            duplicates="drop"
        )

        labels = build_quantile_labels(
            groups.cat.categories
        )

        mapping = dict(
            zip(
                groups.cat.categories,
                labels
            )
        )

        df["group"] = groups.map(mapping)

    else:

        df["group"] = df[variable].astype(str)

    return df

# ==========================================================
# Categorical Grouping
# ==========================================================

def prepare_categorical_groups(
    data,
    variable,
    keep_missing=True,
    rare_threshold=None
):
    """
    Prepare categorical variables.

    Rare categories can be grouped as 'Other'.
    """

    df = data[[variable]].copy()

    if keep_missing:

        df[variable] = df[variable].fillna(
            "Missing"
        )

    else:

        df = df.dropna()

    if rare_threshold is not None:

        freq = (
            df[variable]
            .value_counts(normalize=True)
        )

        rare_categories = freq[
            freq < rare_threshold
        ].index

        df.loc[
            df[variable].isin(rare_categories),
            variable
        ] = "Other"

    df["group"] = df[variable]

    return df

# ==========================================================
# Group Summary
# ==========================================================

def summarize(
    data,
    group,
    target=None
):
    """
    Summarize grouped observations.

    Parameters
    ----------
    data : DataFrame

    group : str

    target : str, optional

    Returns
    -------
    DataFrame
    """

    if target is None:

        summary = (
            data
            .groupby(group, observed=True)
            .size()
            .reset_index(name="observations")
        )

    else:

        summary = (
            data
            .groupby(group, observed=True)
            .agg(
                observations=(target, "size"),
                target_rate=(target, "mean")
            )
            .reset_index()
        )

    return summary

# ==========================================================
# Histogram
# ==========================================================

def plot_histogram(
    ax,
    data,
    variable,
    bins=40,
    kde=True,
    log_scale=False,
    color=None
):
    """
    Draw a standardized histogram.
    """

    if color is None:
        color = COLORS["primary"]

    sns.histplot(
        data=data,
        x=variable,
        bins=bins,
        kde=kde,
        color=color,
        edgecolor="white",
        linewidth=.5,
        log_scale=log_scale,
        ax=ax
    )

    format_axes(
        ax,
        xlabel=variable.replace("_", " ").title(),
        ylabel="Observations"
    )

    # ==========================================================
# Boxplot
# ==========================================================

def plot_boxplot(
    ax,
    data,
    variable,
    clip=None,
    color=None
):
    """
    Draw a standardized boxplot.

    Parameters
    ----------
    clip : float, optional
        Visualization clipping quantile.
    """

    if color is None:
        color = COLORS["secondary"]

    values = data[variable].dropna()

    if clip is not None:

        limit = values.quantile(clip)

        values = values[
            values <= limit
        ]

    sns.boxplot(
        x=values,
        width=.45,
        linewidth=1,
        fliersize=2,
        color=color,
        ax=ax
    )

    format_axes(
        ax,
        xlabel=variable.replace("_", " ").title()
    )

    if clip is not None:

        ax.text(
            .99,
            .95,
            f"Visualization truncated at P{clip:.0%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=FONT["annotation"],
            color=COLORS["dark_gray"]
        )

# ==========================================================
# Bar Plot
# ==========================================================

def plot_bar(
    ax,
    data,
    x,
    y,
    color=None,
    annotate=True,
    order=None
):
    """
    Draw a standardized bar chart.
    """

    if color is None:
        color = COLORS["secondary"]

    sns.barplot(
        data=data,
        x=x,
        y=y,
        order=order,
        color=color,
        edgecolor="white",
        linewidth=.8,
        ax=ax
    )

    if annotate:
        annotate_bars(ax)

    format_axes(ax)

# ==========================================================
# Line Plot
# ==========================================================

def plot_line(
    ax,
    data,
    x,
    y,
    color=None,
    marker="o",
    linewidth=2.5,
    annotate=False,
    percentage=False
):
    """
    Draw a standardized line plot.
    """

    if color is None:
        color = COLORS["danger"]

    sns.lineplot(
        data=data,
        x=x,
        y=y,
        marker=marker,
        linewidth=linewidth,
        color=color,
        sort=False,
        ax=ax
    )

    if annotate:

        for xpos, value in enumerate(data[y]):

            label = (
                f"{value:.1%}"
                if percentage
                else f"{value:,.2f}"
            )

            ax.text(
                xpos,
                value,
                label,
                ha="center",
                va="bottom",
                fontsize=FONT["annotation"],
                fontweight="bold",
                color=color
            )

    if percentage:
        format_percentage_axis(ax)

    format_axes(ax)

    # ==========================================================
# Bar Distribution
# ==========================================================

def plot_bar_distribution(
    ax,
    data,
    variable,
    normalize=False,
    max_categories=None,
    color=None
):
    """
    Plot the distribution of a categorical variable.

    Parameters
    ----------
    normalize : bool
        Display percentages instead of counts.

    max_categories : int, optional
        Display only the top categories.

    Returns
    -------
    pandas.DataFrame
        Summary used for plotting.
    """

    if color is None:
        color = COLORS["secondary"]

    values = (
        data[variable]
        .fillna("Missing")
        .value_counts(normalize=normalize)
    )

    if max_categories is not None:
        values = values.head(max_categories)

    summary = (
        values
        .rename_axis("category")
        .reset_index(name="value")
    )

    plot_bar(
        ax=ax,
        data=summary,
        x="category",
        y="value",
        color=color
    )

    if normalize:

        annotate_bars(
            ax,
            fmt="{:.1%}"
        )

        format_percentage_axis(ax)

        ax.set_ylabel("Percentage")

    else:

        ax.set_ylabel("Observations")

    ax.tick_params(
        axis="x",
        rotation=35
    )

    return summary

# ==========================================================
# Target Rate
# ==========================================================

def plot_target_rate(
    ax,
    summary,
    group="group",
    target_rate="target_rate"
):
    """
    Plot target/default rate by groups.

    Parameters
    ----------
    summary : DataFrame

    group : str

    target_rate : str
    """

    plot_line(
        ax=ax,
        data=summary,
        x=group,
        y=target_rate,
        percentage=True,
        annotate=True,
        color=COLORS["danger"]
    )

    ax.set_ylabel("Target Rate")

    # ==========================================================
# Missing Values
# ==========================================================

def plot_missing(
    ax,
    data,
    variables=None,
    color=None
):
    """
    Plot percentage of missing values.

    Parameters
    ----------
    variables : list, optional
        Variables to evaluate.
    """

    if color is None:
        color = COLORS["warning"]

    if variables is None:
        variables = data.columns

    summary = (
        data[variables]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    summary.columns = [
        "variable",
        "missing_rate"
    ]

    plot_bar(
        ax=ax,
        data=summary,
        x="variable",
        y="missing_rate",
        color=color
    )

    annotate_bars(
        ax,
        fmt="{:.1%}"
    )

    format_percentage_axis(ax)

    ax.set_ylabel("Missing Rate")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    return summary

# ==========================================================
# Heatmap
# ==========================================================

def plot_heatmap(
    ax,
    matrix,
    cmap=None,
    annot=True,
    fmt=".2f",
    center=None
):
    """
    Draw a standardized heatmap.

    Parameters
    ----------
    matrix : DataFrame

    cmap : matplotlib colormap

    annot : bool

    fmt : str

    center : float, optional
    """

    if cmap is None:
        cmap = sns.light_palette(
            COLORS["primary"],
            as_cmap=True
        )

    sns.heatmap(
        matrix,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        linewidths=.5,
        linecolor="white",
        square=True,
        center=center,
        cbar_kws={
            "shrink": .8
        },
        ax=ax
    )

    format_axes(ax)