import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from . import visualization as viz

# ==========================================================
# Dataset Characterization
# ==========================================================

def plot_dataset_characterization(
    data,
    target="target"
):
    """
    Dataset characterization.

    Displays:

    • Target distribution
    • Missing values
    • Variable types
    """

    fig, axes = viz.create_figure(
        n_rows=1,
        n_cols=3,
        figsize=(18,5),
        title="Training Dataset Characterization"
    )

    # ======================================================
    # Target distribution
    # ======================================================

    summary = (
        data[target]
        .value_counts()
        .rename_axis(target)
        .reset_index(name="observations")
    )

    viz.plot_bar(
        ax=axes[0,0],
        data=summary,
        x=target,
        y="observations"
    )

    viz.annotate_bars(axes[0,0])

    viz.format_axes(
        axes[0,0],
        title="Target Distribution",
        xlabel="",
        ylabel="Observations"
    )

    # ======================================================
    # Missing values
    # ======================================================

    viz.plot_missing(
        ax=axes[0,1],
        data=data
    )

    axes[0,1].set_title("Missing Values (%)")

    # ======================================================
    # Variable types
    # ======================================================

    dtype_summary = (
        data.dtypes
        .astype(str)
        .value_counts()
        .rename_axis("dtype")
        .reset_index(name="variables")
    )

    viz.plot_bar(
        ax=axes[0,2],
        data=dtype_summary,
        x="dtype",
        y="variables"
    )

    viz.annotate_bars(axes[0,2])

    viz.format_axes(
        axes[0,2],
        title="Variable Types",
        xlabel="",
        ylabel="Variables"
    )

    plt.tight_layout()
    plt.show()



# ==========================================================
# Numerical Profile
# ==========================================================

def plot_numerical_profile(
    data,
    variables,
    titles=None,
    bins=40,
    log_variables=None,
    truncate_boxplot=None,
    section_title=None
):
        """
    Generic numerical univariate analysis.

    Displays

    • Histogram
    • Boxplot

    Designed for:

    • Financial Profile
    • Credit Risk Indicators
    """

        if titles is None:
            titles = {}

        if log_variables is None:
            log_variables = []

        if truncate_boxplot is None:
            truncate_boxplot = {}

        fig, axes = viz.create_figure(
            n_rows=len(variables),
            n_cols=2,
            figsize=(14, 4 * len(variables)),
            title=section_title
        )

        for i, variable in enumerate(variables):

            title = titles.get(
                variable,
                variable.replace("_", " ").title()
            )

            # ==================================================
            # Histogram
            # ==================================================

            viz.plot_histogram(
                ax=axes[i,0],
                data=data,
                variable=variable,
                bins=bins,
                log_scale=variable in log_variables
            )

            viz.format_axes(
                axes[i,0],
                title=f"{title} Distribution",
                xlabel=title,
                ylabel="Observations"
            )

            # ==================================================
            # Boxplot
            # ==================================================

            viz.plot_boxplot(
                ax=axes[i,1],
                data=data,
                variable=variable,
                clip=truncate_boxplot.get(variable)
            )

            viz.format_axes(
                axes[i,1],
                title=f"{title} Boxplot",
                xlabel=title
            )

        plt.tight_layout()
        plt.show()

# ==========================================================
# Credit History Profile
# ==========================================================

def plot_credit_history_profile(
    data,
    variables,
    titles=None,
    bins=10,
    zero_threshold=0.60,
    section_title= None
):
    """
    Univariate analysis for credit history variables.

    This section focuses on understanding the distribution
    of historical credit behavior, highlighting zero inflation
    and informative missing values.
    """

    if titles is None:
        titles = {}

    fig, axes = viz.create_figure(
        n_rows=len(variables),
        n_cols=3,
        title=section_title
    )

    for i, variable in enumerate(variables):

        title = titles.get(
            variable,
            variable.replace("_", " ").title()
        )

        # -------------------------------
        # Histogram
        # -------------------------------

        viz.plot_histogram(
            ax=axes[i,0],
            data=data,
            variable=variable
        )

        axes[i,0].set_title(title)

        # -------------------------------
        # Boxplot
        # -------------------------------

        viz.plot_boxplot(
            ax=axes[i,1],
            data=data,
            variable=variable
        )

        axes[i,1].set_title("Boxplot")

        # -------------------------------
        # Missing values
        # -------------------------------

        viz.plot_missing(
            ax=axes[i,2],
            data=data,
            variable=variable
        )

        axes[i,2].set_title("Missing Values")

    plt.tight_layout()

    plt.show()

# ==========================================================
# Categorical Profile
# ==========================================================

def plot_categorical_profile(
    data,
    variables,
    titles=None,
    normalize=False,
    rare_threshold=0.01
):
    """
    Univariate analysis for categorical variables.

    Displays category frequencies after grouping
    infrequent categories into 'Other'.
    """

    if titles is None:
        titles = {}

    fig, axes = viz.create_figure(
        n_rows=len(variables),
        n_cols=1,
        title="Loan Characteristics"
    )

    if len(variables) == 1:
        axes = axes.reshape(-1,1)

    for i, variable in enumerate(variables):

        title = titles.get(
            variable,
            variable.replace("_"," ").title()
        )

        prepared = viz.prepare_categorical_groups(
            data=data,
            variable=variable,
            rare_threshold=rare_threshold
        )

        summary = viz.plot_bar_distribution(
            ax=axes[i,0],
            data=prepared,
            variable="group",
            normalize=normalize
        )

        cardinality = prepared["group"].nunique()

        axes[i,0].set_title(
            f"{title} (Cardinality = {cardinality})"
        )

    plt.tight_layout()

    plt.show()

def plot_bivariate_numeric(
    data,
    variables,
    target,
    titles=None,
    bins=10,
    zero_threshold=0.60,
    section_title=None
):
    """
    Bivariate analysis for numerical variables.

    For each variable, displays:

    Left
        Observation distribution

    Right
        Target rate by group.

    Continuous variables are grouped into quantiles.
    Zero-inflated variables are grouped by their original values.
    """

    if titles is None:
        titles = {}

    fig, axes = viz.create_figure(
        n_rows=len(variables),
        n_cols=2,
        figsize=(15, 4 * len(variables)),
        title=section_title
    )

    summaries = {}

    for i, variable in enumerate(variables):

        title = titles.get(
            variable,
            variable.replace("_", " ").title()
        )

        grouped = viz.prepare_numeric_groups(
            data=data,
            variable=variable,
            bins=bins,
            zero_threshold=zero_threshold
        )

        grouped[target] = data.loc[grouped.index, target]

        summary = viz.summarize(
            grouped,
            group="group",
            target=target
        )

        summaries[variable] = summary

        # Distribution

        viz.plot_bar(
            ax=axes[i, 0],
            data=summary,
            x="group",
            y="observations"
        )

        viz.format_axes(
            ax=axes[i, 0],
            title=f"{title}\nObservation Distribution",
            xlabel="",
            ylabel="Observations"
        )

        # Target Rate

        viz.plot_target_rate(
            ax=axes[i,1],
            summary=summary,
            group="group",
            target_rate="target_rate"
        )

        viz.format_axes(
            ax=axes[i,1],
            title=f"{title}\nDefault Rate",
            xlabel="",
            ylabel="Default Rate"
        )

    plt.tight_layout()

    return summaries

def plot_bivariate_categorical(
    data,
    variables,
    target,
    titles=None,
    rare_threshold=0.01,
    section_title=None
):
    """
    Bivariate analysis for categorical variables.

    Left
        Category frequency.

    Right
        Target rate by category.
    """

    if titles is None:
        titles = {}

    fig, axes = viz.create_figure(
        n_rows=len(variables),
        n_cols=2,
        figsize=(18, 4.5 * len(variables)),
        title=section_title
    )

    summaries = {}

    for i, variable in enumerate(variables):

        title = titles.get(
            variable,
            variable.replace("_", " ").title()
        )

        grouped = viz.prepare_categorical_groups(
            data=data,
            variable=variable,
            rare_threshold=rare_threshold
        )

        grouped[target] = data.loc[grouped.index, target]

        summary = viz.summarize(
            grouped,
            group="group",
            target=target
        )

        summaries[variable] = summary

        # Frequency

        # ==================================================
        # Observation Distribution
        # ==================================================
        print(variable)
        print(summary.head())
        print(summary.shape)
        print("-" * 40)
        viz.plot_bar(
            ax=axes[i, 0],
            data=summary,
            x="group",
            y="observations",
            annotate=False
        )

        viz.format_axes(
            ax=axes[i, 0],
            title=f"{title}\nObservation Distribution",
            xlabel="",
            ylabel="Observations"
        )

        # ==================================================
        # Default Rate
        # ==================================================

        viz.plot_target_rate(
            ax=axes[i, 1],
            summary=summary
        )

        viz.format_axes(
            ax=axes[i, 1],
            title=f"{title}\nDefault Rate",
            xlabel="",
            ylabel="Default Rate"
        )
    
    plt.tight_layout()
    plt.show()
    return summaries

# ==========================================================
# Loan Characteristics
# ==========================================================

# ==========================================================
# Loan Characteristics
# ==========================================================

def plot_loan_characteristics(
    data,
    variables,
    titles=None,
    normalize=False,
    rare_threshold=0.01,
    section_title=None
):
    """
    Univariate analysis for loan characteristics.

    Displays

    • Category frequency
    • Cardinality
    • Rare categories
    • Missing values
    """

    if titles is None:
        titles = {}

    fig, axes = viz.create_figure(
        n_rows=len(variables),
        n_cols=1,
        figsize=(18, 4.5 * len(variables)),
        title=section_title
    )

    if len(variables) == 1:
        axes = axes.reshape(-1, 1)

    for i, variable in enumerate(variables):

        title = titles.get(
            variable,
            variable.replace("_", " ").title()
        )

        # --------------------------------------------------
        # Summary statistics
        # --------------------------------------------------

        missing_pct = data[variable].isna().mean()

        rare_categories = (
            data[variable]
            .value_counts(normalize=True)
            .lt(rare_threshold)
            .sum()
        )

        # --------------------------------------------------
        # Prepare grouped categories
        # --------------------------------------------------

        # Cardinality BEFORE grouping rare categories
        n_categories = (
            data[variable]
            .dropna()
            .nunique()
        )

        prepared = viz.prepare_categorical_groups(
            data=data,
            variable=variable,
            rare_threshold=rare_threshold
)

        # --------------------------------------------------
        # Frequency plot
        # --------------------------------------------------
        
        viz.plot_bar_distribution(
            ax=axes[i, 0],
            data=prepared,
            variable="group",
            normalize=normalize,
            annotate=n_categories <= 12
        )

        viz.format_axes(
            ax=axes[i, 0],
            title=title
        )

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        metadata = (
            f"Cardinality: {n_categories}\n"
            f"Rare Categories: {rare_categories}\n"
            f"Missing: {missing_pct:.1%}"
        )

        axes[i, 0].text(
            0.99,
            0.98,
            metadata,
            transform=axes[i, 0].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.90,
                boxstyle="round,pad=.3"
            )
        )

    plt.tight_layout()

    plt.show()

# ==========================================================
# Correlation Analysis
# ==========================================================

def plot_correlation_analysis(
    data,
    variables,
    method="spearman",
    figsize=(12, 10)
):
    """
    Plot correlation matrix for numerical variables.

    Uses Spearman correlation by default because
    credit variables usually contain skewness and outliers.
    """

    corr = (
        data[variables]
        .corr(method=method)
    )

    fig, ax = viz.create_figure(
        figsize=figsize,
        title=f"{method.title()} Correlation Matrix"
    )

    ax = ax[0, 0]

    viz.plot_heatmap(
        ax=ax,
        matrix=corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0
    )

    # ==========================================
    # Axis formatting
    # ==========================================

    ax.tick_params(
        axis="x",
        rotation=45,
        labelsize=9
    )

    ax.tick_params(
        axis="y",
        rotation=0,
        labelsize=9
    )

    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    plt.tight_layout()

    return corr

def find_redundant_variables(
    corr_matrix,
    threshold=0.85
):
    """
    Identify highly correlated variable pairs.

    Parameters
    ----------
    corr_matrix : DataFrame

    threshold : float
        Correlation threshold.

    Returns
    -------
    DataFrame
    """

    correlations = (
        corr_matrix
        .where(
            ~np.eye(
                corr_matrix.shape[0],
                dtype=bool
            )
        )
        .stack()
        .reset_index()
    )

    correlations.columns = [
        "variable_1",
        "variable_2",
        "correlation"
    ]

    redundant = (
        correlations
        .loc[
            correlations["correlation"].abs() >= threshold
        ]
    )

    redundant = redundant[
        redundant["variable_1"]
        <
        redundant["variable_2"]
    ]

    return (
        redundant
        .sort_values(
            "correlation",
            ascending=False
        )
        .reset_index(drop=True)
    )

def calculate_vif(
    data,
    variables
):
    """
    Calculate Variance Inflation Factor.

    """

    X = (
        data[variables]
        .dropna()
        .copy()
    )


    vif = pd.DataFrame()

    vif["variable"] = X.columns

    vif["VIF"] = [
        variance_inflation_factor(
            X.values,
            i
        )
        for i in range(X.shape[1])
    ]

    return (
        vif
        .sort_values(
            "VIF",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ==========================================================
# Preprocessing Summary
# ==========================================================

def plot_preprocessing_summary(
    summary
):
    """
    Display preprocessing recommendations obtained
    during the exploratory analysis.

    Parameters
    ----------
    summary : DataFrame

    Required columns
    ----------------
    variable
    recommendation
    """

    fig, ax = plt.subplots(
        figsize=(12, 0.6 * len(summary) + 1)
    )

    ax.axis("off")

    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    plt.title(
        "Preprocessing Recommendations",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    plt.show()