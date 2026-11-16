# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 CEA and ANSES

# -------------------------------------------------------------
#                           IMPORTS
# -------------------------------------------------------------

from __future__ import annotations

from ipywidgets import Output, HBox, VBox, Dropdown, Layout
from typing import Sequence, Dict, Optional
from IPython.display import display

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import defaultdict

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# -------------------------------------------------------------
#              PREPROCESSING PLOTTING FUNCTIONS
# -------------------------------------------------------------


def plot_spectra_by_source(df: pd.DataFrame, source_series: pd.Series):
    """
    Display spectra grouped by source in a Plotly subplot grid.

    One subplot is created per unique value of ``source_series`` (grid
    of 2 columns); each spectrum of a source is drawn as a line with
    Raman shift (column labels, cast to float) on x and intensity on y.

    Parameters
    ----------
    df : pd.DataFrame
        Spectra DataFrame: rows = spectra, columns = pixels (Raman shift).
    source_series : pd.Series
        Source label of each spectrum, aligned with ``df.index``.

    Returns
    -------
    None
        The figure is shown directly via ``fig.show()``.
    """
    unique_sources = source_series.unique()
    num_sources = len(unique_sources)

    cols = 2
    rows = (num_sources + 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=unique_sources)

    for i, source in enumerate(unique_sources):
        row = (i // cols) + 1
        col = (i % cols) + 1

        subset = df[source_series == source]
        for _, spectrum in subset.iterrows():
            fig.add_trace(
                go.Scatter(x=df.columns.astype(float), y=spectrum, mode='lines', line=dict(width=1), name=str(source)),
                row=row, col=col
            )

        fig.update_xaxes(title_text="Raman Shift", row=row, col=col)
        fig.update_yaxes(title_text="Intensity", row=row, col=col)

    fig.update_layout(height=300*rows, width=800, title_text="Spectra by source", showlegend=False)
    fig.show()


# -------------------------------------------------------------


def plot_spectra_by_source_dropdown(df: pd.DataFrame, source_series: pd.Series):
    """
    Display spectra source by source using an interactive dropdown.

    An ``ipywidgets.Dropdown`` lists the unique sources (sorted); each
    selection redraws a Plotly ``FigureWidget`` containing all spectra of
    the selected source, with hover info (sample id, Raman shift,
    intensity).

    Parameters
    ----------
    df : pd.DataFrame
        Spectra DataFrame: rows = samples, columns = abscissas (Raman
        shift values, cast to float for plotting).
    source_series : pd.Series
        Source label of each sample, with the same index as ``df``.

    Returns
    -------
    None
        The dropdown + figure widget is displayed in the notebook.

    Raises
    ------
    ValueError
        If ``df.index`` and ``source_series.index`` are not identical.

    Notes
    -----
    Requires ``plotly`` (``FigureWidget``) and ``ipywidgets``, and must be
    run in a Jupyter/IPython environment for the dropdown and the figure
    to render.
    """

    if not df.index.equals(source_series.index):
        raise ValueError("Index from df and source_series must be identical.")

    unique_sources = sorted(source_series.unique())
    x_vals = df.columns.astype(float)

    fig = go.FigureWidget(
        layout=go.Layout(
            title="Raman Spectrum",
            xaxis_title="Raman Shift (cm⁻¹)",
            yaxis_title="Intensity (a.u.)",
            height=500,
            width=1200
        )
    )

    dropdown = Dropdown(
        options=unique_sources,
        value=unique_sources[0],
        description="Source :",
        layout={'width': '300px'}
    )

    def update_plot(change=None):

        """Redraw the figure with all spectra of the source selected in the dropdown."""

        source = dropdown.value
        subset = df[source_series == source]

        fig.data = []

        for idx, spectrum in subset.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=spectrum.values,
                    mode='lines',
                    line=dict(width=1),
                    name=str(idx),
                    hovertemplate=f"{idx}<br>Raman Shift: %{{x:.1f}}<br>Intensity: %{{y:.3f}}<extra></extra>"
                )
            )

        fig.update_layout(title=f"Spectra – Source: {source}")

    update_plot()
    dropdown.observe(update_plot, names="value")

    display(VBox([dropdown, fig]))


# -------------------------------------------------------------


def plot_median_by_source_dropdown(
    df: pd.DataFrame,
    source_series: pd.Series,
    alpha: float = 0.25,
    figsize: tuple = (1200, 500)
):
    """
    Display the median ± std of Raman spectra per source via an interactive dropdown.

    An ``ipywidgets.Dropdown`` lists the unique sources (sorted); each
    selection redraws a Plotly ``FigureWidget`` showing, for the selected
    source, the per-pixel median as a line and the ±1 standard deviation
    band as a semi-transparent filled area (one color per source).

    Parameters
    ----------
    df : pd.DataFrame
        Spectra DataFrame: rows = samples, columns = Raman shift values
        (cast to float for plotting).
    source_series : pd.Series
        Source label of each sample, with the same index as ``df``.
    alpha : float, default=0.25
        Opacity of the ±1σ band (between 0 and 1).
    figsize : tuple, default=(1200, 500)
        Figure size as (width, height), in pixels.

    Returns
    -------
    None
        The dropdown + figure widget is displayed in the notebook.

    Raises
    ------
    ValueError
        If ``df.index`` and ``source_series.index`` are not identical.

    Notes
    -----
    Requires ``plotly`` (``FigureWidget``) and ``ipywidgets``, in a
    Jupyter/IPython environment.
    """

    if not df.index.equals(source_series.index):
        raise ValueError("Index from df and source_series must be identical.")

    unique_sources = sorted(source_series.unique())
    x_vals = df.columns.astype(float)
    palette = px.colors.qualitative.Plotly

    def to_rgba(color: str, alpha: float = 0.25) -> str:
        """Convert a Plotly color (#hex or 'rgb(...)') to an 'rgba(...)' string with the given alpha."""
        if color.startswith("#"):
            color = color.lstrip("#")
            r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"
        elif color.startswith("rgb"):
            vals = color.strip("rgb() ").split(",")
            r, g, b = [int(v) for v in vals]
            return f"rgba({r},{g},{b},{alpha})"
        else:
            raise ValueError(f"color format unknown: {color}")

    fig = go.FigureWidget(
        layout=go.Layout(
            title="Median spectra by source",
            xaxis_title="Raman Shift (cm⁻¹)",
            yaxis_title="Intensity (a.u.)",
            height=figsize[1],
            width=figsize[0],
            template="plotly_white",
        )
    )

    dropdown = Dropdown(
        options=unique_sources,
        value=unique_sources[0],
        description="Source :",
        layout={'width': '300px'}
    )

    def update_plot(change=None):
        """Redraw the figure with the median and ±1 std band of the source selected in the dropdown."""
        source = dropdown.value
        subset = df[source_series == source]

        y_med = subset.median().values
        y_std = subset.std().values

        color = palette[unique_sources.index(source) % len(palette)]
        fill_color = to_rgba(color, alpha)

        fig.data = []

        fig.add_trace(go.Scatter(
            x=np.concatenate([x_vals, x_vals[::-1]]),
            y=np.concatenate([y_med + y_std, (y_med - y_std)[::-1]]),
            fill='toself',
            fillcolor=fill_color,
            line=dict(width=0),
            hoverinfo='skip',
            name=f"{source} ± σ",
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_med,
            mode='lines',
            line=dict(color=color, width=2),
            name=f"{source} (median)"
        ))

        fig.update_layout(title=f"median ± σ – Source : {source}")

    update_plot()
    dropdown.observe(update_plot, names="value")

    display(VBox([dropdown, fig]))


# -------------------------------------------------------------


def pca_by_source_metadata(df, source_col_new, meta_index=0, n_components=2, pattern_sep='_'):
    """
    Compute and display one PCA per group of a metadata field.

    Each label of ``source_col_new`` is split on ``pattern_sep`` and the
    part at position ``meta_index`` defines the groups. For each group,
    the corresponding rows of ``df`` are standardized (StandardScaler),
    reduced by PCA, and shown as an interactive Plotly scatter plot
    (PC1 vs PC2) colored by the remaining metadata parts.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (index = samples, columns = features).
    source_col_new : pd.Series
        Metadata label of each sample, with the same index as ``df``;
        each label holds several metadata separated by ``pattern_sep``.
    meta_index : int, default=0
        Position of the metadata part used to split the groups
        (0 = first element before ``pattern_sep``).
    n_components : int, default=2
        Number of principal components computed. Must be >= 2, since
        PC1 and PC2 are used for the 2D plot.
    pattern_sep : str, default='_'
        Separator used to split the metadata labels.

    Returns
    -------
    None
        One interactive Plotly figure is displayed per group.

    Raises
    ------
    ValueError
        If ``df.index`` and ``source_col_new.index`` are not identical.

    Notes
    -----
    - Each label must contain at least ``meta_index + 1`` parts
      (otherwise indexing fails with IndexError).
    - Features are standardized within each group before the PCA.
    - The PC1/PC2 axis labels include the explained variance ratio (%).
    - Points are colored by the remaining metadata parts (all parts
      except the one at ``meta_index``, rejoined with ``pattern_sep``).
    """

    if not df.index.equals(source_col_new.index):
        raise ValueError("Index from df and source_series must be identical.")

    meta_split = source_col_new.astype(str).apply(lambda x: x.split(pattern_sep))
    
    selected_meta = meta_split.apply(lambda x: x[meta_index])
    unique_groups = np.unique(selected_meta)

    print(f"Groups for meta_index={meta_index}: {len(unique_groups)} ({unique_groups})")

    for group in unique_groups:
        mask = selected_meta == group
        df_subset = df[mask]

        meta_others = meta_split[mask].apply(lambda parts: "_".join([p for i, p in enumerate(parts) if i != meta_index]))
        
        X_scaled = StandardScaler().fit_transform(df_subset.values)
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "Color": meta_others.values,
            "Sample": df_subset.index
        })

        # Plot
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Color",
            hover_name="Sample",
            title=f"ACP - Groupe '{group}' (metadata {meta_index})",
            labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)",
                    "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)"}
        )

        fig.show()


# -------------------------------------------------------------


def interactive_pca_by_source_metadata_colored(
    df: pd.DataFrame,
    source_col_new: pd.Series,
    meta_index: int = 0,
    pc_pair: list = [0, 1],
    n_components_total: int = 10,
    pattern_sep: str = '_'
):
    """
    Interactive PCA per metadata group, with click-to-inspect spectra.

    Each label of ``source_col_new`` is split on ``pattern_sep``; the part
    at position ``meta_index`` defines the groups, selected through an
    interactive dropdown. For the selected group, the spectra are
    standardized (StandardScaler) and reduced by PCA, and the two
    components given by ``pc_pair`` are shown as a Plotly scatter plot
    colored by the remaining metadata parts. Clicking a point displays
    the corresponding spectrum in a second figure.

    Parameters
    ----------
    df : pd.DataFrame
        Spectral data (rows = samples, columns = Raman positions).
    source_col_new : pd.Series
        Metadata labels of the samples, with the same index as ``df``.
    meta_index : int, default=0
        Index of the metadata part used to define the groups.
    pc_pair : list of int, default=[0, 1]
        Pair of PC indices to display (0-based), e.g. [0, 1] or [0, 3];
        must contain exactly two elements.
    n_components_total : int, default=10
        Total number of PCs computed; must be greater than max(pc_pair).
    pattern_sep : str, default='_'
        Separator of the metadata parts within each label.

    Returns
    -------
    list of np.ndarray
        PCA loadings (``pca.components_``, shape
        (n_components_total, n_features)) of the group initially shown.

    Raises
    ------
    ValueError
        If ``df.index`` and ``source_col_new.index`` differ, or if
        ``pc_pair`` does not contain exactly two indices.

    Notes
    -----
    Requires ``plotly`` (``FigureWidget``) and ``ipywidgets`` in a
    Jupyter/IPython environment. Each label must contain at least
    ``meta_index + 1`` parts.
    """

    if not df.index.equals(source_col_new.index):
        raise ValueError("Index from df and source_series must be identical.")
    if len(pc_pair) != 2:
        raise ValueError("pc_pair parameter must be a two length list, ex. [0, 1].")

    meta_split = source_col_new.astype(str).apply(lambda x: x.split(pattern_sep))
    selected_meta = meta_split.apply(lambda x: x[meta_index])
    unique_groups = np.unique(selected_meta)

    print(f"Groups for meta_index={meta_index} : {len(unique_groups)} ({unique_groups})")
    print(f"Plotted PC pairs: PC{pc_pair[0]+1} and PC{pc_pair[1]+1}")

    dropdown = Dropdown(
        options=list(unique_groups),
        value=list(unique_groups)[0],
        description="Groupe:",
        layout=Layout(width="300px")
    )

    fig_pca = go.FigureWidget(
        layout=go.Layout(
            title="spectra PCA",
            height=500,
            width=600,
            template="plotly_white"
        )
    )
    fig_spectrum = go.FigureWidget(
        layout=go.Layout(
            title="Selected spectrum",
            xaxis_title="Raman Shift (cm⁻¹)",
            yaxis_title="Intensity (a.u.)",
            height=500,
            width=600,
            template="plotly_white",
            margin=dict(t=90)
        )
    )

    out = Output()

    def update_pca(change=None):
        """Recompute the PCA of the group selected in the dropdown and refresh the scatter plot."""
        group = dropdown.value
        mask = selected_meta == group
        df_subset = df[mask]

        if df_subset.empty:
            with out:
                out.clear_output(wait=True)
                print(f"No samples for the group: {group}")
            fig_pca.data = ()
            fig_spectrum.data = ()
            return

        meta_others = meta_split[mask].apply(
            lambda parts: "_".join([p for i, p in enumerate(parts) if i != meta_index])
        )

        X_scaled = StandardScaler().fit_transform(df_subset.values)

        components = []

        pca = PCA(n_components=n_components_total)
        X_pca = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_ * 100
        components.append(pca.components_)

        pcx, pcy = pc_pair
        pcx_label = f"PC{pcx+1} ({explained[pcx]:.1f}%)"
        pcy_label = f"PC{pcy+1} ({explained[pcy]:.1f}%)"

        pca_df = pd.DataFrame(
            {f"PC{pcx+1}": X_pca[:, pcx], f"PC{pcy+1}": X_pca[:, pcy]},
            index=df_subset.index
        )
        pca_df["Color"] = meta_others.values

        fig_temp = px.scatter(
            pca_df,
            x=f"PC{pcx+1}",
            y=f"PC{pcy+1}",
            color="Color",
            hover_name=pca_df.index,
            custom_data=[pca_df.index],
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=f"PCA - Groupe '{group}' (PC{pcx+1} vs PC{pcy+1})",
            labels={f"PC{pcx+1}": pcx_label, f"PC{pcy+1}": pcy_label}
        )

        for tr in fig_temp.data:
            tr.update(hovertemplate=(
                f"{pcx_label}=%{{x:.3f}}<br>"
                f"{pcy_label}=%{{y:.3f}}<br>"
                "Sample=%{customdata[0]}<extra></extra>"
            ))

        fig_pca.data = ()
        for tr in fig_temp.data:
            fig_pca.add_trace(tr)
        fig_pca.update_layout(fig_temp.layout)

        def on_click(trace, points, selector):
            """Display the spectrum of the clicked PCA point in the second figure."""
            if not points.point_inds:
                return
            ind = points.point_inds[0]
            sample_name = trace.customdata[ind][0]
            spectrum = df_subset.loc[sample_name]

            with out:
                out.clear_output(wait=True)
            fig_spectrum.data = []
            fig_spectrum.layout.annotations = ()
            fig_spectrum.add_trace(
                go.Scatter(
                    x=df.columns.astype(float),
                    y=spectrum.values,
                    mode="lines",
                    line=dict(color="red", width=2),
                    name=str(sample_name),
                    hovertemplate=(
                        f"Sample={sample_name}<br>"
                        "Raman Shift=%{x:.1f}<br>"
                        "Intensity=%{y:.3f}<extra></extra>"
                    )
                )
            )
            fig_spectrum.update_layout(
                title=dict(text="Selected spectrum", x=0.5),
                margin=dict(t=90)
            )
            fig_spectrum.add_annotation(
                x=0, y=1.08, xref="paper", yref="paper",
                text=f"Sample: {sample_name}",
                showarrow=False, xanchor="left", align="left",
                font=dict(size=12)
            )

        for tr in fig_pca.data:
            tr.on_click(on_click)

        return components

    components = update_pca()
    dropdown.observe(update_pca, names="value")

    display(VBox([dropdown, HBox([fig_pca, fig_spectrum]), out]))

    return components


# -------------------------------------------------------------
#              PROCESSING PLOTTING FUNCTIONS
# -------------------------------------------------------------


def plot_group_median_dict(
    dict_dfs: Dict[str, pd.DataFrame],
    title: str = "",
    xlabel: str = "Raman Shift (cm⁻¹)",
    ylabel: str = "Intensity (a.u.)",
    figsize: tuple = (900, 500),
    alpha: float = 0.2,
    sigma_factor: float = 1.0
):
    """
    Plots the median of the Raman spectra for each condition (dictionary key),
    with a ± σ shaded band of the same color, visible in the legend.


    Parameters
    ----------
    dict_dfs : dict[str, pd.DataFrame]
        Dictionary {condition: DataFrame}, where each DataFrame contains:
        - rows = individual spectra
        - columns = Raman shifts (cm⁻¹)
    title : str
        Title of the plot.
    xlabel, ylabel : str
        Axis labels.
    figsize : tuple
        Figure size (width, height).
    alpha : float
        Transparency of the shaded band (between 0 and 1).
    sigma_factor : float
        Multiplicative factor applied to the standard deviation
        (e.g. 2 → ±2σ).
    """

    palette = px.colors.qualitative.Plotly
    colors = {cond: palette[i % len(palette)] for i, cond in enumerate(dict_dfs.keys())}

    def to_rgba(color: str, alpha: float = 0.2) -> str:
        """Covert an hex or rgb() color to rgba()."""
        if color.startswith("#"):
            color = color.lstrip("#")
            r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"
        elif color.startswith("rgb"):
            vals = color.strip("rgb() ").split(",")
            r, g, b = [int(v) for v in vals]
            return f"rgba({r},{g},{b},{alpha})"
        else:
            raise ValueError(f"Unrecognized color format: {color}")

    fig = go.Figure()

    for cond, df in dict_dfs.items():
        if df.empty:
            continue

        x_vals = df.columns.astype(float)
        y_med = df.median().values
        y_std = df.std().values * sigma_factor
        color = colors[cond]
        fill_color = to_rgba(color, alpha)

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_vals, x_vals[::-1]]),
                y=np.concatenate([y_med + y_std, (y_med - y_std)[::-1]]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(width=0),
                hoverinfo="skip",
                name=f"{cond} ± {sigma_factor}σ",
                legendgroup=cond,
                showlegend=True
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_med,
                mode="lines",
                line=dict(width=2, color=color),
                name=f"{cond} (median)",
                legendgroup=cond,
                showlegend=True
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=figsize[0],
        height=figsize[1],
        template="plotly_white",
        legend_title="Conditions",
    )

    fig.show()


# -------------------------------------------------------------


def pca_loadings_and_median_spectra(
    pca_results: dict,
    spectra_median_dict: dict,
    pc_pair: list = [0, 1],
    prefix_selection: list = None,
    label_map: dict = None,
    exclude_labels: list = None,
    colors: list = None,
    label_order: list = None,
    figsize: tuple = (14, 6),
    pattern_sep: str = "_",
    tick_step: int = 1,
):
    """
    Plots, for each group from pca_dict:
        - left: barplots of the loadings of the selected PCs
        - right: the corresponding median spectra

    Parameters
    ----------
    pca_results : dict
        Dictionary {group: PCA results}, where each value contains the fitted
        PCA object (or its attributes) for that group.
    spectra_median_dict : dict
        Dictionary {group: median spectrum}, providing the median spectrum
        associated with each group.
    pc_pair : list, optional
        Indices of the principal components to display (default [0, 1]).
    prefix_selection : list, optional
        Labels (column prefixes) to keep in the analysis; if None, all are kept.
    label_map : dict, optional
        Mapping to rename labels for display (raw label to display name).
    exclude_labels : list, optional
        Labels to exclude from the analysis.
    colors : list, optional
        List of colors assigned to the labels; default palette if None.
    label_order : list, optional
        Order in which labels appear in the legend; if None, order of appearance.
    figsize : tuple, optional
        Figure size (width, height) (default (14, 6)).
    pattern_sep : str, optional
        Separator used to split column names and extract the label
        (default "_").
    tick_step : int, optional
        Step between x-axis ticks (default 1).
    """

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
    })

    default_palette = ["0.3", "0.5", "0.7", "0.85"]
    color_palette = list(colors) if colors else default_palette

    all_prefixes = list(pca_results.keys())
    if prefix_selection:
        ordered_prefixes = [p for p in prefix_selection if p in all_prefixes]
    else:
        ordered_prefixes = all_prefixes

    if len(ordered_prefixes) == 0:
        print("No selected group.")
        return

    fig, axes = plt.subplots(
        nrows=len(ordered_prefixes), ncols=2,
        figsize=figsize, squeeze=False,
        dpi=300
    )

    for row_idx, group_key in enumerate(ordered_prefixes):
        ax_load, ax_spec = axes[row_idx]

        res = pca_results[group_key]
        loadings = res["loadings"]
        explained_var = res["explained_var"]

        x_vals = loadings.index.astype(float)
        width = (x_vals[1] - x_vals[0]) * 0.35 / len(pc_pair)

        for i, pc in enumerate(pc_pair):
            color = color_palette[i % len(color_palette)]
            ax_load.bar(
                x_vals + (i - (len(pc_pair)-1)/2) * width,
                loadings[f"PC{pc+1}"],
                width=width,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                label=f"PC{pc+1} ({explained_var[pc]:.1f}%)"
            )

        tick_start = 400      
        tick_step  = 20  
        tick_end   = x_vals.max()

        tick_positions = np.arange(tick_start, tick_end+1, tick_step)

        ax_load.set_xticks(tick_positions)
        ax_load.set_xticklabels([f"{v:.0f}" for v in tick_positions],
                                rotation=0, ha="center")

        ax_load.grid(axis="x", linestyle="--", linewidth=0.8, alpha=1)

        ax_load.set_ylabel("Loadings (a.u.)")
        ax_load.set_title(f"{group_key} — Loadings")
        ax_load.legend(fontsize=9)
        ax_load.margins(x=0, y=0)

        spectra_entries = [
            (key, spec)
            for key, spec in spectra_median_dict.items()
            if key.startswith(group_key + pattern_sep)
        ]

        labeled_spectra = []
        for key, spec in spectra_entries:
            raw_label = "_".join(key.split(pattern_sep)[1:]) \
                        if pattern_sep in key else key

            mapped_label = label_map.get(raw_label, raw_label) \
                           if label_map else raw_label

            if exclude_labels and mapped_label in exclude_labels:
                continue

            labeled_spectra.append((mapped_label, key, spec))

        if label_order:
            ordering = list(label_order)
        elif label_map:
            ordering = list(label_map.values())
        else:
            ordering = None

        if ordering:
            order_dict = {lbl: i for i, lbl in enumerate(ordering)}
            labeled_spectra.sort(
                key=lambda x: order_dict.get(x[0], len(order_dict))
            )

        for i, (mapped_label, key, spec) in enumerate(labeled_spectra):
            color = color_palette[i % len(color_palette)]

            if isinstance(spec, pd.DataFrame):
                x_vals = spec.columns.astype(float)
                y_vals = spec.median(axis=0)
            elif isinstance(spec, pd.Series):
                x_vals = spec.index.astype(float)
                y_vals = spec.values
            else:
                continue

            ax_spec.plot(
                x_vals, y_vals,
                lw=1.5,
                color=color,
                label=mapped_label
            )

        ax_spec.set_title(f"{group_key} — Median $\\it{{Spectra}}$")
        ax_spec.set_xlabel("Raman Shift (cm⁻¹)")
        ax_spec.set_ylabel("Intensity (a.u.)")
        ax_spec.legend(fontsize=9)
        ax_spec.grid(axis="y", linestyle="--", linewidth=0.8)
        ax_spec.margins(x=0, y=0)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# -------------------------------------------------------------


def boxplot_ratio_and_median_spectra_v2(
    ratio_dict: dict,
    spectra_median_dict: dict,
    title: str = "",
    ylabel: str = "Ratio",
    figsize: tuple = (14, 6),
    label_map: Optional[dict] = None,
    exclude_labels: Optional[Sequence[str]] = None,
    label_order: Optional[Sequence[str]] = None,
    prefix_selection: Optional[Sequence[str]] = None,
    control_labels: Optional[Sequence[str]] = ("temoin", "control"),
    stats_bracket_spacing: float = 0.3,
    colors: Optional[list] = None,  
):
    """
    Plots, for each prefix:
        - a boxplot of the ratios per subgroup (same order as the spectra)
        - a plot of the overlaid median spectra (same colors)

    Parameters
    ----------
    ratio_dict : dict
        Dictionary {prefix: data}, where each value contains the ratios
        to plot as boxplots, organized by subgroup.
    spectra_median_dict : dict
        Dictionary {prefix: median spectrum}, providing the median spectrum
        associated with each prefix.
    title : str, optional
        Title of the figure (default "").
    ylabel : str, optional
        Label of the y-axis of the boxplot (default "Ratio").
    figsize : tuple, optional
        Figure size (width, height) (default (14, 6)).
    label_map : dict, optional
        Mapping to rename labels for display (raw label to display name).
    exclude_labels : sequence of str, optional
        Labels to exclude from the analysis.
    label_order : sequence of str, optional
        Order in which the labels appear on the x-axis and in the legend;
        if None, order of appearance.
    prefix_selection : sequence of str, optional
        Prefixes to keep; if None, all are kept.
    control_labels : sequence of str, optional
        Labels considered as controls (default ("temoin", "control")),
        used as the reference group for statistical comparisons.
    stats_bracket_spacing : float, optional
        Vertical spacing between the statistical annotation brackets
        (default 0.3).
    colors : list, optional
        List of colors assigned to the labels; default palette if None.
    """

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
    })

    default_palette = ["#1f77b4", "#4b9cd3", "#72b7f0", "#9cc8f7"]
    color_palette = list(colors) if colors else default_palette

    grouped_ratios = defaultdict(list)
    for key, ratios in ratio_dict.items():
        prefix = re.split(r"_", key)[0]
        grouped_ratios[prefix].append((key, ratios))

    grouped_spectra = defaultdict(list)
    for key, spec in spectra_median_dict.items():
        prefix = re.split(r"_", key)[0]
        grouped_spectra[prefix].append((key, spec))

    all_prefixes = list(grouped_ratios.keys())
    if prefix_selection:
        ordered_prefixes = [p for p in prefix_selection if p in grouped_ratios]
    else:
        ordered_prefixes = all_prefixes

    n_groups = len(ordered_prefixes)

    fig, axes = plt.subplots(nrows=n_groups, ncols=2, figsize=figsize, squeeze=False, dpi=300)

    for row_idx, prefix in enumerate(ordered_prefixes):
        ax_box, ax_spec = axes[row_idx]

        labeled_ratios = []
        for key, ratios in grouped_ratios[prefix]:
            raw_label = "_".join(key.split("_")[1:]) if "_" in key else key
            mapped_label = label_map.get(raw_label, raw_label) if label_map else raw_label
            
            if exclude_labels and mapped_label in exclude_labels:
                continue
                
            labeled_ratios.append((mapped_label, ratios))

        if label_order:
            ordering = list(label_order)
        elif label_map:
            ordering = list(label_map.values())
        else:
            ordering = None

        if ordering:
            order_dict = {lbl: i for i, lbl in enumerate(ordering)}
            labeled_ratios.sort(key=lambda x: order_dict.get(x[0], len(order_dict)))

        all_data = [ratios for _, ratios in labeled_ratios]
        labels = [label for label, _ in labeled_ratios]

        box = ax_box.boxplot(
            all_data,
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color="black", linewidth=1.2),
            boxprops=dict(linewidth=1, color="black"),
            whiskerprops=dict(linewidth=1, color="black"),
            capprops=dict(linewidth=1, color="black"),
        )

        for patch, color in zip(box["boxes"], color_palette * len(all_data)):
            patch.set_facecolor(color)

        ax_box.set_title(f"{prefix} — $\\it{{Ratio}}$")
        ax_box.set_ylabel(ylabel)
        ax_box.set_xticks(range(1, len(labels) + 1))
        ax_box.set_xticklabels(labels, ha="center")
        ax_box.margins(x=0, y=0)

        control_idx = None
        for i, lbl in enumerate(labels):
            if lbl.lower() in control_labels or any(lbl.lower() == lm.lower() for lm in control_labels):
                control_idx = i
                break

        if control_idx is not None and len(all_data) > 1:
            y_max = max([np.max(d) for d in all_data if len(d) > 0])
            y_min = min([np.min(d) for d in all_data if len(d) > 0])
            y_offset = (y_max - y_min) * 0.1
            current_height = y_max + y_offset * -2
            control_data = all_data[control_idx]

            for i, (lbl, data) in enumerate(zip(labels, all_data)):
                if i == control_idx:
                    continue

                normal_ctrl = shapiro(control_data)[1] > 0.05
                normal_test = shapiro(data)[1] > 0.05
                homo = levene(control_data, data)[1] > 0.05

                if normal_ctrl and normal_test:
                    stat, p = ttest_ind(control_data, data, equal_var=homo)
                else:
                    stat, p = mannwhitneyu(control_data, data, alternative="two-sided")

                stars = ""
                if p < 0.0005:
                    stars = "***"
                elif p < 0.005:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                else:
                    stars = 'n.s.' 

                if stars:
                    x1, x2 = control_idx + 1, i + 1
                    ax_box.plot([x1, x1, x2, x2],
                            [current_height, current_height + y_offset * 0.01, current_height + y_offset * 0.01, current_height],
                            lw=1.2, color='black')
                    ax_box.text((x1 + x2) / 2, current_height + y_offset * 0.010, stars,
                            ha='center', va='bottom', fontsize=12)
                    current_height += y_offset * 0.7

        spectra_entries = grouped_spectra.get(prefix, [])
        labeled_spectra = []
        for key, spec in spectra_entries:
            raw_label = "_".join(key.split("_")[1:]) if "_" in key else key
            mapped_label = label_map.get(raw_label, raw_label) if label_map else raw_label
            
            if exclude_labels and mapped_label in exclude_labels:
                continue
                
            labeled_spectra.append((mapped_label, key, spec))

        if ordering:
            order_dict = {lbl: i for i, lbl in enumerate(ordering)}
            labeled_spectra.sort(key=lambda x: order_dict.get(x[0], len(order_dict)))

        for i, (mapped_label, key, spec) in enumerate(labeled_spectra):
            color = color_palette[i % len(color_palette)]

            if isinstance(spec, pd.DataFrame):
                x_vals = spec.columns.astype(float)
                y_vals = spec.median(axis=0)
            elif isinstance(spec, pd.Series):
                x_vals = spec.index.astype(float)
                y_vals = spec.values
            else:
                continue

            ax_spec.plot(
                x_vals, y_vals,
                lw=1.5,
                color=color,
                label=mapped_label
            )

        ax_spec.set_title(f"{prefix} — Median $\\it{{Spectra}}$")
        ax_spec.set_xlabel("Raman Shift (cm⁻¹)")
        ax_spec.set_ylabel("Intensity (a.u.)")
        ax_spec.legend(fontsize=9)
        ax_spec.grid(axis="y", linestyle="--", linewidth=0.8)
        ax_spec.margins(x=0, y=0)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# -------------------------------------------------------------


def boxplot_pca_dict_matplotlib_v2(
    pca_results: dict,
    pc_numbers: Sequence[int],
    title: str = "",
    ylabel: str = "Scores",
    figsize: tuple = (10, 6),
    label_map: Optional[dict] = None,
    exclude_labels: Optional[Sequence[str]] = None,
    label_order: Optional[Sequence[str]] = None,
    group_order: Optional[Sequence[str]] = None,
    control_labels: Optional[Sequence[str]] = ("temoin", "control"),
    colors: Optional[Sequence[str]] = None
):
    """
    Displays a mosaic of boxplots of the PCA scores for several groups,
    with statistical comparison against the control.


    Parameters
    ----------
    pca_results : dict
        Dictionary {group: PCA results}, where each value contains the
        PCA scores (and associated labels) for that group.
    pc_numbers : sequence of int
        Indices of the principal components to display; one boxplot panel
        is generated per PC.
    title : str, optional
        Title of the figure (default "").
    ylabel : str, optional
        Label of the y-axis of the boxplots (default "Scores").
    figsize : tuple, optional
        Figure size (width, height) (default (10, 6)).
    label_map : dict, optional
        Mapping to rename labels for display (raw label to display name).
    exclude_labels : sequence of str, optional
        Labels to exclude from the analysis.
    label_order : sequence of str, optional
        Order in which the labels appear on the x-axis and in the legend;
        if None, order of appearance.
    group_order : sequence of str, optional
        Order in which the groups appear in the mosaic; if None,
        order of the pca_results keys.
    control_labels : sequence of str, optional
        Possible names of the control group (default ("temoin", "control")),
        used as the reference for statistical comparisons.
    colors : sequence of str, optional
        List of colors assigned to the labels; default palette if None.
    """

    hatch_patterns = ["---", "....", "///"]

    if len(pc_numbers) != len(pca_results):
        raise ValueError("pc_numbers length must be the same as keys length in pca_results.")

    grouped = defaultdict(list)
    for (key, result), pc_idx in zip(pca_results.items(), pc_numbers):
        prefix = re.split(r'_', key)[0]
        grouped[prefix].append((key, result, pc_idx))

    all_prefixes = list(grouped.keys())
    if group_order:
        ordered_prefixes = [g for g in group_order if g in grouped]
        ordered_prefixes += [g for g in all_prefixes if g not in group_order]
    else:
        ordered_prefixes = all_prefixes

    n_groups = len(ordered_prefixes)
    ncols = min(3, n_groups)
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False, dpi=300)
    axes = axes.flatten()

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0
    })

    grayscale_palette = ["0.3", "0.5", "0.7", "0.85"]

    for ax, prefix in zip(axes, ordered_prefixes):
        entries = grouped[prefix]
        all_data = []
        labels = []

        for (key, result, pc_idx) in entries:
            pcs = result["pcs"]
            subgroup_labels = result["subgroup_labels"]

            if pc_idx < 1 or pc_idx > pcs.shape[1]:
                raise ValueError(f"pc_number {pc_idx} is out of bond for '{key}'.")

            pc_scores = pcs[:, pc_idx - 1]
            df_scores = pd.DataFrame({
                "Score": pc_scores,
                "Subgroupe": subgroup_labels
            })

            for label in pd.unique(df_scores["Subgroupe"]):

                mapped_label = label_map.get(label, label) if label_map else label

                if exclude_labels and mapped_label in exclude_labels:
                    continue

                subset = df_scores.loc[df_scores["Subgroupe"] == label, "Score"]
                all_data.append(subset)
                labels.append(mapped_label)

        if label_order:
            order_dict = {lbl: i for i, lbl in enumerate(label_order)}
            sorted_idx = sorted(range(len(labels)), key=lambda i: order_dict.get(labels[i], len(order_dict)))
            all_data = [all_data[i] for i in sorted_idx]
            labels = [labels[i] for i in sorted_idx]

        if not all_data:
            ax.set_visible(False)
            continue

        box = ax.boxplot(
            all_data,
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color='black', linewidth=1.2),
            boxprops=dict(linewidth=1, color='black'),
            whiskerprops=dict(linewidth=1, color='black'),
            capprops=dict(linewidth=1, color='black'),
        )

        if colors:
            color_cycle = list(colors)
            while len(color_cycle) < len(all_data):
                color_cycle += colors
        else:
            color_cycle = ["#BBBBBB", "#999999", "#777777", "#555555"]

        for i, (patch, color) in enumerate(zip(box['boxes'], color_cycle)):
            patch.set_facecolor(color)

            if hatch_patterns:
                patch.set_hatch(hatch_patterns[i % len(hatch_patterns)])

        ax.set_title(prefix)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, ha='center')

        control_idx = None
        for i, lbl in enumerate(labels):
            if lbl.lower() in control_labels or any(lbl.lower() == lm.lower() for lm in control_labels):
                control_idx = i
                break

        if control_idx is not None:
            y_max = max([np.max(d) for d in all_data if len(d) > 0])
            y_offset = (y_max - np.min([np.min(d) for d in all_data if len(d) > 0])) * 0.1
            current_height = y_max + y_offset

            control_data = all_data[control_idx]

            for i, (lbl, data) in enumerate(zip(labels, all_data)):
                if i == control_idx:
                    continue

                # Test de normalité
                normal_ctrl = shapiro(control_data)[1] > 0.05
                normal_test = shapiro(data)[1] > 0.05

                # Test d'homoscédasticité
                homo = levene(control_data, data)[1] > 0.05

                # Choix du test
                if normal_ctrl and normal_test:
                    if homo:
                        stat, p = ttest_ind(control_data, data, equal_var=True)
                    else:
                        stat, p = ttest_ind(control_data, data, equal_var=False)
                else:
                    stat, p = mannwhitneyu(control_data, data, alternative="two-sided")

                # Déterminer nombre d’étoiles
                if p < 0.0005:
                    stars = '***'
                elif p < 0.005:
                    stars = '**'
                elif p < 0.05:
                    stars = '*'
                else:
                    stars = 'n.s.'

                if stars:
                    x1, x2 = control_idx + 1, i + 1
                    ax.plot([x1, x1, x2, x2],
                            [current_height, current_height + y_offset * 0.1, current_height + y_offset * 0.1, current_height],
                            lw=1.2, color='black')
                    ax.text((x1 + x2) / 2, current_height + y_offset * 0.005, stars,
                            ha='center', va='bottom', fontsize=12)
                    current_height += y_offset * 0.7

    for j in range(len(ordered_prefixes), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

