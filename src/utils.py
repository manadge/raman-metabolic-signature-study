# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 CEA and ANSES

# -------------------------------------------------------------
#                           IMPORTS
# -------------------------------------------------------------

from __future__ import annotations

import pandas as pd
from scipy.integrate import trapezoid
from typing import Dict


# -------------------------------------------------------------
#                 RAW DATA MANAGEMENT HELPER
# -------------------------------------------------------------


def split_df_by_metadata(
    df: pd.DataFrame,
    split_by: list,
    fixed_filters: dict = None,
    meta_cols: int = 6,
    print_summary: bool = True
):
    """
    Split a DataFrame into sub-DataFrames according to metadata columns.

    Rows of ``df`` are first filtered based on ``fixed_filters``, then
    grouped by the columns listed in ``split_by``. For each group, the
    leading metadata columns are dropped and the remaining data is
    stored in a dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame whose first ``meta_cols`` columns are metadata
        and whose remaining columns hold the data (e.g. spectra).
    split_by : list of str
        Names of the metadata columns used to group the rows. Each
        dictionary key is built by joining the group values with "_",
        in the order of ``split_by`` (e.g. "A_1").
    fixed_filters : dict, optional
        Mapping of metadata column name to a single value or to a list
        of values. Only rows matching all the filters are kept.
    meta_cols : int, default=6
        Number of leading metadata columns. These columns are excluded
        from the returned sub-DataFrames.
    print_summary : bool, default=True
        If True, print the number of sub-DataFrames created and a preview
        of their keys.

    Returns
    -------
    dict of str to pd.DataFrame
        Mapping of each group key to the corresponding sub-DataFrame
        (original index and columns preserved, minus the metadata
        columns). Returns an empty dict if the filters match no row.
    """

    df_filtered = df.copy()

    if fixed_filters:
        for key, value in fixed_filters.items():
            if isinstance(value, (list, tuple, set)):
                df_filtered = df_filtered[df_filtered[key].isin(value)]
            else:
                df_filtered = df_filtered[df_filtered[key] == value]

    if df_filtered.empty:
        print("No sample found with the filter:", fixed_filters)
        return {}

    groups = df_filtered.groupby(split_by)

    dict_dfs = {}
    for keys, subdf in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)

        mapping = dict(zip(split_by, keys))

        key = "_".join(str(v) for v in mapping.values())
        
        dict_dfs[key] = subdf.iloc[:, meta_cols:]

    if print_summary:
        print(f"{len(dict_dfs)} sub-DataFrames created")
        print("Keys :", list(dict_dfs.keys())[:10], "...")
    
    return dict_dfs


# -------------------------------------------------------------
#               PROCESSING HELPER FUNCTIONS
# -------------------------------------------------------------


def select_windows(df, windows):

    """
    Select columns of a DataFrame that fall within one or more numeric windows.

    The DataFrame is transposed so that column names are treated as numeric
    values (e.g. wavenumbers, wavelengths, energies). For each window
    ``(start, end)`` given in ``windows``, the columns whose numeric name
    lies between ``start`` and ``end`` (inclusive) are kept. The masks of
    all windows are combined, so a column is kept if it belongs to at least
    one window.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame whose column names are numeric values (or
        convertible to float). Rows are preserved as-is.
    windows : list of tuple
        List of ``(start, end)`` pairs defining the selection intervals,
        bounds inclusive. Each bound may be a number, or the string
        "first" / "last" to refer to the smallest / largest column name
        of the DataFrame.
        (e.g. [("first", 1000), (2800, "last")])

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the columns inside at least one window,
        with the original row index preserved and the columns re-sorted in
        ascending numeric order. Returns an empty DataFrame if no column
        falls inside any window.
    """

    df_T = df.T
    col_nums = df_T.index.astype(float)

    combined_mask = pd.Series(False, index=col_nums)

    for start, end in windows:
        if start == "first":
            start_val = col_nums.min()
        else:
            start_val = float(start)

        if end == "last":
            end_val = col_nums.max()
        else:
            end_val = float(end)

        combined_mask |= (col_nums >= start_val) & (col_nums <= end_val)

    return df_T.loc[combined_mask].T


# -------------------------------------------------------------


def select_two_windows_dict(
    dict_dfs: Dict[str, pd.DataFrame],
    window1: tuple,
    window2: tuple
) -> Dict[str, pd.DataFrame]:

    """
    Apply two windows to every DataFrame of a dictionary.

    For each ``(key, df)`` pair in ``dict_dfs``, the columns of ``df`` are
    filtered twice with :func:`select_windows`, once with ``window1`` and
    once with ``window2``. The results are stored in two dictionaries
    that keep the original keys, and both are returned as a tuple.

    Parameters
    ----------
    dict_dfs : dict of str to pd.DataFrame
        Mapping of group keys to DataFrames whose column names are
        numeric values (or convertible to float), as expected by
        :func:`select_windows`.
    window1 : tuple
        First ``(start, end)`` window applied to every DataFrame.
        Each bound may be a number, or the string "first" / "last"
        (see :func:`select_windows`).
    window2 : tuple
        Second ``(start, end)`` window applied to every DataFrame,
        same conventions as ``window1``.

    Returns
    -------
    tuple of (dict of str to pd.DataFrame, dict of str to pd.DataFrame)
        ``(window1_dict, window2_dict)``, where ``window1_dict`` maps
        each original key to the DataFrame restricted to ``window1``,
        and ``window2_dict`` to the DataFrame restricted to ``window2``.
        Keys with an empty selection are preserved (mapped to an empty
        DataFrame).
    """
    
    window1_dict = {}
    window2_dict = {}

    for key, df in dict_dfs.items():
        df_window1 = select_windows(df, [window1])
        df_window2 = select_windows(df, [window2])

        window1_dict[key] = df_window1
        window2_dict[key] = df_window2

    return window1_dict, window2_dict


# -------------------------------------------------------------


def crop_spectra_dict(
    spectra_dict: dict[str, pd.DataFrame],
    x_min: float,
    x_max: float
) -> dict[str, pd.DataFrame]:

    """
    Crop the spectra of a dictionary of DataFrames to a common x-range.

    For each ``(key, df)`` pair in ``spectra_dict``, the columns of
    ``df`` are interpreted as numeric x-values (e.g. wavenumbers) and
    only those lying between ``x_min`` and ``x_max`` (inclusive) are
    kept. Rows are left unchanged, and the original column order is
    preserved (no sorting is performed).

    Parameters
    ----------
    spectra_dict : dict of str to pd.DataFrame
        Mapping of group keys to DataFrames whose column names are
        numeric x-values, or values convertible to float (e.g. strings
        such as "1200.5").
    x_min : float
        Lower bound of the crop range, inclusive.
    x_max : float
        Upper bound of the crop range, inclusive.

    Returns
    -------
    dict of str to pd.DataFrame
        Mapping of each original key to the cropped DataFrame. Keys
        whose DataFrame has no column inside the range are preserved
        (mapped to an empty DataFrame).
    """
    
    cropped = {}
    for key, df in spectra_dict.items():
        try:
            cols = df.columns.astype(float)
        except Exception:
            cols = df.columns.map(float)
        mask = (cols >= x_min) & (cols <= x_max)
        cropped[key] = df.loc[:, mask]
    return cropped


# -------------------------------------------------------------
#             PROCESSING CALCULATION HELPERS
# -------------------------------------------------------------


def calculate_area_ratio_dicts(
    dict1: Dict[str, pd.DataFrame],
    dict2: Dict[str, pd.DataFrame]
) -> Dict[str, pd.Series]:

    """
    Compute, for each spectrum, the ratio of the first area to the total area.

    For each key present in both dictionaries, each row of ``dict1[key]``
    and ``dict2[key]`` is treated as a spectrum: the area under the curve
    is estimated with the trapezoidal rule, using the numeric column
    names as x-values. The ratio is defined as
    ``area1 / (area1 + area2)``, i.e. the fraction of the total area
    contributed by the first window.

    Parameters
    ----------
    dict1 : dict of str to pd.DataFrame
        Mapping of group keys to DataFrames holding the spectra of the
        first region. Column names must be numeric x-values. The row
        index identifies individual spectra.
    dict2 : dict of str to pd.DataFrame
        Mapping of group keys to DataFrames holding the spectra of the
        second region, with the same keys and row indices as ``dict1``.

    Returns
    -------
    dict of str to pd.Series
        Mapping of each group key to a Series of ratios, indexed like
        the rows of ``dict1[key]``. If ``area1 + area2`` is zero for a
        spectrum, the ratio is set to 0 instead of raising an error.
    """
    
    ratio_dict = {}

    for key in dict1.keys():
        df1 = dict1[key]
        df2 = dict2[key]

        ratios = []

        for idx in df1.index:
            y1 = df1.loc[idx].values
            y2 = df2.loc[idx].values

            x1 = df1.columns.astype(float).values
            x2 = df2.columns.astype(float).values

            area1 = trapezoid(y1, x1)
            area2 = trapezoid(y2, x2)

            ratio = area1 / (area1 + area2) if (area1 + area2) != 0 else 0
            ratios.append(ratio)

        ratio_dict[key] = pd.Series(ratios, index=df1.index)

    return ratio_dict



# -------------------------------------------------------------


def combiner_groupes(dico_df, liste_unElement, new_name, sep="_"):

    """
    Merge DataFrames that share the same second key element into one.

    Keys of ``dico_df`` are expected to have the form
    ``"<element>_<element2>"`` (split by ``sep``). For each distinct
    ``element2`` whose first part belongs to ``liste_unElement``, all
    corresponding DataFrames are concatenated along the row axis into
    a single DataFrame, stored under the new key
    ``f"{new_name}_{element2}"``. Keys whose first element is not in
    ``liste_unElement`` are dropped from the output.

    Parameters
    ----------
    dico_df : dict of str to pd.DataFrame
        Mapping of keys of the form ``"<element>_<element2>"`` to
        DataFrames. Each key must contain exactly two parts once split
        by ``sep``, otherwise a ValueError is raised.
    liste_unElement : list of str
        First key elements whose DataFrames should be merged. Keys
        starting with other elements are ignored.
    new_name : str
        First part of the new keys, e.g. ``"fusion"`` yields keys like
        ``"fusion_<element2>"``.
    sep : str, default="_"
        Separator used to split the keys and to build the new ones.

    Returns
    -------
    dict of str to pd.DataFrame
        Mapping of ``f"{new_name}_{element2}"`` to the concatenated
        DataFrame of all matching groups (row-wise concatenation,
        original indices preserved). Groups containing a single
        DataFrame are returned unchanged (concat of one element).

    Raises
    ------
    ValueError
        If a key of ``dico_df`` does not contain exactly two parts once
        split by ``sep``.

    Examples
    --------
    >>> dico_df = {'elA_tr1': df1, 'elA_tr2': df2,
    ...            'elB_tr1': df3, 'elB_tr2': df4}
    >>> combiner_groupes(dico_df, ['elA', 'elB'], 'fusion')
    {'fusion_tr1': concat(df1, df3), 'fusion_tr2': concat(df2, df4)}
    """

    regroupement = {}

    for key, df in dico_df.items():
        parts = key.split(sep)
        if len(parts) != 2:
            raise ValueError(f"Invalid key: {key} (must be 'unElement_element2')")
        unElement, element2 = parts

        if unElement in liste_unElement:
            regroupement.setdefault(element2, []).append(df)

    nouveau_dict = {
        f"{new_name}_{element2}": pd.concat(df_list, axis=0)
        for element2, df_list in regroupement.items()
    }

    return nouveau_dict

