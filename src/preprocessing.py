# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 CEA and ANSES

# -------------------------------------------------------------
#                           IMPORTS
# -------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve



# -------------------------------------------------------------
#                  PREPROCESSING FUNCTIONS
# -------------------------------------------------------------

# ----------------- AsLS Baseline removal --------------------------


def als_baseline_cor_df(df: pd.DataFrame, lam: float = 1e4, p: float = 0.001, niter: int = 10, return_baseline: bool = False):
    """
    Correct the baseline of a spectra DataFrame using the ALS
    (Asymmetric Least Squares) method.

    Parameters
    ----------
    df : pd.DataFrame
        Spectra DataFrame of shape (n_spectra, n_pixels).
        Each row is a spectrum, each column is a pixel.

    lam : float
        Regularization constraint (larger -> smoother baseline).

    p : float
        Asymmetry weight (smaller -> lower baseline).

    niter : int
        Number of iterations used to optimize the baseline.

    return_baseline : bool
        If True, also return the baseline.

    zero_ref : str, optional
        Zero-recentering method:
        - None : no centering
        - "min" : subtract the minimum of each corrected spectrum
        - "q1" : subtract the 1st percentile (more robust than min)
        - "clip" : set all negative values to 0

    Returns
    -------
    corrected : pd.DataFrame
        The baseline-corrected spectra.
    baseline : pd.DataFrame, optional
        The estimated baseline, only if return_baseline is True.
    """
    sp = df.to_numpy(copy=True)
    sp = np.array(sp, ndmin=2) 
    n_spectra, sp_length = sp.shape

    diag = sparse.diags([1, -2, 1], [0, -1, -2], shape=(sp_length, sp_length - 2))
    diag = lam * diag.dot(diag.transpose())

    baseline = np.zeros_like(sp)

    for n in range(n_spectra):
        w = np.ones(sp_length)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, sp_length, sp_length)
            Z = W + diag
            baseline[n] = spsolve(Z, w * sp[n])
            w = p * (sp[n] > baseline[n]) + (1 - p) * (sp[n] < baseline[n])

    corrected = sp - baseline

    corrected_df = pd.DataFrame(corrected, index=df.index, columns=df.columns)
    baseline_df = pd.DataFrame(baseline, index=df.index, columns=df.columns)

    if return_baseline:
        return corrected_df, baseline_df
    return corrected_df


# ------------- AsLS Baseline removal with baseline 0 correction ------------


def als_baseline_zero_cor_df(
    df: pd.DataFrame,
    lam: float = 1e4,
    p: float = 0.001,
    niter: int = 10,
    return_baseline: bool = False,
    zero_ref: str = None
):
    """
    Correct the baseline of each spectrum in a DataFrame using ALS.

    Applies Asymmetric Least Squares (AsLS) baseline estimation to every
    row (spectrum) of ``df``, then optionally re-centers each corrected
    spectrum to zero, row by row, according to ``zero_ref``.

    Parameters
    ----------
    df : pd.DataFrame
        Spectra DataFrame of shape (n_spectra, n_pixels): one spectrum
        per row, one pixel per column.
    lam : float, default=1e4
        Regularization strength (second-difference penalty). Larger
        values yield a smoother baseline.
    p : float, default=0.001
        Asymmetry weight. Smaller values push the baseline lower, beneath
        most of the peaks.
    niter : int, default=10
        Number of ALS iterations performed on each spectrum.
    return_baseline : bool, default=False
        If True, also return the estimated baseline.
    zero_ref : {None, "min", "q1", "clip"}, optional
        Selective zero-reset applied to each corrected spectrum:
        - None : no re-centering (default)
        - "min" : subtract the minimum of each corrected spectrum
        - "q1" : subtract the 1st percentile of each corrected spectrum,
            then clip any remaining negative values to 0
        - "clip" : clip all negative values of each corrected spectrum to 0

    Returns
    -------
    corrected : pd.DataFrame
        Baseline-corrected spectra, with the same index and columns as
        ``df``.
    baseline : pd.DataFrame, optional
        Estimated baseline, returned only if ``return_baseline=True``.

    Raises
    ------
    ValueError
        If ``zero_ref`` is not None, "min", "q1" or "clip".
    """

    sp = df.to_numpy(copy=True)
    sp = np.array(sp, ndmin=2)
    n_spectra, sp_length = sp.shape

    diag = sparse.diags([1, -2, 1], [0, -1, -2], shape=(sp_length, sp_length - 2))
    diag = lam * diag.dot(diag.transpose())

    baseline = np.zeros_like(sp)

    for n in range(n_spectra):
        w = np.ones(sp_length)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, sp_length, sp_length)
            Z = W + diag
            baseline[n] = spsolve(Z, w * sp[n])
            w = p * (sp[n] > baseline[n]) + (1 - p) * (sp[n] < baseline[n])

    corrected = sp - baseline

    if zero_ref is not None:
        for i in range(n_spectra):
            if zero_ref == "min":
                offset = np.min(corrected[i])
                corrected[i] -= offset
            elif zero_ref == "q1":
                q = np.percentile(corrected[i], 1)
                corrected[i] -= q
                corrected[i][corrected[i] < 0] = 0
            elif zero_ref == "clip":
                corrected[i][corrected[i] < 0] = 0
            else:
                raise ValueError("zero_ref must be None, 'min', 'q1' ou 'clip'.")

    corrected_df = pd.DataFrame(corrected, index=df.index, columns=df.columns)
    baseline_df = pd.DataFrame(baseline, index=df.index, columns=df.columns)

    if return_baseline:
        return corrected_df, baseline_df
    return corrected_df


# ------------- Improved Modified Polynomial Baseline Removal ------------


def imodpoly_baseline_cor_df(
    df: pd.DataFrame,
    poly_order: int = 3,
    niter: int = 25,
    k: float = 1.0,
    abs_threshold: float | None = None,
    use_mad: bool = True,
    tol: float = 1e-6,
    return_baseline: bool = False,
):
    """
    Baseline correction using IModPoly (Improved Modified Polynomial)[1]_.

    Principle:
    - Fits a polynomial of order `poly_order` to a "clipped" signal:
        y_clipped = min(y, baseline_prev + threshold)
    - The threshold is either k * sigma (sigma = MAD or std of the
        residuals), or `abs_threshold` (absolute).
    - Repeats until convergence or `niter`.

    Parameters
    ----------
    df : DataFrame (n_spectra, n_points)
        Rows = spectra, columns = abscissas (pixels / Raman shift). If the
        columns are numeric, they are used as x.
    poly_order : int
        Polynomial order (typically 2-5).
    niter : int
        Number of IModPoly iterations.
    k : float
        Multiplier of the robust standard deviation (MAD) of the residuals
        used for clipping (ignored if abs_threshold is provided).
    abs_threshold : float | None
        Absolute clipping threshold. If provided,
        y_clipped = min(y, baseline_prev + abs_threshold).
    use_mad : bool
        If True, sigma = 1.4826 * MAD(residuals), otherwise sigma is the
        regular standard deviation.
    tol : float
        Stopping criterion based on the relative change of the baseline.
    return_baseline : bool
        If True, also return the estimated baseline.

    Returns
    -------
    corrected_df [, baseline_df]

    Notes
    -----
    This function is a reimplementation of the ``IModPoly`` algorithm from
    the `BaselineRemoval <https://pypi.org/project/BaselineRemoval/>`_
    package, adapted to process a whole DataFrame of spectra (one spectrum
    per row) instead of a single 1D array, and to optionally re-center each
    corrected spectrum to zero via ``zero_ref``.

    References
    ----------
    .. [1] Zhao, J., Lui, H., McLean, D. I., & Zeng, H. (2007).
    Automated autofluorescence background subtraction algorithm for
    biomedical Raman spectroscopy. *Applied Spectroscopy*, 61(11),
    1225-1232. https://doi.org/10.1366/000370207782597003
    """

    if df.shape[1] <= poly_order:
        raise ValueError("poly_order doit être strictement inférieur au nombre de points du spectre.")

    cols = df.columns.to_numpy()
    if np.issubdtype(cols.dtype, np.number) and np.all(np.isfinite(cols)):
        x = cols.astype(float)
    else:
        x = np.arange(df.shape[1], dtype=float)

    sp = df.to_numpy(copy=True).astype(float)
    n_spectra, sp_length = sp.shape

    x_mean = x.mean()
    x_scale = x.std() if x.std() > 0 else 1.0
    x_norm = (x - x_mean) / x_scale

    baseline = np.zeros_like(sp)

    for i in range(n_spectra):
        y = sp[i]

        coeffs = np.polyfit(x_norm, y, deg=poly_order)
        base = np.polyval(coeffs, x_norm)

        for _ in range(niter):
            resid = y - base

            if abs_threshold is not None:
                thr = abs_threshold
            else:
                if use_mad:
                    mad = np.median(np.abs(resid - np.median(resid)))
                    sigma = 1.4826 * mad
                else:
                    sigma = resid.std(ddof=1)
                if not np.isfinite(sigma) or sigma == 0:
                    sigma = 1e-12
                thr = k * sigma

            y_clip = np.minimum(y, base + thr)

            coeffs_new = np.polyfit(x_norm, y_clip, deg=poly_order)
            base_new = np.polyval(coeffs_new, x_norm)

            denom = np.maximum(1.0, np.linalg.norm(base))
            if np.linalg.norm(base_new - base) / denom < tol:
                base = base_new
                break
            base = base_new

        baseline[i] = base

    corrected = sp - baseline

    corrected_df = pd.DataFrame(corrected, index=df.index, columns=df.columns)
    baseline_df = pd.DataFrame(baseline, index=df.index, columns=df.columns)

    if return_baseline:
        return corrected_df, baseline_df
    return corrected_df


# ------------- Cosmic spikes removal, mask method ------------


def cosmic_filter_df(df: pd.DataFrame, width: int = 3, threshold: float = 11.0) -> pd.DataFrame:

    """
    Remove cosmic spikes from a spectra DataFrame.

    A sliding window of ``width`` pixels is moved along the pixel axis.
    For each window position, a boolean mask (``cosmspike_mask``) flags
    the spectra whose value at the central pixel deviates from the window
    median by more than ``threshold * std``. Only flagged pixels are
    replaced, by linear interpolation between the values at the two window
    edges; all other pixels are left unchanged.


    Notes
    -----
    - Each row of the DataFrame is treated as one spectrum
    (shape: n_spectra, n_pixels).
    - Does not work if the DataFrame contains a single spectrum (1 row).
    - Detected cosmic peaks are replaced by linear interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        Spectra DataFrame of shape (n_spectra, n_pixels).
    width : positive odd int, default=3
        Size of the sliding window (in pixels).
    threshold : positive float, default=11.0
        Detection threshold for cosmic peaks. A pixel is flagged when its
        value exceeds ``median + threshold * std`` of the window.

    Returns
    -------
    pd.DataFrame
        Cosmic-ray-corrected spectra.
    """

    if df.shape[0] == 1:
        raise ValueError("Does not work if the DataFrame contains a single spectrum")

    sp = df.to_numpy(copy=True)
    filtered_spectrum = sp.copy()
    half_width = width // 2

    for i in range(half_width, sp.shape[1] - half_width): 
        window = sp[:, i - half_width:i + half_width + 1]
        median_value = np.median(window)
        std_value = np.std(window)

        abs_diff = np.abs(sp[:, i] - median_value)
        cosmspike_mask = abs_diff > threshold * std_value

        for n in range(sp.shape[0]):
            if cosmspike_mask[n]:
                left_index = max(i - half_width - 1, 0)
                right_index = min(i + half_width + 1, sp.shape[1] - 1)
                left_value = sp[n, left_index]
                right_value = sp[n, right_index]
                filtered_spectrum[n, i] = np.interp(i, [left_index, right_index], [left_value, right_value])

    return pd.DataFrame(filtered_spectrum, index=df.index, columns=df.columns)


# ------------- Cosmic spikes removal, multispectral method ------------


def remove_cosmics_multispectra(df: pd.DataFrame, z_thresh: float = 8.0) -> pd.DataFrame:
    """
    Detect and remove cosmic spikes by cross-spectrum comparison.

    Useful when several acquisitions of the same sample are available.
    Each pixel of each spectrum is compared to the median of all spectra
    at that pixel (column-wise, across rows): a boolean mask flags every
    point whose deviation from this per-pixel median exceeds
    ``z_thresh`` standard deviations[1]_. Flagged points are then replaced
    by linear interpolation over the unmasked points of the same
    spectrum.

    Parameters
    ----------
    df : pd.DataFrame
        Spectra DataFrame of shape (n_spectra, n_pixels): rows are
        acquisitions, columns are Raman pixels.
    z_thresh : float, default=8.0
        Detection threshold, in units of standard deviation. Point
        (i, j) is flagged when ``abs(arr[i, j] - med[j]) >
        z_thresh * std[j]``, where ``med[j]`` and ``std[j]`` are the
        median and standard deviation across all spectra at pixel j.

    Returns
    -------
    pd.DataFrame
        Cosmic-spike-corrected spectra, with the same index and columns
        as ``df``.

    Notes
    -----
    - The mask has shape (n_spectra, n_pixels) and is computed in one
      shot: ``mask = abs(arr - med) > z_thresh * std``, with ``med`` and
      ``std`` taken along axis 0 (across spectra, per pixel).
    - Only flagged points are modified. For each spectrum containing at
      least one flagged point, the corrected values are obtained by
      ``np.interp`` over that spectrum's unmasked pixels; points outside
      the range of valid pixels take the nearest edge value (np.interp
      clamps at the boundaries).
    - With a single spectrum (one row), ``std`` is 0 everywhere and no
      point is flagged, so no correction is applied.

    References
    ----------
    .. [1] Whitaker, D. A., & Hayes, K. (2018). A simple algorithm for
    despiking Raman spectra. *Chemometrics and Intelligent Laboratory
    Systems*, 179, 82-84. https://doi.org/10.1016/j.chemolab.2018.06.009
    """

    arr = df.to_numpy(copy=True)
    med = np.median(arr, axis=0)
    std = np.std(arr, axis=0)
    mask = np.abs(arr - med) > z_thresh * std
    corrected = arr.copy()

    for i in range(arr.shape[0]):
        if np.any(mask[i, :]):
            idx = np.arange(arr.shape[1])
            corrected[i, :] = np.interp(idx, idx[~mask[i, :]], arr[i, ~mask[i, :]])

    return pd.DataFrame(corrected, index=df.index, columns=df.columns)


# ------------- Normalization methods selection ------------


def spectral_normalization_df(df: pd.DataFrame, norm: str = 'max') -> pd.DataFrame:
    """
    Normalize a spectra DataFrame, row by row (per spectrum).

    Parameters
    ----------
    df : pd.DataFrame
        Spectra DataFrame of shape (n_spectra, n_pixels): each row is
        one spectrum, each column one pixel.
    norm : {'l2', 'l1', 'max', 'minmax', 'snv'}, default='max'
        Normalization method, applied independently to each row:
        - 'l2' : divide by the Euclidean norm,
          ``x / sqrt(sum(x**2))`` (L2 norm = 1)
        - 'l1' : divide by the sum of absolute values,
          ``x / sum(abs(x))`` (L1 norm = 1)
        - 'max' : divide by the row maximum, ``x / max(x)`` (max = 1)
        - 'minmax' : rescale to [0, 1], ``(x - min(x)) / (max(x) - min(x))``
        - 'snv' : Standard Normal Variate, ``(x - mean(x)) / std(x)``
          (mean = 0, std = 1; std computed with ddof=0)

    Returns
    -------
    pd.DataFrame
        Normalized spectra, with the same index and columns as ``df``.

    Raises
    ------
    ValueError
        If ``norm`` is not one of 'l2', 'l1', 'max', 'minmax', 'snv'.

    Notes
    -----
    - All statistics (max, min, mean, std) are computed per row
      (``axis=1``), so each spectrum is normalized independently.
    - A flat (constant) spectrum causes a division by zero for 'minmax'
      (max == min) and 'snv' (std == 0), and for 'max'/'l1'/'l2' if the
      spectrum is all zeros; the result then contains NaN/inf values.
    """
    sp = df.to_numpy(copy=True)
    sp = np.array(sp, ndmin=2)

    sp_max = np.max(sp, axis=1, keepdims=True)
    sp_min = np.min(sp, axis=1, keepdims=True)
    sp_mean = np.mean(sp, axis=1, keepdims=True)
    sp_std = np.std(sp, axis=1, keepdims=True)

    if norm == 'max':
        normed = sp / sp_max
    elif norm == 'l1':
        normed = sp / np.sum(np.abs(sp), axis=1, keepdims=True)
    elif norm == 'l2':
        normed = sp / np.sqrt(np.sum(sp**2, axis=1, keepdims=True))
    elif norm == 'minmax':
        normed = (sp - sp_min) / (sp_max - sp_min)
    elif norm == 'snv':
        normed = (sp - sp_mean) / sp_std
    else:
        raise ValueError(f"{norm} is not a valid choice. possible choices: 'l2', 'l1', 'max', 'minmax', 'snv'")

    return pd.DataFrame(normed, index=df.index, columns=df.columns)

