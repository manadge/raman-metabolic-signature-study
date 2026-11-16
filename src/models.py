# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 CEA and ANSES

# -------------------------------------------------------------
#                           IMPORTS
# -------------------------------------------------------------

from __future__ import annotations
from IPython.display import display

import pandas as pd
import numpy as np
import os, pickle
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import savgol_filter, find_peaks

from typing import Sequence,  Any, Dict, Optional
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# -------------------------------------------------------------
#                      PCA to dictionnary
# -------------------------------------------------------------

def pca_dict(
    dico_df: dict,
    pc_pair: list = [0, 1],
    n_components_total: int = 10,
    meta_index: int = 0,
    pattern_sep: str = '_',
    tick_step: int = 1
):
    
    """
    Performs a PCA on a dictionary of DataFrames.
    Each PCA is performed for a group of DataFrames sharing the same meta_index value in their key.

    Returns a dictionary containing the PCA models and the associated data for each group.

    Parameters
    ----------
    dico_df : dict
        Dictionary of DataFrames (same columns expected)
    pc_pair : list, default [0, 1]
        Pairs of principal components to display (e.g., [0, 1] or [1, 2])
    n_components_total : int, default 10
        Total number of components to compute in the PCA
    meta_index : int, default 0
        Position of the key element used to group the DataFrames (with pattern_sep)
    pattern_sep : str, default '_'
        Separator used to split the keys
    tick_step : int, default 1
        Step for the X axis display on the loadings bar plots

    Returns
    -------
    dict
        Dictionary {group_key: {"pca": PCA_model, "data": DataFrame used, "pcs": scores, "explained_var": array, "loadings": DataFrame}}
    """
    
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    
    results = {}
    
    grouped = {}
    for key, df in dico_df.items():
        parts = key.split(pattern_sep)
        group_key = parts[meta_index]
        grouped.setdefault(group_key, []).append((key, df))
    
    for group_key, df_list in grouped.items():
        print(f"\n=== PCA by group: {group_key} ===")

        dfs = [df for _, df in df_list]
        data = pd.concat(dfs, ignore_index=True)
        
        subgroup_labels = []
        for key, df in df_list:
            n = len(df)
            parts = key.split(pattern_sep)
            subgroup_label = pattern_sep.join([p for i, p in enumerate(parts) if i != meta_index])
            subgroup_labels.extend([subgroup_label] * n)
        subgroup_labels = np.array(subgroup_labels)
        
        data_num = data.select_dtypes(include=[np.number]).dropna(axis=0)
        valid_idx = data_num.index
        subgroup_labels = subgroup_labels[valid_idx]
        
        pca = PCA(n_components=n_components_total)
        pcs = pca.fit_transform(data_num)
        explained_var = pca.explained_variance_ratio_ * 100

        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(subgroup_labels)
        color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)}

        for label in unique_labels:
            idx = subgroup_labels == label
            plt.scatter(
                pcs[idx, pc_pair[0]],
                pcs[idx, pc_pair[1]],
                color=color_map[label],
                label=label,
                alpha=0.7,
                edgecolors='k',
                s=40
            )

        plt.xlabel(f"PC{pc_pair[0]+1} ({explained_var[pc_pair[0]]:.1f}%)")
        plt.ylabel(f"PC{pc_pair[1]+1} ({explained_var[pc_pair[1]]:.1f}%)")
        plt.title(f"PCA — Group {group_key}")
        plt.legend(title="Subgroup", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_components_total)],
            index=data_num.columns
        )

        for pc in pc_pair:
            plt.figure(figsize=(10, 4))
            loadings[f"PC{pc+1}"].plot(kind='bar', color="#1f77b4")
            plt.title(f"Variable importances — PC{pc+1} ({explained_var[pc]:.1f}%) — {group_key}")
            plt.ylabel("Loading")
            plt.grid(axis='y')

            ticks = np.arange(0, len(loadings.index), tick_step)
            plt.xticks(ticks, loadings.index[ticks], rotation=45, ha='right')

            plt.tight_layout()
            plt.show()

        results[group_key] = {
            "pca": pca,
            "data": data_num,
            "pcs": pcs,
            "explained_var": explained_var,
            "loadings": loadings,
            "subgroup_labels": subgroup_labels
        }

    return results


# -------------------------------------------------------------
#                  Classification Functions
# -------------------------------------------------------------


def gridsearch_classif_dict_v2(
    dico_df: dict,
    meta_index: int = 0,
    pattern_sep: str = '_',
    test_size: float = 0.2,
    random_state: int = 42,
    scoring: str = "accuracy",
    cv: int = 5,
    exclude_labels: Optional[Sequence[str]] = None
):
    
    """
    Performs a GridSearchCV for several classifiers on each group of the dictionary.
    Compares their performances, displays the confusion matrices and produces a global summary.

    Parameters
    ----------
    dico_df : dict
        Dictionary of DataFrames
    meta_index : int
        Position of the key element used to group the DataFrames
    pattern_sep : str
        Key separator
    test_size : float
        Size of the test set
    random_state : int
        Random seed
    scoring : str
        Main metric used to select the best model
    cv : int
        Number of folds for cross-validation
    exclude_labels : list[str]
        Optional list of sub-groups (labels) to exclude from the analyses.

    Returns
    -------
    dict
        Full results for each group + global summary.
    """

    results = {}
    global_summary = []

    model_grid = {
        "SVM": (SVC(), {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", 0.1]   
        }),

        "RandomForest": (RandomForestClassifier(random_state=random_state, n_jobs=-1), {
            "n_estimators": [200],             
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 5],
            "max_features": ["sqrt", 0.5]
        }), 

        "LogisticRegression": (
                OneVsRestClassifier(
                    LogisticRegression(max_iter=2000, random_state=random_state)
                ),
                {
                    "estimator__solver": ["liblinear"],
                    "estimator__penalty": ["l2", "l1"],
                    "estimator__C": [0.1, 1, 10],
                },
        ), 

        "KNN": (KNeighborsClassifier(), {
            "n_neighbors": [3, 7, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2]  
        }), 

        "DecisionTree": (DecisionTreeClassifier(random_state=random_state), {
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 5],
            "criterion": ["gini", "entropy"]
        }) 
    }

    grouped = {}
    for key, df in dico_df.items():
        parts = key.split(pattern_sep)
        group_key = parts[meta_index]
        grouped.setdefault(group_key, []).append((key, df))

    for group_key, df_list in grouped.items():
        print(f"\nGridSearch for the group: {group_key}")

        dfs = [df for _, df in df_list]
        data = pd.concat(dfs, ignore_index=True)

        subgroup_labels = []
        for key, df in df_list:
            n = len(df)
            parts = key.split(pattern_sep)
            subgroup_label = pattern_sep.join([p for i, p in enumerate(parts) if i != meta_index])
            subgroup_labels.extend([subgroup_label] * n)
        subgroup_labels = np.array(subgroup_labels)

        if exclude_labels:
            mask = ~np.isin(subgroup_labels, exclude_labels)
            data = data.loc[mask].reset_index(drop=True)
            subgroup_labels = subgroup_labels[mask]

        data_num = data.select_dtypes(include=[np.number]).dropna(axis=0)
        valid_idx = data_num.index
        subgroup_labels = subgroup_labels[valid_idx]

        if len(np.unique(subgroup_labels)) < 2:
            print(f"Not enough remaining classes after filtering {group_key}, group ignored.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            data_num, subgroup_labels, test_size=test_size, random_state=random_state, stratify=subgroup_labels
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_scores = []
        best_models = {}

        for model_name, (model, param_grid) in model_grid.items():
            print(f"GridSearch for {model_name}...")
            grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
            grid.fit(X_train_scaled, y_train)
            y_pred = grid.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            rec = recall_score(y_test, y_pred, average="macro")
            prec = precision_score(y_test, y_pred, average="macro")

            model_scores.append({
                "Group": group_key,
                "Model": model_name,
                "Best params": grid.best_params_,
                "Accuracy": acc,
                "F1": f1,
                "Recall": rec,
                "Precision": prec
            })

            best_models[model_name] = (grid.best_estimator_, grid.best_score_, y_pred)

        df_perf = pd.DataFrame(model_scores).sort_values(by="Accuracy", ascending=False)
        display(df_perf)

        best_row = df_perf.iloc[0]
        best_model_name = best_row["Model"]
        best_model = best_models[best_model_name][0]
        best_pred = best_models[best_model_name][2]

        print(f"\n Best model: {best_model_name}")
        print(f"Accuracy = {best_row['Accuracy']:.3f}, F1 = {best_row['F1']:.3f}")

        conf_mat = confusion_matrix(y_test, best_pred, labels=np.unique(subgroup_labels))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=np.unique(subgroup_labels))
        disp.plot(cmap="Blues", values_format='d')
        plt.grid(False)
        plt.title(f"Confusion matrix — {group_key} ({best_model_name})")
        plt.tight_layout()
        plt.show()

        results[group_key] = {
            "best_model_name": best_model_name,
            "best_model": best_model,
            "performance_table": df_perf,
            "conf_matrix": conf_mat,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": best_pred
        }

        global_summary.extend(model_scores)

    if global_summary:
        df_global = pd.DataFrame(global_summary)
        df_summary = (
            df_global.groupby("Model")[["Accuracy", "F1", "Recall", "Precision"]]
            .mean()
            .sort_values(by="Accuracy", ascending=False)
        )

        print("\n Full overview of the models")
        display(df_summary.style.background_gradient(cmap="Blues", axis=None).format("{:.3f}"))

        results["global_summary"] = df_summary

    return results


# -------------------------------------------------------------


def gridsearch_logreg_dict(
    dico_df: dict,
    meta_index: int = 0,
    pattern_sep: str = '_',
    test_size: float = 0.2,
    random_state: int = 42,
    scoring: str = "accuracy",
    cv: int = 5,
    exclude_labels: Optional[Sequence[str]] = None,
    smooth_window: int = 5,
    smooth_poly: int = 2,
    tick_spacing_x: int = 5,
    tick_spacing_y: float = 0.1
):
    
    """
    GridSearch with logistic regression only.

    Tests several compatible solvers/penalties,
    displays a smoothed spectrum of the importances and the normalized confusion matrices (%).

    Parameters
    ----------
    dico_df : dict
        Dictionary of DataFrames
    meta_index : int, default 0
        Position of the key element used to group the DataFrames
    pattern_sep : str, default '_'
        Key separator
    test_size : float, default 0.2
        Size of the test set
    random_state : int, default 42
        Random seed
    scoring : str, default "accuracy"
        Main metric used to select the best model
    cv : int, default 5
        Number of folds for cross-validation
    exclude_labels : Sequence[str], optional
        Optional list of sub-groups (labels) to exclude from the analyses
    smooth_window : int, default 5
        Window size of the smoothing filter applied to the importance spectrum
    smooth_poly : int, default 2
        Polynomial order of the smoothing filter applied to the importance spectrum
    tick_spacing_x : int, default 5
        Spacing of ticks on the X axis of the plots
    tick_spacing_y : float, default 0.1
        Spacing of ticks on the Y axis of the plots

    Returns
    -------
    dict
        Full results for each group + global summary.
    """

    results = {}
    CACHE_DIR = "cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    global_summary = []

    # --- Grille brute ---
    param_grid_raw = {
        "C": np.logspace(-4, 4, 10),
        "penalty": ["l1", "l2", "elasticnet", None],
        "solver": ["lbfgs", "liblinear", "saga", "newton-cg"],
        "l1_ratio": [0, 0.25, 0.5, 0.75, 1],
        "class_weight": [None, "balanced"]
    }

    grouped = {}
    for key, df in dico_df.items():
        parts = key.split(pattern_sep)
        group_key = parts[meta_index]
        grouped.setdefault(group_key, []).append((key, df))

    for group_key, df_list in grouped.items():
        cache_file = os.path.join(CACHE_DIR, f"cache_logreg_{group_key}.pkl")

        if os.path.exists(cache_file):
            print(f"\nloaded from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                results[group_key] = pickle.load(f)
            continue

        print(f"\nLogistic Regression — Groupe : {group_key}")

        dfs = [df for _, df in df_list]
        data = pd.concat(dfs, ignore_index=True)

        subgroup_labels = []
        for key, df in df_list:
            n = len(df)
            parts = key.split(pattern_sep)
            subgroup_label = pattern_sep.join([p for i, p in enumerate(parts) if i != meta_index])
            subgroup_labels.extend([subgroup_label] * n)
        subgroup_labels = np.array(subgroup_labels)

        if exclude_labels:
            mask = np.isin(subgroup_labels, exclude_labels, invert=True)
            data = data.loc[mask].reset_index(drop=True)
            subgroup_labels = subgroup_labels[mask]

        data_num = data.select_dtypes(include=[np.number]).dropna(axis=0)
        valid_idx = data_num.index
        subgroup_labels = subgroup_labels[valid_idx]

        n_classes = len(np.unique(subgroup_labels))
        if n_classes < 2:
            print(f"Not enough classes for {group_key}, ignored.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            data_num, subgroup_labels, test_size=test_size,
            random_state=random_state, stratify=subgroup_labels
        )

        print("\nClass distribution:")
        classes, counts_total = np.unique(subgroup_labels, return_counts=True)
        counts_train = [np.sum(y_train == c) for c in classes]
        counts_test = [np.sum(y_test == c) for c in classes]

        for c, n_tot, n_tr, n_te in zip(classes, counts_total, counts_train, counts_test):
            perc_train = 100 * n_tr / n_tot
            perc_test = 100 * n_te / n_tot
            print(f"  Class '{c}': {n_tot} samples: "
                  f"{n_tr} train ({perc_train:.1f}%) / {n_te} test ({perc_test:.1f}%)")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        valid_params = []
        for p in ParameterGrid(param_grid_raw):
            pen, sol = p["penalty"], p["solver"]

            if n_classes > 2 and sol == "liblinear":
                continue

            valid = (
                (sol == "lbfgs" and pen in ["l2", None]) or
                (sol == "liblinear" and pen in ["l1", "l2"]) or
                (sol == "saga" and pen in ["l1", "l2", "elasticnet", None]) or
                (sol == "newton-cg" and pen in ["l2", None])
            )

            if pen != "elasticnet":
                p = {k: v for k, v in p.items() if k != "l1_ratio"}

            if valid:
                valid_params.append({k: [v] for k, v in p.items()})

        print(f"{len(valid_params)} valid combinaisons tested")

        model = LogisticRegression(max_iter=5000, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = GridSearchCV(
                model,
                param_grid=valid_params,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                error_score="raise"
            )
            grid.fit(X_train_scaled, y_train)

        y_pred = grid.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        prec = precision_score(y_test, y_pred, average="macro")

        print(f"\nBest parameters: {grid.best_params_}")
        print(f"Accuracy = {acc:.3f}, F1 = {f1:.3f}")

        conf_mat = confusion_matrix(y_test, y_pred, labels=classes, normalize="true") * 100
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=classes)
        disp.plot(cmap="Blues", values_format=".1f")
        plt.title(f"Confusion matrix (% by class) — {group_key}")
        plt.xlabel("Predicted class")
        plt.ylabel("True Class")
        plt.tight_layout()
        plt.show()

        if hasattr(grid.best_estimator_, "coef_"):
            coefs = grid.best_estimator_.coef_
            if coefs.ndim > 1 and coefs.shape[0] > 1:
                coefs = np.mean(np.abs(coefs), axis=0)
            else:
                coefs = np.abs(coefs).flatten()

            importance_df = pd.DataFrame({
                "Feature": data_num.columns,
                "Importance": coefs
            })

            y = importance_df["Importance"].values
            x = np.arange(len(y))

            if len(y) >= smooth_window:
                y_smooth = savgol_filter(y, window_length=smooth_window, polyorder=smooth_poly)
            else:
                y_smooth = y

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x, y_smooth, color="black", linewidth=1.8)
            ax.fill_between(x, y_smooth, color="black", alpha=0.05)

            ax.spines["bottom"].set_linewidth(1.5)
            ax.spines["left"].set_linewidth(1.5)
            ax.tick_params(width=1.5, length=6)

            ax.set_xticks(np.arange(0, len(x), tick_spacing_x))
            ax.set_xticklabels(importance_df["Feature"].iloc[::tick_spacing_x], rotation=90)
            y_max = np.max(y)
            ax.set_yticks(np.arange(0, y_max * 1.05, tick_spacing_y * y_max))

            ax.set_title(f"Feature importance — {group_key}")
            ax.set_xlabel("Original importances")
            ax.set_ylabel("|Coefficient|")
            plt.tight_layout()
            plt.show()

        results[group_key] = {
            "best_model": grid.best_estimator_,
            "best_params": grid.best_params_,
            "performance": {
                "Accuracy": acc,
                "F1": f1,
                "Recall": rec,
                "Precision": prec
            },
            "conf_matrix": conf_mat,
            "y_test": y_test,
            "y_pred": y_pred
        }

        with open(cache_file, "wb") as f:
            pickle.dump(results[group_key], f)

        print(f"Results saved in: {cache_file}")

        global_summary.append({
            "Group": group_key,
            "Accuracy": acc,
            "F1": f1,
            "Recall": rec,
            "Precision": prec
        })

    if global_summary:
        df_global = pd.DataFrame(global_summary)
        df_summary = df_global.mean(numeric_only=True).to_frame("Moyenne").T
        print("\nGlobal summary")
        display(df_global)
        display(df_summary)
        results["global_summary"] = df_summary

    return results


# -------------------------------------------------------------


def analyse_modele_dict_v3(
    dico_df: dict,
    best_models: dict,
    meta_index: int = 0,
    pattern_sep: str = "_",
    do_smooth: bool = True,   
    smooth_window: int = 5,
    smooth_poly: int = 2,
    tick_spacing_x: int = 5,
    tick_spacing_y: float = 0.1,     # not used here
    exclude_labels: Optional[Sequence[str]] = None,
    top_features: int = 10,          # not used here
    sensi_peakfinder: float = 0.05,
    class_name_map: Optional[dict] = None,
    median_colors: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    
    """
    In-depth analysis and visualization of already-trained SVM or LogReg models.
    do_smooth:
      - True  -> Savitzky-Golay smoothing applied (when possible)
      - False -> no smoothing

    Parameters
    ----------
    dico_df : dict
        Dictionary of DataFrames
    best_models : dict
        Dictionary of already-trained models (per group)
    meta_index : int, default 0
        Position of the key element used to group the DataFrames
    pattern_sep : str, default '_'
        Key separator
    do_smooth : bool, default True
        Whether to apply Savitzky-Golay smoothing to the spectra
    smooth_window : int, default 5
        Window size of the Savitzky-Golay smoothing filter
    smooth_poly : int, default 2
        Polynomial order of the Savitzky-Golay smoothing filter
    tick_spacing_x : int, default 5
        Spacing of ticks on the X axis of the plots
    tick_spacing_y : float, default 0.1
        Spacing of ticks on the Y axis of the plots (kept for compatibility, not used here)
    exclude_labels : Sequence[str], optional
        Optional list of sub-groups (labels) to exclude from the analyses
    top_features : int, default 10
        Number of top features to display (kept for compatibility, not used here)
    sensi_peakfinder : float, default 0.05
        Sensitivity threshold used by the peak-finding algorithm
    class_name_map : dict, optional
        Optional mapping of class indices to display names
    median_colors : Sequence[str], optional
        Optional sequence of colors used for the median spectra

    Returns
    -------
    dict
        Full analysis results for each group.
    """

    current_style = plt.rcParams.copy()
    plt.style.use("default")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    def _savgol(y: np.ndarray, window: int, poly: int) -> np.ndarray:
        """
        Savitzky-Golay smoothing:
        - window must be odd
        - window <= len(y)
        - poly < window
        if not possible, returns an unchanged y.
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        if (not do_smooth) or n < 3:
            return y

        w = int(window)
        if w < 3:
            return y
        if w > n:
            w = n
        if w % 2 == 0:
            w -= 1
        if w < 3:
            return y

        p = int(poly)
        if p >= w:
            p = w - 1
        if p < 1:
            return y

        return savgol_filter(y, window_length=w, polyorder=p)

    results: Dict[str, Dict[str, Any]] = {}

    for group_key, info in best_models.items():
        print(f"Model analysis — Group: {group_key}")

        if isinstance(info, dict) and "best_model" in info:
            model = info["best_model"]
            y_test = info.get("y_test", None)
            y_pred = info.get("y_pred", None)
        elif isinstance(info, dict) and "model" in info:
            model = info["model"]
            y_test = info.get("y_test", None)
            y_pred = info.get("y_pred", None)
        else:
            model = info
            y_test, y_pred = None, None

        if model is None:
            print(f"No valid model {group_key}, ignored.")
            continue

        group_dfs = [
            (k, df)
            for k, df in dico_df.items()
            if k.split(pattern_sep)[meta_index] == group_key
        ]
        if not group_dfs:
            print(f"No valid model {group_key}, ignored.")
            continue

        dfs = [df for _, df in group_dfs]
        data = pd.concat(dfs, ignore_index=True)

        subgroup_labels = []
        for key, df in group_dfs:
            n = len(df)
            parts = key.split(pattern_sep)
            subgroup_label = pattern_sep.join([p for i, p in enumerate(parts) if i != meta_index])
            subgroup_labels.extend([subgroup_label] * n)
        subgroup_labels = np.array(subgroup_labels)

        if exclude_labels:
            mask = ~np.isin(subgroup_labels, exclude_labels)
            data = data.loc[mask].reset_index(drop=True)
            subgroup_labels = subgroup_labels[mask]
            print(f"Excluded classes: {exclude_labels}")

        if class_name_map is not None:
            subgroup_labels = np.array([class_name_map.get(lbl, lbl) for lbl in subgroup_labels])
            if y_test is not None:
                y_test = np.array([class_name_map.get(lbl, lbl) for lbl in y_test])
            if y_pred is not None:
                y_pred = np.array([class_name_map.get(lbl, lbl) for lbl in y_pred])
            print(f"Applied the classes remapping ({len(class_name_map)}).")

        data_num = data.select_dtypes(include=[np.number]).dropna(axis=0)
        valid_idx = data_num.index
        subgroup_labels = subgroup_labels[valid_idx]

        if len(np.unique(subgroup_labels)) < 2:
            print(f"Not enough classes after filtering for {group_key}, ignored.")
            continue

        if y_test is not None and y_pred is not None:
            print("Confusion matrix")
            classes = np.unique(np.concatenate([y_test, y_pred]))
            conf_mat = confusion_matrix(y_test, y_pred, labels=classes, normalize="true") * 100

            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=classes)
            disp.plot(cmap="Blues", ax=ax, values_format=".1f", colorbar=False)

            ax.grid(False)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_color("black")
                ax.spines[spine].set_linewidth(1.2)

            ax.set_title("")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("True Class")
            plt.tight_layout()
            plt.show()

        peaks = np.array([], dtype=int)
        has_coef = hasattr(model, "coef_")

        if has_coef:
            print("Feature importance")
            coefs = model.coef_
            if coefs.ndim > 1 and coefs.shape[0] > 1:
                coefs = np.mean(np.abs(coefs), axis=0)
            else:
                coefs = np.abs(coefs).flatten()

            feature_names = info.get("feature_names", list(data_num.columns))

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": coefs
            })

            y = importance_df["Importance"].to_numpy(dtype=float)
            x = np.arange(len(y))

            y_used = _savgol(y, smooth_window, smooth_poly)

            if np.max(y_used) > 0:
                peaks, _ = find_peaks(y_used, prominence=np.max(y_used) * sensi_peakfinder)
            else:
                peaks = np.array([], dtype=int)

            print(f"{len(peaks)} bands detected in the importances")

            fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
            ax.plot(x, y_used, color="black", linewidth=1.8)
            ax.fill_between(x, y_used, color="black", alpha=0.05)

            if len(peaks) > 0:
                ax.scatter(x[peaks], y_used[peaks], color="red", s=40, zorder=5, label="Peaks")
                for p in peaks:
                    ax.axvline(p, color="red", linestyle="--", alpha=0.4, linewidth=1)
                    feature_label = str(importance_df["Feature"].iloc[p])
                    ax.text(
                        p,
                        ax.get_ylim()[1] * 0.98,
                        feature_label,
                        rotation=90,
                        ha="right",
                        va="top",
                        fontsize=8,
                        color="black",
                    )

            ax.set_xlim(-0.5, len(x) - 0.5)
            ax.grid(False)

            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            for side in ["bottom", "left"]:
                ax.spines[side].set_color("black")
                ax.spines[side].set_linewidth(1.5)

            ax.tick_params(width=1.2, length=6, direction="out")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

            ticks_all = np.arange(0, len(x), tick_spacing_x)
            ticks_major = [t for i, t in enumerate(ticks_all) if i % 3 == 0]
            ticks_minor = [t for i, t in enumerate(ticks_all) if i % 3 != 0]

            ax.set_xticks(ticks_major)
            ax.set_xticklabels([str(importance_df["Feature"].iloc[t]) for t in ticks_major], rotation=0)
            ax.set_xticks(ticks_minor, minor=True)

            ax.tick_params(axis="x", which="major", length=8, width=1.4, direction="out")
            ax.tick_params(axis="x", which="minor", length=4, width=1.0, direction="out")

            ax.set_title("")
            ax.set_xlabel("Raman shift (cm⁻¹)")
            ax.set_ylabel("|Coefficient|")
            plt.tight_layout()
            plt.show()

        print("Median profile by class")
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

        default_colors = ["#000033", "#3366CC", "#99CCFF", "#99CCFF"]
        if median_colors is None:
            median_colors = default_colors

        if class_name_map is not None:
            ordered_classes = list(class_name_map.values())
        else:
            ordered_classes = list(np.unique(subgroup_labels))

        x_axis = np.arange(data_num.shape[1])

        for i, classe in enumerate(ordered_classes):
            if classe not in subgroup_labels:
                continue

            data_classe = data_num[subgroup_labels == classe]
            median_profile = data_classe.median(axis=0).to_numpy(dtype=float)
            median_used = _savgol(median_profile, smooth_window, smooth_poly)

            color = median_colors[i % len(median_colors)]
            ax.plot(x_axis, median_used, linewidth=2, label=classe, color=color)

        ax.set_xlim(-0.5, len(x_axis) - 0.5)

        if len(peaks) > 0:
            for p in peaks:
                ax.axvline(p, color="black", linestyle="--", alpha=0.6, linewidth=1)
                feature_label = str(data_num.columns[p])
                ax.text(
                    p,
                    ax.get_ylim()[1] * 0.98,
                    feature_label,
                    rotation=90,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="black",
                )

        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.35)

        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["bottom", "left"]:
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(1.5)

        ax.tick_params(width=1.5, length=6, direction="out")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

        ticks_all = np.arange(0, len(x_axis), tick_spacing_x)
        ticks_major = [t for i, t in enumerate(ticks_all) if i % 3 == 0]
        ticks_minor = [t for i, t in enumerate(ticks_all) if i % 3 != 0]

        ax.set_xticks(ticks_major)
        ax.set_xticklabels([str(data_num.columns[t]) for t in ticks_major], rotation=0)
        ax.set_xticks(ticks_minor, minor=True)

        ax.tick_params(axis="x", which="major", length=8, width=1.4, direction="out")
        ax.tick_params(axis="x", which="minor", length=4, width=1.0, direction="out")

        ax.legend(title="", fontsize=9)
        ax.set_title("")
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Normalized intensity (a.u.)")

        plt.tight_layout()
        plt.show()

        results[group_key] = {
            "coef_plot_done": bool(has_coef),
            "median_profiles_done": True,
            "confusion_matrix_done": y_test is not None and y_pred is not None,
            "n_peaks": int(len(peaks)),
            "smoothing_enabled": bool(do_smooth),
        }

    plt.rcParams.update(current_style)
    print("Analysis finished")
    return results

