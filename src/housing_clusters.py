"""Agrupación de viviendas del censo de California y clasificador de esos clusters.

El enunciado fija k = 6. El diagnóstico de inercia y silhouette es complementario
y no cambia ese valor. La semilla y un hilo de BLAS mantienen el ajuste repetible.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "housing.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "housing_with_cluster.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
KMEANS_PATH = MODEL_DIR / "kmeans.joblib"
SUPERVISED_PATH = MODEL_DIR / "supervised_classifier.joblib"
METADATA_PATH = MODEL_DIR / "training_metadata.json"

DATA_URL = (
    "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"
)
OFFICIAL_SHA256 = "0e4a0bc8b337d88c95259a8ec05a58c662f26f87487f9cbf2448fe93207bcb5c"
FEATURES = ["Latitude", "Longitude", "MedInc"]
N_CLUSTERS = 6
TEST_SIZE = 0.2
RANDOM_STATE = 42
K_DIAGNOSTIC_RANGE = range(2, 11)

CANVAS = "#F7F8FA"
SURFACE = "#FFFFFF"
INK = "#18202B"
MUTED = "#505E70"
BORDER = "#788598"
SUBTLE = "#E7ECF2"
ACTION = "#B53A17"
FONT_CANDIDATES = (
    (
        "Arial",
        (
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
        ),
    ),
    (
        "Helvetica",
        (
            Path("/usr/share/fonts/truetype/msttcorefonts/Helvetica.ttf"),
        ),
    ),
    (
        "Arimo",
        (
            Path("/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf"),
            Path("/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf"),
        ),
    ),
    (
        "Liberation Sans",
        (
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ),
    ),
)
CHOSEN_FONT: str | None = None

CLUSTER_COLORS = ["#332288", "#0B6E4F", "#9A3E00", "#882255", "#005F73", "#6E4B12"]
CLUSTER_MARKERS = ["o", "s", "^", "D", "v", "P"]

MODEL_SPECS = (
    {
        "name": "logistic_regression",
        "label": "Regresión logística",
        "simplicity": 0,
    },
    {
        "name": "decision_tree",
        "label": "Árbol de decisión",
        "simplicity": 1,
    },
    {
        "name": "knn",
        "label": "k vecinos más cercanos (15)",
        "simplicity": 2,
    },
    {
        "name": "random_forest",
        "label": "Bosque aleatorio (100 árboles)",
        "simplicity": 3,
    },
)
SELECTION_RULE = (
    "Se elige el clasificador con mayor F1 macro medio en validación cruzada "
    "estratificada de 5 pliegues sobre el entrenamiento. Si el F1 medio empata, "
    "se prefiere la menor desviación típica de los pliegues y, después, el modelo "
    "más simple de esta lista: regresión logística, árbol de decisión, k vecinos, "
    "bosque aleatorio. El conjunto de prueba no entra en la elección."
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _channel_linear(value: int) -> float:
    channel = value / 255
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(color: str) -> float:
    red, green, blue = _hex_to_rgb(color)
    return (
        0.2126 * _channel_linear(red)
        + 0.7152 * _channel_linear(green)
        + 0.0722 * _channel_linear(blue)
    )


def contrast_ratio(foreground: str, background: str) -> float:
    light = max(relative_luminance(foreground), relative_luminance(background))
    dark = min(relative_luminance(foreground), relative_luminance(background))
    return (light + 0.05) / (dark + 0.05)


def annotation_color(background: str) -> str:
    ink_ratio = contrast_ratio(INK, background)
    white_ratio = contrast_ratio("#FFFFFF", background)
    if ink_ratio >= 4.5 and ink_ratio >= white_ratio:
        return INK
    if white_ratio >= 4.5:
        return "#FFFFFF"
    return INK if ink_ratio >= white_ratio else "#FFFFFF"


def mix_hex(start: str, end: str, amount: float) -> str:
    amount = min(1.0, max(0.0, amount))
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    mixed = [int(round(a + (b - a) * amount)) for a, b in zip(start_rgb, end_rgb)]
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def choose_font() -> str:
    global CHOSEN_FONT
    if CHOSEN_FONT:
        return CHOSEN_FONT
    for family, paths in FONT_CANDIDATES:
        present = [path for path in paths if path.is_file()]
        if not present:
            continue
        for path in present:
            font_manager.fontManager.addfont(str(path))
        CHOSEN_FONT = family
        return family
    raise FileNotFoundError(
        "No están Arial, Helvetica, Arimo ni Liberation Sans. "
        "No se sustituye la fuente por otra familia."
    )


def palette_contrast_report() -> dict:
    family = choose_font()
    text = []
    for name, color in (("ink", INK), ("muted", MUTED), ("action", ACTION)):
        for background_name, background in (("canvas", CANVAS), ("surface", SURFACE)):
            text.append(
                {
                    "name": name,
                    "hex": color,
                    "background": background_name,
                    "ratio": round(contrast_ratio(color, background), 2),
                }
            )
    marks = []
    for cluster, color in enumerate(CLUSTER_COLORS):
        marks.append(
            {
                "cluster": cluster,
                "hex": color,
                "ratio_on_canvas": round(contrast_ratio(color, CANVAS), 2),
                "ratio_on_surface": round(contrast_ratio(color, SURFACE), 2),
            }
        )
    return {
        "font_requested_by_signal": "Arial, Helvetica, sans-serif",
        "font_used": family,
        "font_reason": (
            "Orden de búsqueda: Arial, Helvetica, Arimo, Liberation Sans. "
            "No se descargan fuentes. En esta máquina Arial y Helvetica no están; "
            "Arimo es la siguiente, métricamente compatible con Arial."
            if family == "Arimo"
            else f"Fuente usada: {family}."
        ),
        "text": text,
        "marks": marks,
    }


_STYLE_READY = False


def apply_chart_style() -> None:
    global _STYLE_READY
    family = choose_font()
    if not _STYLE_READY:
        _STYLE_READY = True
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 16,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.edgecolor": BORDER,
            "axes.linewidth": 1,
            "figure.facecolor": CANVAS,
            "axes.facecolor": SURFACE,
            "axes.titlecolor": INK,
            "savefig.facecolor": CANVAS,
        }
    )


def _exact_features(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != FEATURES:
        raise ValueError(
            "Se esperaban exactamente las columnas "
            f"{FEATURES}, en ese orden. Llegaron {list(frame.columns)}."
        )
    return frame


def load_feature_frame(path: Path = RAW_CSV) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"No está el dataset: {path}")
    digest = file_sha256(path)
    if digest != OFFICIAL_SHA256:
        raise ValueError(
            "housing.csv no coincide con el archivo oficial del tutorial "
            f"({DATA_URL}). sha256 obtenido: {digest}"
        )
    raw = pd.read_csv(path)
    missing = [column for column in FEATURES if column not in raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas del enunciado: {missing}")
    features = raw.loc[:, FEATURES].copy()
    if features.isna().any().any():
        raise ValueError("Hay valores ausentes en Latitude, Longitude o MedInc.")
    info = {
        "sha256": digest,
        "n_rows": int(len(raw)),
        "columns_file": list(raw.columns),
        "columns_used": list(FEATURES),
        "columns_unused": [column for column in raw.columns if column not in FEATURES],
        "bytes": path.stat().st_size,
    }
    return features, info


def split_train_test(
    features: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _exact_features(features)
    train, test = train_test_split(
        features,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train = train.copy()
    test = test.copy()
    if len(train) + len(test) != len(features):
        raise RuntimeError("La división no conserva todas las filas.")
    if not train.index.intersection(test.index).empty:
        raise RuntimeError("Entrenamiento y prueba comparten filas.")
    return train, test


def fit_kmeans(
    train_features: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    train_features = _exact_features(train_features)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    n_init=10,
                    random_state=random_state,
                    algorithm="lloyd",
                ),
            ),
        ]
    )
    model.fit(train_features)
    return model


def cluster_series(labels: np.ndarray, index: pd.Index) -> tuple[pd.Series, str]:
    raw = pd.Series(np.asarray(labels), index=index, name="cluster")
    dtype_before = str(raw.dtype)
    categorized = raw.astype(int).astype(
        pd.CategoricalDtype(categories=list(range(N_CLUSTERS)))
    )
    categorized.name = "cluster"
    return categorized, dtype_before


def predict_clusters(model: Pipeline, features: pd.DataFrame) -> tuple[pd.Series, str]:
    features = _exact_features(features)
    labels = model.predict(features)
    return cluster_series(labels, features.index)


def attach_split(
    features: pd.DataFrame, clusters: pd.Series, split_name: str
) -> pd.DataFrame:
    labeled = features.copy()
    labeled["cluster"] = clusters.astype(int).astype(
        pd.CategoricalDtype(categories=list(range(N_CLUSTERS)))
    )
    labeled["split"] = split_name
    return labeled


def summarize_clusters(train_labeled: pd.DataFrame, test_labeled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster in range(N_CLUSTERS):
        train_part = train_labeled.loc[train_labeled["cluster"].astype(int) == cluster]
        test_part = test_labeled.loc[test_labeled["cluster"].astype(int) == cluster]
        rows.append(
            {
                "cluster": cluster,
                "n_train": int(len(train_part)),
                "n_test": int(len(test_part)),
                "mean_latitude": float(train_part["Latitude"].mean()),
                "mean_longitude": float(train_part["Longitude"].mean()),
                "mean_medinc": float(train_part["MedInc"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    if int(summary["n_train"].sum()) != len(train_labeled):
        raise RuntimeError("El resumen de entrenamiento no cuadra.")
    if int(summary["n_test"].sum()) != len(test_labeled):
        raise RuntimeError("El resumen de prueba no cuadra.")
    return summary


def k_diagnostics(
    train_features: pd.DataFrame,
    ks: range = K_DIAGNOSTIC_RANGE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    train_features = _exact_features(train_features)
    scaled = StandardScaler().fit_transform(train_features)
    rows = []
    for k in ks:
        print(f"  diagnóstico k={k}", flush=True)
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=random_state,
            algorithm="lloyd",
        )
        labels = model.fit_predict(scaled)
        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(scaled, labels)),
            }
        )
    return pd.DataFrame(rows)


def make_supervised(name: str, random_state: int = RANDOM_STATE):
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                ),
            ]
        )
    if name == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=15)),
            ]
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Clasificador desconocido: {name}")


def compare_supervised(
    train_features: pd.DataFrame,
    train_clusters: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    train_features = _exact_features(train_features)
    target = train_clusters.astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows = []
    for spec in MODEL_SPECS:
        print(f"  validando {spec['name']}", flush=True)
        scores = cross_val_score(
            make_supervised(spec["name"], random_state),
            train_features,
            target,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        )
        rows.append(
            {
                "name": spec["name"],
                "label": spec["label"],
                "simplicity": spec["simplicity"],
                "f1_macro_mean": float(scores.mean()),
                "f1_macro_std": float(scores.std(ddof=1)),
                "fold_0": float(scores[0]),
                "fold_1": float(scores[1]),
                "fold_2": float(scores[2]),
                "fold_3": float(scores[3]),
                "fold_4": float(scores[4]),
            }
        )
    return pd.DataFrame(rows)


def choose_supervised(comparison: pd.DataFrame) -> pd.Series:
    ranked = comparison.sort_values(
        by=["f1_macro_mean", "f1_macro_std", "simplicity"],
        ascending=[False, True, True],
    )
    return ranked.iloc[0]


def importance_note(model, name: str) -> str:
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        pairs = ", ".join(
            f"{feature}={value:.3f}"
            for feature, value in zip(FEATURES, estimator.feature_importances_)
        )
        return f"Importancia por impureza, variables sin estandarizar: {pairs}."
    if hasattr(estimator, "coef_"):
        magnitude = np.mean(np.abs(estimator.coef_), axis=0)
        pairs = ", ".join(
            f"{feature}={value:.3f}" for feature, value in zip(FEATURES, magnitude)
        )
        return (
            "Media del valor absoluto de los coeficientes sobre variables "
            f"estandarizadas: {pairs}."
        )
    if name == "knn":
        return (
            "k vecinos no produce coeficientes. La clase sale de los 15 vecinos "
            "más cercanos en el espacio estandarizado."
        )
    return "Este estimador no expone coeficientes ni importancias."


def evaluate_supervised(model, test_features: pd.DataFrame, test_clusters: pd.Series) -> dict:
    test_features = _exact_features(test_features)
    observed = test_clusters.astype(int).to_numpy()
    predicted = model.predict(test_features)
    labels = list(range(N_CLUSTERS))
    report = classification_report(
        observed,
        predicted,
        labels=labels,
        digits=3,
        zero_division=0,
    )
    per_class = f1_score(observed, predicted, labels=labels, average=None, zero_division=0)
    matrix = confusion_matrix(observed, predicted, labels=labels)
    worst = int(np.argmin(per_class))
    return {
        "accuracy": float(accuracy_score(observed, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
        "f1_macro": float(f1_score(observed, predicted, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(observed, predicted, average="weighted", zero_division=0)
        ),
        "f1_per_cluster": [float(value) for value in per_class],
        "worst_cluster": worst,
        "worst_f1": float(per_class[worst]),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "predicted": predicted,
    }


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    return joblib.load(path)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK, labelsize=12)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.grid(True, color=SUBTLE, linewidth=1)
    ax.set_axisbelow(True)


def label_positions(summary: pd.DataFrame, minimum_degrees: float = 0.55) -> dict[int, tuple[float, float]]:
    """Separa etiquetas cuyos centros geográficos casi coinciden."""
    points = {
        int(row.cluster): np.array([row.mean_longitude, row.mean_latitude], dtype=float)
        for row in summary.itertuples(index=False)
    }
    offsets = {key: np.zeros(2) for key in points}
    for _ in range(12):
        keys = list(points)
        for left_index, left in enumerate(keys):
            for right in keys[left_index + 1 :]:
                delta = (points[left] + offsets[left]) - (points[right] + offsets[right])
                distance = float(np.hypot(delta[0], delta[1]))
                if distance >= minimum_degrees:
                    continue
                direction = np.array([0.0, 1.0]) if distance < 1e-9 else delta / distance
                push = (minimum_degrees - distance) / 2
                offsets[left] = offsets[left] + direction * push
                offsets[right] = offsets[right] - direction * push
    return {
        key: (float(points[key][0] + offsets[key][0]), float(points[key][1] + offsets[key][1]))
        for key in points
    }


def plot_clusters(
    train_labeled: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
    test_labeled: pd.DataFrame | None = None,
) -> None:
    apply_chart_style()
    fig, ax = plt.subplots(figsize=(11.2, 8.4))
    fig.patch.set_facecolor(CANVAS)
    _style_axes(ax)
    for cluster in range(N_CLUSTERS):
        part = train_labeled.loc[train_labeled["cluster"].astype(int) == cluster]
        ax.scatter(
            part["Longitude"],
            part["Latitude"],
            s=16,
            alpha=0.55,
            c=CLUSTER_COLORS[cluster],
            marker=CLUSTER_MARKERS[cluster],
            linewidths=0,
            label=f"Cluster {cluster}",
            zorder=2,
        )
    if test_labeled is not None:
        for cluster in range(N_CLUSTERS):
            part = test_labeled.loc[test_labeled["cluster"].astype(int) == cluster]
            ax.scatter(
                part["Longitude"],
                part["Latitude"],
                s=28,
                alpha=0.9,
                c=CLUSTER_COLORS[cluster],
                marker="x",
                linewidths=0.8,
                zorder=3,
            )
    positions = label_positions(summary)
    for row in summary.itertuples(index=False):
        cluster = int(row.cluster)
        label_lon, label_lat = positions[cluster]
        moved = abs(label_lon - row.mean_longitude) + abs(label_lat - row.mean_latitude) > 0.05
        if moved:
            ax.plot(
                [row.mean_longitude, label_lon],
                [row.mean_latitude, label_lat],
                color=INK,
                linewidth=1,
                zorder=4,
            )
        ax.annotate(
            str(cluster),
            (label_lon, label_lat),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": SURFACE,
                "edgecolor": INK,
                "linewidth": 1,
            },
            zorder=5,
        )
    ax.set_xlabel("Longitud (grados)")
    ax.set_ylabel("Latitud (grados)")
    if test_labeled is None:
        ax.set_title("Clusters del entrenamiento (k = 6)")
        footer = (
            "Cada marca es una vivienda del entrenamiento. El número identifica el cluster. "
            "Si dos centros casi coinciden, la etiqueta se separa y un trazo la une al centro. Semilla 42."
        )
    else:
        ax.set_title("Clusters de entrenamiento y prueba (k = 6)")
        footer = (
            "Marca rellena: entrenamiento. Cruz: prueba, con el cluster predicho. "
            "Si dos centros casi coinciden, la etiqueta se separa y un trazo la une al centro. Semilla 42."
        )
    ax.set_aspect("equal", adjustable="box")
    legend = ax.legend(
        title="Cluster",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        edgecolor=BORDER,
        facecolor=SURFACE,
    )
    legend.get_title().set_color(INK)
    fig.text(0.01, -0.02, footer, ha="left", va="top", fontsize=12, color=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=CANVAS)
    plt.close(fig)


def plot_income(summary: pd.DataFrame, path: Path) -> None:
    apply_chart_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor(CANVAS)
    _style_axes(ax)
    positions = np.arange(N_CLUSTERS)
    ax.barh(
        positions,
        summary["mean_medinc"],
        color=CLUSTER_COLORS,
        height=0.7,
        zorder=2,
    )
    for position, value in zip(positions, summary["mean_medinc"]):
        ax.text(
            value + 0.08,
            position,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=12,
            color=INK,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels([f"Cluster {cluster}" for cluster in range(N_CLUSTERS)])
    ax.invert_yaxis()
    ax.set_xlim(0, float(summary["mean_medinc"].max()) + 1.3)
    ax.set_xlabel("MedInc medio (unidades del archivo)")
    ax.set_title("Ingreso mediano medio por cluster")
    fig.text(
        0.01,
        -0.04,
        "Medias del entrenamiento. El archivo llega a 15.0001: en el California Housing "
        "público esa unidad son decenas de miles de dólares de 1990.",
        ha="left",
        va="top",
        fontsize=12,
        color=INK,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=CANVAS)
    plt.close(fig)


def plot_diagnostics(diagnostics: pd.DataFrame, path: Path) -> None:
    apply_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.6))
    fig.patch.set_facecolor(CANVAS)
    for ax, column, ylabel, title in (
        (axes[0], "inertia", "Inercia", "Inercia en el espacio estandarizado"),
        (axes[1], "silhouette", "Silhouette", "Silhouette en el espacio estandarizado"),
    ):
        _style_axes(ax)
        ax.plot(
            diagnostics["k"],
            diagnostics[column],
            color=INK,
            marker="o",
            linewidth=2,
            markersize=7,
            zorder=2,
        )
        ax.axvline(N_CLUSTERS, color=ACTION, linewidth=2, zorder=1)
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(list(diagnostics["k"]))
    fig.suptitle("Diagnóstico de k. El modelo usa k = 6, como pide el enunciado.", color=INK)
    fig.text(
        0.01,
        -0.05,
        "Línea roja: k = 6 del enunciado. Calculado solo con el entrenamiento estandarizado. Semilla 42.",
        ha="left",
        va="top",
        fontsize=12,
        color=INK,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=CANVAS)
    plt.close(fig)


def plot_confusion(matrix: list[list[int]], model_label: str, path: Path) -> None:
    apply_chart_style()
    values = np.asarray(matrix, dtype=float)
    peak = float(values.max()) if values.size else 1.0
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    fig.patch.set_facecolor(CANVAS)
    ax.set_facecolor(SURFACE)
    for row in range(N_CLUSTERS):
        for col in range(N_CLUSTERS):
            amount = 0 if peak == 0 else values[row, col] / peak
            color = mix_hex(CANVAS, "#73230E", amount)
            ax.add_patch(
                plt.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor=color, edgecolor=SURFACE)
            )
            ax.text(
                col,
                row,
                f"{int(values[row, col])}",
                ha="center",
                va="center",
                fontsize=12,
                color=annotation_color(color),
            )
    ax.set_xlim(-0.5, N_CLUSTERS - 0.5)
    ax.set_ylim(N_CLUSTERS - 0.5, -0.5)
    ax.set_xticks(range(N_CLUSTERS))
    ax.set_yticks(range(N_CLUSTERS))
    ax.set_xlabel("Predicción del clasificador")
    ax.set_ylabel("Cluster de K-Means")
    ax.set_title(f"Matriz de confusión en prueba\n{model_label}")
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    fig.text(
        0.01,
        -0.02,
        "El número de la celda es el conteo. El color repite ese conteo y no es el único dato.",
        ha="left",
        va="top",
        fontsize=12,
        color=INK,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=CANVAS)
    plt.close(fig)


def lectura_escala(train_features: pd.DataFrame) -> str:
    deviation = train_features[FEATURES].std(ddof=1)
    return (
        "En el entrenamiento, la desviación típica es "
        f"{deviation['Latitude']:.2f} en latitud, {deviation['Longitude']:.2f} en longitud "
        f"y {deviation['MedInc']:.2f} en MedInc. Las cifras brutas se parecen, pero las "
        "unidades no: grados frente a la escala de ingreso del archivo. StandardScaler se "
        "ajusta solo con el entrenamiento y K-Means usa la distancia euclídea en ese espacio."
    )


def lectura_k(diagnostics: pd.DataFrame) -> str:
    inertia = diagnostics["inertia"].to_numpy()
    relative_drop = (inertia[:-1] - inertia[1:]) / inertia[:-1]
    drop_at = int(np.argmax(relative_drop))
    k_from = int(diagnostics.iloc[drop_at]["k"])
    k_to = int(diagnostics.iloc[drop_at + 1]["k"])
    best_row = diagnostics.loc[diagnostics["silhouette"].idxmax()]
    used = diagnostics.loc[diagnostics["k"] == N_CLUSTERS].iloc[0]
    by_k = diagnostics.set_index("k")
    local = ""
    if 5 in by_k.index and 6 in by_k.index and 7 in by_k.index:
        sil_5 = float(by_k.loc[5, "silhouette"])
        sil_6 = float(by_k.loc[6, "silhouette"])
        sil_7 = float(by_k.loc[7, "silhouette"])
        if sil_6 > sil_5 and sil_6 > sil_7:
            local = (
                f" De k=5 a k=6 el silhouette pasa de {sil_5:.3f} a {sil_6:.3f} "
                f"y en k=7 baja a {sil_7:.3f}. Es un repunte local, no el máximo del rango."
            )
    return (
        f"La inercia baja de {diagnostics.iloc[0]['inertia']:.0f} en k="
        f"{int(diagnostics.iloc[0]['k'])} a {diagnostics.iloc[-1]['inertia']:.0f} en k="
        f"{int(diagnostics.iloc[-1]['k'])}. El mayor descenso relativo entre k consecutivos "
        f"está entre k={k_from} y k={k_to} ({100 * float(relative_drop[drop_at]):.1f} %). "
        f"El silhouette máximo del rango es {best_row['silhouette']:.3f} en k="
        f"{int(best_row['k'])}. El modelo guardado usa k=6, con silhouette "
        f"{used['silhouette']:.3f} e inercia {used['inertia']:.0f}, porque ese k lo fija "
        f"el enunciado.{local} Este diagnóstico no lo sustituye."
    )


def lectura_clusters(
    summary: pd.DataFrame,
    n_train: int,
    n_test: int,
    features: pd.DataFrame,
) -> str:
    north = summary.loc[summary["mean_latitude"].idxmax()]
    south = summary.loc[summary["mean_latitude"].idxmin()]
    west = summary.loc[summary["mean_longitude"].idxmin()]
    east = summary.loc[summary["mean_longitude"].idxmax()]
    richest = summary.loc[summary["mean_medinc"].idxmax()]
    poorest = summary.loc[summary["mean_medinc"].idxmin()]
    share_gap = (summary["n_train"] / n_train - summary["n_test"] / n_test).abs()
    gap_row = summary.loc[share_gap.idxmax()]
    indexed = summary.set_index("cluster")
    close_sentences = []
    for left in range(N_CLUSTERS):
        for right in range(left + 1, N_CLUSTERS):
            distance = float(
                np.hypot(
                    indexed.loc[left, "mean_latitude"] - indexed.loc[right, "mean_latitude"],
                    indexed.loc[left, "mean_longitude"] - indexed.loc[right, "mean_longitude"],
                )
            )
            if distance >= 1:
                continue
            close_sentences.append(
                f"Los clusters {left} y {right} tienen centros a {distance:.2f} grados "
                f"(MedInc medio {indexed.loc[left, 'mean_medinc']:.2f} y "
                f"{indexed.loc[right, 'mean_medinc']:.2f}; "
                f"n={int(indexed.loc[left, 'n_train'])} y {int(indexed.loc[right, 'n_train'])})."
            )
    close_text = " ".join(close_sentences)
    return (
        f"El entrenamiento tiene {n_train} viviendas y la prueba {n_test}. "
        "K-Means se ajusta solo con el entrenamiento y después asigna la prueba.\n\n"
        f"El grupo más al norte es el {int(north.cluster)} "
        f"(latitud media {north.mean_latitude:.2f}, longitud media {north.mean_longitude:.2f}, "
        f"MedInc medio {north.mean_medinc:.2f}, n={int(north.n_train)} en entrenamiento). "
        f"El más al sur es el {int(south.cluster)} (latitud {south.mean_latitude:.2f}, "
        f"MedInc {south.mean_medinc:.2f}). "
        f"El más al oeste es el {int(west.cluster)} (longitud {west.mean_longitude:.2f}) "
        f"y el más al este es el {int(east.cluster)} (longitud {east.mean_longitude:.2f}). "
        "Longitud más baja quiere decir más al oeste, hacia el Pacífico, porque el gráfico "
        "usa grados sin proyección.\n\n"
        f"El MedInc medio más alto está en el cluster {int(richest.cluster)} "
        f"({richest.mean_medinc:.2f}, n={int(richest.n_train)}) y el más bajo en el "
        f"{int(poorest.cluster)} ({poorest.mean_medinc:.2f}). "
        f"En el archivo, MedInc va de {float(features['MedInc'].min()):.4f} a "
        f"{float(features['MedInc'].max()):.4f}. Ese tope coincide con el California Housing "
        "público, donde la unidad son decenas de miles de dólares de 1990.\n\n"
        "En el mapa hay grupos que ocupan la misma zona y se distinguen por el ingreso, "
        "porque el gráfico solo muestra latitud y longitud y MedInc también entra en la "
        f"distancia. {close_text}\n\n"
        f"Todos los clusters reciben viviendas de prueba. El mínimo es "
        f"{int(summary['n_test'].min())}. La mayor diferencia de proporción entre "
        f"entrenamiento y prueba es de {100 * float(share_gap.max()):.1f} puntos "
        f"porcentuales, en el cluster {int(gap_row.cluster)}. Las cruces de la prueba "
        "caen sobre las mismas nubes que el entrenamiento: la predicción no abre una región nueva."
    )


def lectura_modelo(comparison: pd.DataFrame, chosen: pd.Series, metrics: dict) -> str:
    ranked = comparison.sort_values(
        by=["f1_macro_mean", "f1_macro_std", "simplicity"],
        ascending=[False, True, True],
    )
    second = ranked.iloc[1]
    margin = float(chosen["f1_macro_mean"] - second["f1_macro_mean"])
    return (
        f"{SELECTION_RULE}\n\n"
        f"Resultado: {chosen['label']} con F1 macro medio {chosen['f1_macro_mean']:.3f} "
        f"(desviación {chosen['f1_macro_std']:.3f}). El siguiente es {second['label']} "
        f"con {second['f1_macro_mean']:.3f}. La diferencia es {margin:.3f}.\n\n"
        f"En la prueba, que no participó en la elección, la exactitud es {metrics['accuracy']:.3f}, "
        f"el F1 macro {metrics['f1_macro']:.3f}, el F1 ponderado {metrics['f1_weighted']:.3f} "
        f"y la exactitud balanceada {metrics['balanced_accuracy']:.3f}. "
        f"El cluster con F1 más bajo es el {metrics['worst_cluster']} "
        f"({metrics['worst_f1']:.3f}). Estas cifras miden si el clasificador reproduce "
        "las etiquetas de K-Means. No miden el precio de la vivienda ni una etiqueta humana."
    )


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_pipeline() -> dict:
    if N_CLUSTERS != 6:
        raise RuntimeError("El enunciado fija 6 clusters.")
    print("Cargando housing.csv...", flush=True)
    features, info = load_feature_frame(RAW_CSV)
    train, test = split_train_test(features)
    print(
        f"Filas {info['n_rows']}. Entrenamiento {len(train)}. Prueba {len(test)}.",
        flush=True,
    )
    print("Ajustando K-Means (k=6)...", flush=True)
    kmeans = fit_kmeans(train)
    train_clusters, dtype_before = predict_clusters(kmeans, train)
    test_clusters, _ = predict_clusters(kmeans, test)
    train_labeled = attach_split(train, train_clusters, "train")
    test_labeled = attach_split(test, test_clusters, "test")
    summary = summarize_clusters(train_labeled, test_labeled)
    print("Calculando inercia y silhouette para k=2..10...", flush=True)
    diagnostics = k_diagnostics(train)
    print("Comparando clasificadores...", flush=True)
    comparison = compare_supervised(train, train_clusters)
    chosen = choose_supervised(comparison)
    supervised = make_supervised(str(chosen["name"]))
    supervised.fit(train, train_clusters.astype(int))
    metrics = evaluate_supervised(supervised, test, test_clusters)
    importances = importance_note(supervised, str(chosen["name"]))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "clusters_train": FIG_DIR / "clusters_train.png",
        "clusters_train_test": FIG_DIR / "clusters_train_test.png",
        "cluster_income": FIG_DIR / "cluster_income.png",
        "k_diagnostics": FIG_DIR / "k_diagnostics.png",
        "confusion_matrix": FIG_DIR / "confusion_matrix.png",
    }
    plot_clusters(train_labeled, summary, paths["clusters_train"])
    plot_clusters(train_labeled, summary, paths["clusters_train_test"], test_labeled)
    plot_income(summary, paths["cluster_income"])
    plot_diagnostics(diagnostics, paths["k_diagnostics"])
    plot_confusion(metrics["confusion_matrix"], str(chosen["label"]), paths["confusion_matrix"])

    processed = pd.concat([train_labeled, test_labeled], axis=0).sort_index()
    processed_out = processed.copy()
    processed_out["cluster"] = processed_out["cluster"].astype(int)
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    processed_out.to_csv(PROCESSED_CSV, index=True, index_label="row")

    save_model(kmeans, KMEANS_PATH)
    save_model(supervised, SUPERVISED_PATH)
    reloaded_kmeans = load_model(KMEANS_PATH)
    reloaded_supervised = load_model(SUPERVISED_PATH)
    reload_kmeans_ok = bool(
        np.array_equal(reloaded_kmeans.predict(test), kmeans.predict(test))
    )
    reload_supervised_ok = bool(
        np.array_equal(
            reloaded_supervised.predict(test),
            supervised.predict(test),
        )
    )
    if not reload_kmeans_ok or not reload_supervised_ok:
        raise RuntimeError("El modelo recargado no reproduce la predicción.")

    scale = train[FEATURES].agg(["mean", "std", "min", "max"]).T
    scale.index.name = "feature"
    texts = {
        "escala": lectura_escala(train),
        "k": lectura_k(diagnostics),
        "clusters": lectura_clusters(summary, len(train), len(test), features),
        "modelo": lectura_modelo(comparison, chosen, metrics) + "\n\n" + importances,
    }
    contrast = palette_contrast_report()
    import sklearn

    metadata = {
        "dataset_url": DATA_URL,
        "dataset_sha256": info["sha256"],
        "dataset_bytes": info["bytes"],
        "n_rows": info["n_rows"],
        "features": FEATURES,
        "columns_unused": info["columns_unused"],
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_clusters": N_CLUSTERS,
        "n_train": len(train),
        "n_test": len(test),
        "kmeans": {
            "scaler": "StandardScaler ajustado en entrenamiento",
            "n_init": 10,
            "algorithm": "lloyd",
            "inertia_on_scaled_train": float(kmeans.named_steps["kmeans"].inertia_),
        },
        "selected_supervised_model": str(chosen["name"]),
        "selected_supervised_label": str(chosen["label"]),
        "selection_rule": SELECTION_RULE,
        "python": sys.version.split()[0],
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "models": {
            "kmeans": str(KMEANS_PATH.relative_to(ROOT)),
            "supervised_classifier": str(SUPERVISED_PATH.relative_to(ROOT)),
        },
    }
    evaluation = {
        "metadata": metadata,
        "cluster_dtype_before": dtype_before,
        "cluster_dtype_after": str(train_clusters.dtype),
        "cluster_categories": [int(value) for value in train_clusters.cat.categories],
        "cluster_summary": summary.to_dict(orient="records"),
        "k_diagnostics": diagnostics.to_dict(orient="records"),
        "model_comparison": comparison.to_dict(orient="records"),
        "test_metrics": {
            key: value
            for key, value in metrics.items()
            if key != "predicted"
        },
        "importance_note": importances,
        "reload_kmeans_ok": reload_kmeans_ok,
        "reload_supervised_ok": reload_supervised_ok,
        "lecturas": texts,
        "contrast": contrast,
    }
    _write_json(METADATA_PATH, metadata)
    _write_json(REPORT_DIR / "evaluation.json", evaluation)
    _write_json(REPORT_DIR / "contrast.json", contrast)
    summary.to_csv(REPORT_DIR / "cluster_summary.csv", index=False)
    diagnostics.to_csv(REPORT_DIR / "k_diagnostics.csv", index=False)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "classification_report.txt").write_text(
        metrics["classification_report"], encoding="utf-8"
    )
    (REPORT_DIR / "lectura.txt").write_text(
        "\n\n".join(
            [
                texts["escala"],
                texts["k"],
                texts["clusters"],
                texts["modelo"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = {
        "info": info,
        "head": features.head(5),
        "describe": features.describe().T,
        "scale_train": scale.reset_index(),
        "n_train": len(train),
        "n_test": len(test),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "kmeans_text": str(kmeans),
        "supervised_text": str(supervised),
        "cluster_dtype_before": dtype_before,
        "cluster_dtype_after": str(train_clusters.dtype),
        "cluster_values": sorted(int(value) for value in train_clusters.astype(int).unique()),
        "train_counts": train_clusters.astype(int).value_counts().sort_index(),
        "test_counts": test_clusters.astype(int).value_counts().sort_index(),
        "cluster_summary": summary,
        "diagnostics": diagnostics,
        "comparison": comparison,
        "selected_name": str(chosen["name"]),
        "selected_label": str(chosen["label"]),
        "selection_rule": SELECTION_RULE,
        "classification_report": metrics["classification_report"],
        "metrics": {key: value for key, value in metrics.items() if key != "predicted"},
        "importance_note": importances,
        "lectura_escala": texts["escala"],
        "lectura_k": texts["k"],
        "lectura_clusters": texts["clusters"],
        "lectura_modelo": texts["modelo"],
        "reload_kmeans_ok": reload_kmeans_ok,
        "reload_supervised_ok": reload_supervised_ok,
        "paths": {key: str(value.relative_to(ROOT)) for key, value in paths.items()},
        "processed_csv": str(PROCESSED_CSV.relative_to(ROOT)),
        "models": {
            "kmeans": str(KMEANS_PATH.relative_to(ROOT)),
            "supervised": str(SUPERVISED_PATH.relative_to(ROOT)),
        },
    }
    print("Modelos guardados.", flush=True)
    print(f"CSV derivado: {PROCESSED_CSV.relative_to(ROOT)}", flush=True)
    return result


if __name__ == "__main__":
    run_pipeline()
