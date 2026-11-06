"""Comprueba semilla, k del enunciado, hash del dataset y contraste de las marcas."""

import numpy as np
import pandas as pd
import pytest

from housing_clusters import (
    FEATURES,
    N_CLUSTERS,
    OFFICIAL_SHA256,
    RANDOM_STATE,
    RAW_CSV,
    ROOT,
    annotation_color,
    contrast_ratio,
    file_sha256,
    fit_kmeans,
    label_positions,
    load_feature_frame,
    load_model,
    make_supervised,
    palette_contrast_report,
    predict_clusters,
    save_model,
    split_train_test,
)


@pytest.fixture(scope="module")
def split_data():
    features, info = load_feature_frame(RAW_CSV)
    train, test = split_train_test(features)
    return features, info, train, test


def test_dataset_hash_matches_official_file(split_data):
    _, info, _, _ = split_data
    assert file_sha256(RAW_CSV) == OFFICIAL_SHA256
    assert info["sha256"] == OFFICIAL_SHA256
    assert info["columns_used"] == FEATURES


def test_split_is_stable_and_disjoint(split_data):
    features, _, train, test = split_data
    train_again, test_again = split_train_test(features)
    assert train.index.equals(train_again.index)
    assert test.index.equals(test_again.index)
    assert train.index.intersection(test.index).empty
    assert len(train) + len(test) == len(features)


def test_kmeans_labels_repeat_with_the_same_seed(split_data):
    _, _, train, test = split_data
    first = fit_kmeans(train)
    second = fit_kmeans(train)
    assert first.named_steps["kmeans"].n_clusters == N_CLUSTERS
    assert first.named_steps["kmeans"].random_state == RANDOM_STATE
    assert np.array_equal(first.predict(train), second.predict(train))
    assert np.array_equal(first.predict(test), second.predict(test))


def test_reloaded_kmeans_matches_the_fit(split_data, tmp_path):
    _, _, train, test = split_data
    model = fit_kmeans(train)
    path = tmp_path / "kmeans.joblib"
    save_model(model, path)
    loaded = load_model(path)
    assert np.array_equal(loaded.predict(test), model.predict(test))


def test_cluster_column_is_categorical_with_six_levels(split_data):
    _, _, train, _ = split_data
    clusters, dtype_before = predict_clusters(fit_kmeans(train), train)
    assert "int" in dtype_before
    assert isinstance(clusters.dtype, pd.CategoricalDtype)
    assert [int(value) for value in clusters.cat.categories] == list(range(6))
    assert set(clusters.astype(int).unique()).issubset(set(range(6)))


def test_supervised_tree_repeats_with_the_same_seed(split_data):
    _, _, train, test = split_data
    labels, _ = predict_clusters(fit_kmeans(train), train)
    target = labels.astype(int)
    first = make_supervised("decision_tree")
    second = make_supervised("decision_tree")
    first.fit(train, target)
    second.fit(train, target)
    assert np.array_equal(first.predict(test), second.predict(test))


def test_saved_models_use_six_clusters():
    kmeans = load_model(ROOT / "models" / "kmeans.joblib")
    supervised = load_model(ROOT / "models" / "supervised_classifier.joblib")
    assert kmeans.named_steps["kmeans"].n_clusters == 6
    assert supervised is not None


def test_overlapping_centroids_get_readable_labels():
    summary = pd.read_csv(ROOT / "reports" / "cluster_summary.csv")
    positions = label_positions(summary)
    found_close_pair = False
    for left in range(6):
        for right in range(left + 1, 6):
            row_left = summary.loc[summary["cluster"] == left].iloc[0]
            row_right = summary.loc[summary["cluster"] == right].iloc[0]
            raw_distance = np.hypot(
                row_left["mean_longitude"] - row_right["mean_longitude"],
                row_left["mean_latitude"] - row_right["mean_latitude"],
            )
            if raw_distance < 0.4:
                found_close_pair = True
                placed = np.hypot(
                    positions[left][0] - positions[right][0],
                    positions[left][1] - positions[right][1],
                )
                assert placed >= 0.55
    assert found_close_pair


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError):
        fit_kmeans(pd.DataFrame({"Latitude": [1.0], "Longitude": [2.0]}))


def test_chart_text_and_marks_meet_contrast():
    report = palette_contrast_report()
    assert report["font_used"] in {"Arial", "Helvetica", "Arimo", "Liberation Sans"}
    assert all(item["ratio"] >= 4.5 for item in report["text"])
    assert all(item["ratio_on_canvas"] >= 3 for item in report["marks"])
    assert all(item["ratio_on_surface"] >= 3 for item in report["marks"])
    assert contrast_ratio(annotation_color("#F7F8FA"), "#F7F8FA") >= 4.5
    assert contrast_ratio(annotation_color("#73230E"), "#73230E") >= 4.5
