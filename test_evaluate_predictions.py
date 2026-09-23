import numpy as np
import pytest

from evaluate_predictions import evaluate


def pair():
    reference = {"positions_zyx": np.array([[1., 2., 3.], [4., 5., 6.]]),
                 "tangents_zyx": np.array([[1., 0., 0.], [0., 1., 0.]]),
                 "weights": np.array([.5, .5]), "fiber_ids": np.array([0, 1])}
    predicted = {"positions_zyx": reference["positions_zyx"].copy(),
                 "axes_zyx": -reference["tangents_zyx"].copy(), "confidence": np.array([.8, .8])}
    return reference, predicted


def test_missing_predictions_cannot_improve_all_sample_score():
    ref, pred = pair()
    pred["axes_zyx"][1] = np.nan
    report = evaluate(ref, pred)
    assert report["all_samples"]["mean_degrees"] == 45
    assert report["invalid_predictions"] == 1
    assert report["thresholds"]["0.75"]["coverage"] == .5


def test_wrong_coordinates_are_rejected_even_if_vectors_match():
    ref, pred = pair()
    pred["positions_zyx"] = pred["positions_zyx"][:, ::-1]
    with pytest.raises(ValueError, match="registration"):
        evaluate(ref, pred)


def test_axial_predictions_need_no_sign_combing():
    ref, pred = pair()
    assert evaluate(ref, pred)["all_samples"]["mean_degrees"] == 0
