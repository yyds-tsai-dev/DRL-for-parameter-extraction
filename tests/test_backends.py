from __future__ import annotations

import numpy as np
import pytest

from env.backends import CommitteePackageBackend, PredictionBackend, PredictionResult


class FakeArrayModel:
    """Mimics the InferenceModel surface CommitteePackageBackend relies on."""

    def __init__(self, model_package_path):
        self.model_package_path = model_package_path
        self.targets = ["hardness"]
        self.calls = []
        self.closed = False

    def predict_array(self, input_data):
        self.calls.append(input_data)
        return np.array([[720.0]]), np.array([[12.5]])

    def close(self):
        self.closed = True


class FakeTwoTargetModel(FakeArrayModel):
    def __init__(self, model_package_path):
        super().__init__(model_package_path)
        self.targets = ["hardness", "density"]

    def predict_array(self, input_data):
        self.calls.append(input_data)
        return np.array([[700.0, 7.9]]), np.array([[10.0, 0.2]])


def test_prediction_result_is_immutable():
    result = PredictionResult(values={"hardness": 1.0}, uncertainties={})

    with pytest.raises(Exception):
        result.values = {}


def test_committee_backend_maps_targets_to_values_and_uncertainties():
    backend = CommitteePackageBackend(
        "/tmp/fake.zip", inference_model_cls=FakeArrayModel
    )

    result = backend.predict({"frac_Ni": 0.2})

    assert result.values == {"hardness": 720.0}
    assert result.uncertainties == {"hardness": 12.5}
    assert backend.targets == ["hardness"]


def test_committee_backend_passes_features_as_single_row():
    backend = CommitteePackageBackend(
        "/tmp/fake.zip", inference_model_cls=FakeArrayModel
    )

    backend.predict({"frac_Ni": 0.2, "frac_Al": 0.1})

    assert backend._model.calls == [[{"frac_Ni": 0.2, "frac_Al": 0.1}]]


def test_committee_backend_supports_multiple_targets():
    backend = CommitteePackageBackend(
        "/tmp/fake.zip", inference_model_cls=FakeTwoTargetModel
    )

    result = backend.predict({"frac_Ni": 0.2})

    assert result.values == {"hardness": 700.0, "density": 7.9}
    assert result.uncertainties == {"hardness": 10.0, "density": 0.2}


def test_committee_backend_propagates_non_finite_values():
    class NonFiniteModel(FakeArrayModel):
        def predict_array(self, input_data):
            return np.array([[np.nan]]), np.array([[np.inf]])

    backend = CommitteePackageBackend(
        "/tmp/fake.zip", inference_model_cls=NonFiniteModel
    )

    result = backend.predict({"frac_Ni": 0.2})

    assert np.isnan(result.values["hardness"])
    assert np.isinf(result.uncertainties["hardness"])


def test_committee_backend_close_forwards_and_satisfies_protocol():
    backend = CommitteePackageBackend(
        "/tmp/fake.zip", inference_model_cls=FakeArrayModel
    )

    backend.close()

    assert backend._model.closed is True
    assert isinstance(backend, PredictionBackend)
