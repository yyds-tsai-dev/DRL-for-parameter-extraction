from pathlib import Path

import numpy as np

from demo.eehemt_helper import eehemt_model_helper
from demo.eehemt_helper.eehemt_model_helper import EEHEMTModelHelper


class _FakeParam:
    def __init__(self, default: float) -> None:
        self.default = default


class _FakeIdsFunction:
    def eval(self, *, temperature, voltages, **params):
        return np.asarray(voltages["br_gisi"], dtype=float) + params["Gain"]


class _FakeModel:
    def __init__(self) -> None:
        self.modelcard = {"Gain": _FakeParam(1.0)}
        self.functions = {"Ids": _FakeIdsFunction()}


def test_simulate_ids_from_json_init_loads_init_values_and_delegates_to_simulate(
    monkeypatch, tmp_path
):
    va_path = tmp_path / "model.va"
    va_path.write_text("// fake va file\n", encoding="utf-8")
    json_path = tmp_path / "modelcard.json"
    json_path.write_text('{"Gain": {"init": 2.5}}', encoding="utf-8")
    monkeypatch.setattr(eehemt_model_helper.verilogae, "load", lambda _: _FakeModel())

    helper = EEHEMTModelHelper(str(va_path))
    simulated = helper.simulate_ids_from_json_init(
        json_path=str(json_path),
        vgs=np.array([0.0, 1.0]),
        vds_voltage=0.5,
    )

    assert np.allclose(simulated, np.array([2.5, 3.5]))
    assert Path(helper.save_params_to_json(str(tmp_path / "saved.json"))).exists()
