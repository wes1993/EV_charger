"""Tests for the qccdz firmware that streams metering through asynchronous DP 6 frames."""

from __future__ import annotations

import asyncio

import pytest

CHARGING = {
    "1": 117,
    "3": "charger_charging",
    "4": 13,
    "5": 3,
    "6": "CRAAM/QADA0=",
    "10": 0,
    "14": "charge_now",
    "18": True,
    "23": "2.0.0",
    "24": 23,
    "25": 18,
    "27": "online",
}

PAUSED = {
    **CHARGING,
    "3": "charger_pause",
    "6": "CYgAASwAAEk=",
}

IDLE = {
    **CHARGING,
    "3": "charger_free",
    "6": "CZIAAAAAAAA=",
}


def _profile():
    from tuya_ev_charger.dp_profile import DP_PROFILE_MAP

    return DP_PROFILE_MAP["qccdz_dp6_v1"]


def _decode(dps):
    from tuya_ev_charger.charger_metrics import decode_qccdz_dp6_metrics

    return decode_qccdz_dp6_metrics(dps, _profile())


def test_dp6_frame_decodes_voltage_current_and_power():
    metrics = _decode(CHARGING)

    assert metrics.status == "charging"
    assert metrics.work_state_debug == "WORKING"
    assert metrics.voltage_l1 == pytest.approx(232.0)
    assert metrics.current_l1 == pytest.approx(13.3)
    assert metrics.power_l1 == pytest.approx(3.085)
    assert metrics.total_power == pytest.approx(3.085)
    assert metrics.session_energy_kwh == pytest.approx(0.18)
    assert metrics.temperature == pytest.approx(23.0)
    assert metrics.current_target == 13
    assert metrics.max_current_cfg == 32
    assert metrics.charger_info == {"sw_version": "2.0.0"}
    assert metrics.alarm is None


def test_dp6_frame_uses_the_actual_metered_values_when_paused():
    metrics = _decode(PAUSED)

    assert metrics.status == "paused"
    assert metrics.voltage_l1 == pytest.approx(244.0)
    assert metrics.current_l1 == pytest.approx(0.3)
    assert metrics.power_l1 == pytest.approx(0.073)


def test_dp6_zero_frame_is_a_valid_idle_measurement():
    metrics = _decode(IDLE)

    assert metrics.status == "idle"
    assert metrics.voltage_l1 == pytest.approx(245.0)
    assert metrics.current_l1 == 0.0
    assert metrics.power_l1 == 0.0


@pytest.mark.parametrize("raw", ["", "not-base64", "AQI=", 123])
def test_invalid_dp6_frame_is_ignored(raw):
    metrics = _decode({**CHARGING, "6": raw})

    assert metrics.phases == {}
    assert metrics.total_power == 0.0


@pytest.mark.parametrize(
    ("raw_state", "status"),
    [
        ("charger_free", "idle"),
        ("charger_insert", "plugged_in"),
        ("charger_free_fault", "fault"),
        ("charger_wait", "waiting"),
        ("charger_charging", "charging"),
        ("charger_pause", "paused"),
        ("charger_end", "charged"),
        ("charger_fault", "fault"),
    ],
)
def test_dp3_state_maps_to_existing_status_vocabulary(raw_state, status):
    metrics = _decode({**CHARGING, "3": raw_state})

    assert metrics.status == status


def test_profile_assistant_recognises_dp3_layout_without_dp6_in_initial_status():
    from tuya_ev_charger.profile_assistant import suggest_profile

    initial = {key: value for key, value in CHARGING.items() if key != "6"}

    assert suggest_profile(initial) == "qccdz_dp6_v1"


def test_dp6_profile_does_not_expose_unverified_charge_command():
    client = _client(CHARGING)

    assert not client.has_dp("do_charge")
    assert asyncio.run(client.async_set_charge_enabled(True)) is False
    assert client._device.set_calls == []


def test_dp6_profile_accepts_one_amp_steps():
    client = _client(CHARGING)
    calls = []

    async def _send(dp_id, value, verify=True):
        calls.append((dp_id, value, verify))
        return True

    client._async_send_command = _send

    assert asyncio.run(client.async_set_charge_current(14)) is True
    assert calls == [("4", 14, True)]


def test_dp6_profile_rejects_above_32a():
    client = _client(CHARGING)

    with pytest.raises(ValueError):
        asyncio.run(client.async_set_charge_current(33))


class _FakeDevice:
    def __init__(self, dps):
        self._dps = dps
        self.set_calls: list[tuple[str, object]] = []

    def status(self):
        return {"dps": self._dps}

    def set_value(self, dp_id, value):
        self.set_calls.append((dp_id, value))
        return None


def _client(dps):
    from tuya_ev_charger.tuya_ev_charger import TuyaEVChargerClient

    client = TuyaEVChargerClient("dev", "1.2.3.4", "key", "3.5", charger_profile="qccdz_dp6_v1")
    client._device = _FakeDevice(dps)
    return client
