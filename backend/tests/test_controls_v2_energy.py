"""Controls training must report the Rust step's complete energy ledger."""

import os

import pytest
import torch

from src.cspace_proxies import ManifoldConfig, ManifoldEnergyInfo, controls_v2_sequence


def _inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    zero = torch.zeros(1)
    return zero, zero, zero, zero, zero, zero, torch.zeros(1, dtype=torch.int64)


def test_sequence_preserves_rust_energy_info_without_fabricating_diagnostics(
    runtime_core_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = ManifoldEnergyInfo(
        kinetic=1.25, potential=2.5, total=3.75, delta_total=-0.125, delta_kinetic=0.5
    )

    def step(
        c_re: float, c_im: float, vx: float, vy: float, *_args: object
    ) -> tuple[float, float, float, float, ManifoldEnergyInfo]:
        return c_re, c_im, vx, vy, expected

    def unavailable(*_args: object) -> float:
        raise RuntimeError("unexpected second geometry evaluation")

    monkeypatch.setattr(runtime_core_module, "controls_integrate_step", step)
    monkeypatch.setattr(runtime_core_module, "manifold_kinetic_energy", unavailable)
    monkeypatch.setattr(runtime_core_module, "manifold_potential_energy", unavailable)
    _, infos = controls_v2_sequence(
        *_inputs(), dt=0.001, initial_c=torch.tensor([0.3 + 0.05j])
    )
    assert len(infos) == 1
    assert (infos[0].kinetic, infos[0].potential, infos[0].total) == (1.25, 2.5, 3.75)
    assert (infos[0].delta_total, infos[0].delta_kinetic) == (-0.125, 0.5)


def test_sequence_matches_real_rust_energy_and_preserves_ste(
    runtime_core_module,
) -> None:
    rc = runtime_core_module
    config = ManifoldConfig(mu=1_000_000.0)
    native_config = rc.ManifoldConfig(
        config.d_ref, config.epsilon, config.lambda_sq, config.kappa, config.mu
    )
    controls = rc.MotionControls(
        direction_x=0.5,
        direction_y=0.25,
        throttle=0.5,
        brake=0.0,
        grip=0.0,
        impulse=0.25,
    )
    point = complex(0.3, 0.05)
    assert rc.manifold_wall_potential(point, native_config) > 1e-9
    expected_re, expected_im, _, _, expected = rc.controls_integrate_step(
        point.real, point.imag, 0.02, -0.01, controls, 0.001, native_config
    )
    direction_x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    throttle = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    trajectory, infos = controls_v2_sequence(
        direction_x,
        torch.tensor([0.25], dtype=torch.float64),
        throttle,
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        torch.tensor([0.25], dtype=torch.float64),
        torch.zeros(1, dtype=torch.int64),
        dt=0.001,
        config=config,
        initial_c=torch.tensor([point], dtype=torch.complex128),
        initial_v=(0.02, -0.01),
    )
    assert trajectory[0].real.item() == pytest.approx(expected_re, abs=1e-6)
    assert trajectory[0].imag.item() == pytest.approx(expected_im, abs=1e-6)
    actual_energy = (
        infos[0].kinetic,
        infos[0].potential,
        infos[0].total,
        infos[0].delta_total,
        infos[0].delta_kinetic,
    )
    expected_energy = (
        expected.kinetic,
        expected.potential,
        expected.total,
        expected.delta_total,
        expected.delta_kinetic,
    )
    assert actual_energy == pytest.approx(expected_energy, rel=0.0, abs=1e-12)
    trajectory[0].real.backward()
    assert throttle.grad is not None and torch.isfinite(throttle.grad).all()
    assert throttle.grad.abs().sum() > 0.0


def test_sequence_propagates_native_refusal_without_returning_partial_results(
    runtime_core_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = ManifoldEnergyInfo(
        kinetic=1.0, potential=2.0, total=3.0, delta_total=0.0, delta_kinetic=0.0
    )
    calls = 0

    def step(
        c_re: float, c_im: float, vx: float, vy: float, *_args: object
    ) -> tuple[float, float, float, float, ManifoldEnergyInfo]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("GeometryUnavailable at endpoint")
        return c_re, c_im, vx, vy, expected

    monkeypatch.setattr(runtime_core_module, "controls_integrate_step", step)
    controls = (
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
    )
    initial = torch.tensor([0.3 + 0.05j, 0.3 + 0.05j])
    with pytest.raises(RuntimeError, match="GeometryUnavailable"):
        controls_v2_sequence(
            *controls, torch.zeros(2, dtype=torch.int64), dt=0.001, initial_c=initial
        )
    assert calls == 2
    assert torch.equal(initial, torch.tensor([0.3 + 0.05j, 0.3 + 0.05j]))
    assert all(torch.count_nonzero(control) == 0 for control in controls)


def test_sequence_refuses_when_native_controls_binding_is_missing(
    runtime_core_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(runtime_core_module, "controls_integrate_step")
    with pytest.raises(RuntimeError, match="controls_integrate_step is unavailable"):
        controls_v2_sequence(
            *_inputs(), dt=0.001, initial_c=torch.tensor([0.3 + 0.05j])
        )


@pytest.mark.skipif(
    os.environ.get("FRACTALSYNC_RETAINED_FIELD_STAGE_TEST") != "1",
    reason="requires the frozen retained-field asset and four-substep profile",
)
def test_retained_field_intermediate_stage_refusal_escapes_training_sequence(
    runtime_core_module,
) -> None:
    assert (
        "operational:9721bb7647a0b8e6"
        in runtime_core_module.geometry_provider_query(complex(0.3, 0.05), 1e-4)
    )
    initial = torch.tensor([complex(0.70899, 0.05)], dtype=torch.complex128)
    before = initial.clone()
    with pytest.raises(RuntimeError, match="RK stage 2:.*GeometryUnavailable"):
        controls_v2_sequence(
            *_inputs(), dt=0.001, initial_c=initial, initial_v=(1.0, 0.0)
        )
    assert torch.equal(initial, before)


def test_optimizer_updates_through_controls_sequence_ste(runtime_core_module) -> None:
    model = torch.nn.Linear(2, 3)
    with torch.no_grad():
        model.weight.fill_(0.1)
        model.bias.fill_(0.2)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    outputs = model(torch.ones(5, 2))
    direction = torch.tanh(outputs[:, :2])
    throttle = torch.sigmoid(outputs[:, 2])
    zero = torch.zeros(5)
    trajectory, infos = controls_v2_sequence(
        direction[:, 0],
        direction[:, 1],
        throttle,
        zero,
        zero,
        zero,
        torch.zeros(5, dtype=torch.int64),
        dt=0.001,
        initial_c=torch.tensor([0.3 + 0.05j] * 5),
    )
    assert len(infos) == 5 and torch.isfinite(trajectory.real).all()
    loss = (trajectory.real - 0.2).square().sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(old, current.detach())
        for old, current in zip(before, model.parameters(), strict=True)
    )
