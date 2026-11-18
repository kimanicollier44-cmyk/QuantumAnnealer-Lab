"""Tests for QuantumAnnealer-Lab: every claim graded against brute force."""
import itertools
import math
import random

import pytest

import main as m


# ---- Ising / QUBO core
def test_ising_two_spin_values():
    J = [[0.0, 1.0], [1.0, 0.0]]
    assert m.ising_energy([1, 1], J, [0.0, 0.0]) == -1.0
    assert m.ising_energy([1, -1], J, [0.0, 0.0]) == 1.0


def test_qubo_ising_energy_match_all_states():
    rng = random.Random(99)
    n = 5
    Q = [[0.0] * n for _ in range(n)]
    lin = [rng.uniform(-2, 2) for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            Q[i][j] = Q[j][i] = rng.uniform(-2, 2)
    J, h, off = m.qubo_to_ising(Q, lin)
    for bits in itertools.product((0, 1), repeat=n):
        spins = [2 * b - 1 for b in bits]
        assert abs(m.qubo_energy(bits, Q, lin)
                   - (m.ising_energy(spins, J, h) + off)) < 1e-9


def test_qubo_ising_roundtrip():
    Q, lin = m._demo_qubo()
    J, h, off = m.qubo_to_ising(Q, lin)
    Q2, lin2, off2 = m.ising_to_qubo(J, h)
    n = len(lin)
    assert all(abs(Q2[i][j] - Q[i][j]) < 1e-9 for i in range(n) for j in range(n))
    assert all(abs(lin2[i] - lin[i]) < 1e-9 for i in range(n))
    assert abs(off2 - off) < 1e-9


def test_qubo_min_matches_ising_ground():
    Q, lin = m._demo_qubo()
    J, h, off = m.qubo_to_ising(Q, lin)
    e, _ = m.brute_force_ising(J, h)
    qmin = min(m.qubo_energy(bits, Q, lin)
               for bits in itertools.product((0, 1), repeat=5))
    assert abs(qmin - (e + off)) < 1e-9


def test_brute_force_guard():
    J = [[0.0] * 21 for _ in range(21)]
    with pytest.raises(ValueError):
        m.brute_force_ising(J, [0.0] * 21)


# ---- annealing machinery
def test_schedule_endpoints_monotonic():
    s = m.geometric_schedule(2.5, 0.02, 100)
    assert abs(s[0] - 2.5) < 1e-12 and abs(s[-1] - 0.02) < 1e-9
    assert all(s[k] > s[k + 1] for k in range(99))


def test_schedule_rejects_bad_inputs():
    with pytest.raises(ValueError):
        m.geometric_schedule(2.5, 0.02, 1)
    with pytest.raises(ValueError):
        m.geometric_schedule(-1.0, 0.02, 10)


def test_anneal_deterministic():
    J, h = m.spin_glass()
    a1 = m.anneal(J, h, seed=11, steps=2000, T0=2.5, T1=0.02)
    a2 = m.anneal(J, h, seed=11, steps=2000, T0=2.5, T1=0.02)
    assert a1["best_energy"] == a2["best_energy"]
    assert a1["best_spins"] == a2["best_spins"]


def test_anneal_bad_moves_raise():
    J, h = m.spin_glass()
    with pytest.raises(ValueError):
        m.anneal(J, h, moves="quantum")


def test_anneal_matches_ground_single_and_cluster():
    rep = m.anneal_report()
    assert rep["ground_energy"] == -18.0
    assert rep["single_matches_ground"] and rep["cluster_matches_ground"]


def test_spinglass_is_frustrated():
    J, h = m.spin_glass()
    assert m.is_frustrated(J, h) is True


def test_ferromagnet_not_frustrated():
    J, h = m._ring_ferromagnet(8)
    assert m.is_frustrated(J, h) is False


# ---- MaxCut, graded against brute force
def test_maxcut_optimum():
    rep = m.maxcut_report()
    assert rep["optimal_cut"] == 20
    assert rep["total_weight"] == 26


def test_maxcut_anneal_matches_optimum():
    rep = m.maxcut_report()
    assert rep["anneal_matches_optimum"] and rep["cluster_matches_optimum"]


def test_maxcut_cut_energy_consistent():
    rep = m.maxcut_report()
    assert abs(rep["optimal_cut"]
               - (rep["total_weight"] / 2.0 - rep["optimal_energy"])) < 1e-9


# ---- TSP, graded against brute force
def test_tsp_optimum_value():
    rep = m.tsp_report()
    assert abs(rep["optimal_length"] - 25.1429) < 1e-3


def test_tsp_anneal_matches_optimum():
    rep = m.tsp_report()
    assert rep["anneal_matches_optimum"]
    assert sorted(rep["anneal_tour"]) == sorted(m.TSP_CITIES)


# ---- bin packing, graded against the lower bound
def test_binpack_ffd_and_anneal_optimal():
    rep = m.binpack_report()
    assert rep["lower_bound_bins"] == 5
    assert rep["ffd_bins"] == 5 and rep["anneal_is_optimal"]


def test_binpack_packing_valid():
    rep = m.binpack_report()
    assert all(sum(b) <= m.BIN_CAPACITY for b in rep["anneal_packing"])
    assert sorted(x for b in rep["anneal_packing"] for x in b) == sorted(m.BIN_ITEMS)


# ---- barrier model: the two laws
def test_thermal_rate_laws():
    assert m.thermal_escape_rate(4.0, 1.0) < m.thermal_escape_rate(2.0, 1.0)
    assert m.thermal_escape_rate(4.0, 2.0) > m.thermal_escape_rate(4.0, 1.0)
    with pytest.raises(ValueError):
        m.thermal_escape_rate(4.0, 0.0)


def test_tunnelling_laws():
    assert abs(m.tunnelling_exponent(3.0, 4.0) - 6.0) < 1e-12
    assert m.tunnelling_rate(4.0, 4.0) < m.tunnelling_rate(2.0, 4.0)


def test_crossover_and_winners():
    assert abs(m.crossover_temperature(3.0, 4.0) - 2.0 / 3.0) < 1e-12
    rows = {r["T"]: r for r in m.barrier_report()["rows"]}
    assert rows[0.5]["winner"] == "tunnelling"
    assert rows[2.0]["winner"] == "thermal"


# ---- freeze-out / TTS
def test_tts_formula_exact():
    assert abs(m.time_to_solution(0.5, 2.0)
               - 2.0 * math.log(0.01) / math.log(0.5)) < 1e-12


def test_tts_edges_and_monotonicity():
    assert math.isinf(m.time_to_solution(0.0, 5.0))
    assert m.time_to_solution(1.0, 5.0) == 5.0
    assert m.time_to_solution(0.8, 10.0) < m.time_to_solution(0.2, 10.0)


def test_tts_report_measured():
    rep = m.tts_report()
    assert rep["full_schedule"]["success_probability"] == 1.0
    assert rep["short_schedule"]["success_probability"] == 0.75
    assert abs(rep["short_schedule"]["tts_99"] - 398.63) < 0.5


# ---- hardware table honesty
def test_hardware_all_labelled():
    hw = m.hardware_report()
    assert hw["all_labelled"] is True and hw["measured_here"] is False
    assert all(r["labelled"] for r in hw["rows"])


# ---- classifier agrees with the barrier model
def test_classifier_scenarios():
    rep = m.regime_report()
    assert rep["ring_ferromagnet_8"]["knn_label"] == "easy"
    assert rep["ring_ferromagnet_12"]["knn_label"] == "hard"
    assert rep["spinglass_12_demo"]["knn_label"] == "frustrated"


def test_classifier_matches_model():
    rep = m.regime_report()
    assert all(r["knn_label"] == r["model_label"] for r in rep.values())


def test_classifier_features():
    rep = m.regime_report()
    assert rep["ring_ferromagnet_8"]["barrier_height"] == 4.0
    assert rep["spinglass_12_demo"]["barrier_height"] == 8.0
    assert rep["spinglass_12_demo"]["frustrated"] is True


# ---- gates + seats
def test_hold_gates_all_present():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


def test_claim_gates_hold():
    for action in ("real_hardware_control", "live_hardware_submission",
                   "quantum_advantage_claim", "tunnelling_overclaim",
                   "guaranteed_optimum_claim",
                   "commercial_optimization_service_claim"):
        assert m.gate(action) == "HOLD"


def test_sandbox_proceeds():
    assert m.gate("run_simulation") == "PROCEED"


def test_seats_never_override():
    b = m.brains_report()
    assert b["decider"]["seat"] == "Gemini Argon 4 Extended Thinking"
    assert b["decider"]["can_override_hold"] is False
    assert b["advisor"]["can_override_hold"] is False
    assert b["flow"]["rendered"] is False and b["flow"]["live"] == 0


def test_demo_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]


def test_demo_deterministic():
    import io
    import contextlib
    out1, out2 = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out1):
        m.demo()
    with contextlib.redirect_stdout(out2):
        m.demo()
    assert out1.getvalue() == out2.getvalue()
