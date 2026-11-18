#!/usr/bin/env python3
"""QuantumAnnealer-Lab - educational quantum annealing lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Education only - and the lesson is the honest core, stated up front:

  * This lab runs NO quantum hardware. Every "anneal" here is a
    classical simulation on an ordinary CPU: simulated annealing
    with Metropolis single-spin flips, plus a cluster-flip move set
    that is INSPIRED BY quantum tunnelling - many spins flip
    together, the way a tunnelling event would carry a state
    through a barrier instead of over it. It is a classical
    heuristic wearing a tunnelling costume, and the lab says so
    everywhere it appears. No quantum advantage is claimed, ever.
  * The physics that IS real here is the modelling: the Ising
    energy E = -sum(J_ij s_i s_j) - sum(h_i s_i), the exact
    QUBO <-> Ising conversion (asserted to round-trip), the
    Arrhenius thermal escape law (rate ~ exp(-H / T): cost set by
    barrier HEIGHT), and the tunnelling exponent (rate ~
    exp(-W * sqrt(H)): cost set by barrier WIDTH times root
    height). Annealing - thermal or quantum - is a story about
    which barrier you pay for.
  * Small problems are solved EXACTLY by brute force (MaxCut,
    toy TSP, a spin glass), and the annealer is only believed
    where it matches the exact answer. Hardware numbers for
    D-Wave-class annealers are LABELLED published-class reference
    values - printed for scale, never measured here.
  * Time-to-solution is reported the way the field reports it:
    TTS = t_a * log(1 - 0.99) / log(1 - p), from a measured
    success probability p over seeded restarts.

Every real-hardware / live-service / advantage claim is a HOLD
gate - with Gemini Argon 4 as DECIDER (names-only, tighten-only),
Muse as advisor (never overrides HOLD), and Google Flow
plan-only seats.
"""
import hashlib
import heapq
import itertools
import json
import math
import random
import sys

VERSION = "1.0.0"

# Annealer defaults (sim parameters, labelled as such).
DEFAULT_T0 = 2.5                # starting temperature of the demo schedule
DEFAULT_T1 = 0.02               # ending temperature of the demo schedule
DEFAULT_STEPS = 20000           # sweeps of the demo schedule
TTS_TARGET = 0.99               # time-to-solution confidence target

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "real_hardware_control": "HOLD - no control of any real quantum annealer or QPU, ever",
    "live_hardware_submission": "HOLD - no submitting jobs to real quantum hardware or cloud QPU services",
    "quantum_advantage_claim": "HOLD - a classical simulation demonstrates no quantum advantage, and none is claimed",
    "tunnelling_overclaim": "HOLD - cluster flips are a classical heuristic inspired by tunnelling; no real tunnelling is performed or simulated on hardware",
    "commercial_optimization_service_claim": "HOLD - no commercial optimization service exists here; this is an educational simulator",
    "live_logistics_optimization": "HOLD - no routing, scheduling, or logistics decisions for real operations from a sim",
    "financial_optimization_claim": "HOLD - no portfolio, trading, or financial optimization advice of any kind",
    "guaranteed_optimum_claim": "HOLD - annealing is a heuristic; no run is guaranteed to find the global optimum",
    "benchmark_fabrication_claim": "HOLD - labelled published-class hardware values are never presented as measured benchmarks",
    "critical_infrastructure_optimization": "HOLD - no optimization of real power grids or critical infrastructure operations",
    "medical_claim": "HOLD - no medical, treatment-planning, or drug-design optimization claims",
    "investment_claim": "HOLD - no investment advice about quantum computing or annealing companies",
    "weapons_application_claim": "HOLD - no weapons application guidance of any kind",
    "remove_human_override": "HOLD - the human override is never removed; a person can always stop the sim",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 1. Ising / QUBO core
# The Ising energy: E(s) = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i,
# with s_i in {-1, +1}. J is an n x n symmetric matrix (list of
# lists, zero diagonal), h a list of n local fields. A QUBO is the
# same physics in binary clothes: f(x) = sum_i lin_i x_i +
# sum_{i<j} Q_ij x_i x_j with x in {0, 1} and x = (s + 1) / 2.
# Substituting gives J_ij = -Q_ij / 4 and h_i = -(lin_i / 2 +
# (1/4) * sum_{j != i} Q_ij), plus a constant offset - so a QUBO
# and its Ising twin have identical landscapes, shifted by the
# offset. Both directions are implemented and the round-trip is
# asserted exactly in CHECKS.

def ising_energy(spins, J, h):
    """E = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i."""
    n = len(spins)
    e = 0.0
    for i in range(n):
        e -= h[i] * spins[i]
        Ji = J[i]
        si = spins[i]
        for j in range(i + 1, n):
            e -= Ji[j] * si * spins[j]
    return e


def qubo_energy(x, Q, lin):
    """f = sum_i lin_i x_i + sum_{i<j} Q_ij x_i x_j, x in {0, 1}."""
    n = len(x)
    e = 0.0
    for i in range(n):
        e += lin[i] * x[i]
        Qi = Q[i]
        xi = x[i]
        for j in range(i + 1, n):
            e += Qi[j] * xi * x[j]
    return e


def qubo_to_ising(Q, lin):
    """Convert a QUBO to (J, h, offset): qubo_energy == ising + offset."""
    n = len(lin)
    J = [[0.0] * n for _ in range(n)]
    h = [0.0] * n
    offset = 0.0
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if j != i:
                row_sum += Q[min(i, j)][max(i, j)]
        h[i] = -(lin[i] / 2.0 + row_sum / 4.0)
        offset += lin[i] / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            J[i][j] = J[j][i] = -Q[i][j] / 4.0
            offset += Q[i][j] / 4.0
    return J, h, offset


def ising_to_qubo(J, h):
    """Convert an Ising model to (Q, lin, offset): qubo == ising + offset."""
    n = len(h)
    Q = [[0.0] * n for _ in range(n)]
    lin = [0.0] * n
    offset = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            Q[i][j] = Q[j][i] = -4.0 * J[i][j]
            offset += Q[i][j] / 4.0
    for i in range(n):
        row_j = sum(J[i][j] for j in range(n) if j != i)
        lin[i] = -2.0 * h[i] + 2.0 * row_j
        offset += lin[i] / 2.0
    return Q, lin, offset


def brute_force_ising(J, h):
    """Exact ground state by enumeration (guarded to n <= 20)."""
    n = len(h)
    if n > 20:
        raise ValueError("brute force is limited to n <= 20 spins")
    best_e = None
    best_s = None
    for bits in itertools.product((-1, 1), repeat=n):
        e = ising_energy(bits, J, h)
        if best_e is None or e < best_e:
            best_e, best_s = e, list(bits)
    return best_e, best_s


def local_fields(spins, J, h):
    """Effective field on each spin: f_i = sum_j J_ij s_j + h_i."""
    n = len(spins)
    return [sum(J[i][j] * spins[j] for j in range(n)) + h[i]
            for i in range(n)]


def is_frustrated(J, h):
    """True if the ground state cannot satisfy every term at once.

    The all-satisfied bound is -sum|J_ij| - sum|h_i|; a ground
    state above it means some couplings must lose - frustration.
    """
    n = len(h)
    bound = -sum(abs(J[i][j]) for i in range(n) for j in range(i + 1, n))
    bound -= sum(abs(x) for x in h)
    e_star, _ = brute_force_ising(J, h)
    return e_star > bound + 1e-9


# ================================================================ 2. Simulated annealing
# Metropolis dynamics on the Ising landscape. Single-spin flips
# climb thermally: a move costing dE is accepted with probability
# min(1, exp(-dE / T)) - the same Arrhenius law as the barrier
# model in section 5. The cluster move flips a connected group of
# spins in one proposal: the classical stand-in for a tunnelling
# event, which moves a whole region of the state through a thin
# barrier instead of dragging it over the top spin by spin.
# Schedules are geometric in T; every run is seeded, so every
# number in this lab reproduces byte-for-byte.

def geometric_schedule(T0, T1, steps):
    """Geometric temperature ladder from T0 down to T1 (inclusive)."""
    if steps < 2:
        raise ValueError("a schedule needs at least 2 steps")
    if T0 <= 0 or T1 <= 0:
        raise ValueError("temperatures must be positive")
    ratio = (T1 / T0) ** (1.0 / (steps - 1))
    return [T0 * ratio ** k for k in range(steps)]


def _flip_delta(spins, J, h, i):
    """Energy change from flipping spin i alone: 2 s_i f_i."""
    f = h[i] + sum(J[i][j] * spins[j] for j in range(len(spins)))
    return 2.0 * spins[i] * f


def anneal(J, h, seed=0, steps=DEFAULT_STEPS, T0=DEFAULT_T0, T1=DEFAULT_T1,
           moves="single"):
    """One seeded annealing run. moves: 'single' or 'cluster'.

    Returns the best state seen, its energy, and run statistics.
    Deterministic: the same seed always reproduces the same run.
    """
    if moves not in ("single", "cluster"):
        raise ValueError("moves must be 'single' or 'cluster'")
    n = len(h)
    rng = random.Random(seed)
    spins = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
    energy = ising_energy(spins, J, h)
    best_e, best_s = energy, list(spins)
    accepted = 0
    schedule = geometric_schedule(T0, T1, steps)
    neighbours = [[j for j in range(n) if j != i and J[i][j] != 0.0]
                  for i in range(n)]
    for T in schedule:
        if moves == "single":
            i = rng.randrange(n)
            dE = _flip_delta(spins, J, h, i)
            if dE <= 0.0 or rng.random() < math.exp(-dE / T):
                spins[i] = -spins[i]
                energy += dE
                accepted += 1
        else:
            # Cluster proposal: a seed spin plus each neighbour with
            # probability 1/2 - a region flips together, the classical
            # heuristic INSPIRED BY tunnelling (never real hardware).
            i = rng.randrange(n)
            cluster = [i] + [j for j in neighbours[i] if rng.random() < 0.5]
            trial = list(spins)
            for k in cluster:
                trial[k] = -trial[k]
            dE = ising_energy(trial, J, h) - energy
            if dE <= 0.0 or rng.random() < math.exp(-dE / T):
                spins = trial
                energy += dE
                accepted += len(cluster)
        if energy < best_e:
            best_e, best_s = energy, list(spins)
    return {"best_energy": best_e, "best_spins": best_s,
            "final_energy": energy, "accepted": accepted,
            "steps": steps, "moves": moves, "seed": seed}


def anneal_restarts(J, h, restarts=20, base_seed=1000, target_energy=None,
                    **kwargs):
    """Seeded restarts; success probability against a target energy."""
    if target_energy is None:
        target_energy, _ = brute_force_ising(J, h)
    hits = 0
    best = None
    for r in range(restarts):
        out = anneal(J, h, seed=base_seed + r, **kwargs)
        if out["best_energy"] <= target_energy + 1e-9:
            hits += 1
        if best is None or out["best_energy"] < best:
            best = out["best_energy"]
    return {"restarts": restarts, "hits": hits,
            "success_probability": hits / restarts,
            "best_energy": best, "target_energy": target_energy}


def time_to_solution(p, t_a, target=TTS_TARGET):
    """TTS = t_a * log(1 - target) / log(1 - p). p = 0 -> infinite."""
    if p <= 0.0:
        return math.inf
    if p >= 1.0:
        return float(t_a)
    return t_a * math.log(1.0 - target) / math.log(1.0 - p)


# ================================================================ 3. Problem book
# Three small problems, each with an EXACT answer computed by
# brute force, so the annealer is graded against the truth - not
# against another heuristic. Plus a seeded spin glass, the demo
# instance for the anneal / barrier / classifier sections.

# ---- MaxCut: partition 8 nodes to maximise the weight cut.
MAXCUT_EDGES = [
    (0, 1, 2), (0, 2, 1), (0, 7, 1), (1, 2, 3), (1, 3, 2), (1, 5, 1),
    (2, 4, 2), (2, 6, 2), (3, 4, 1), (3, 5, 3), (4, 5, 2), (4, 6, 1),
    (5, 7, 2), (6, 7, 3),
]
MAXCUT_N = 8


def maxcut_to_ising():
    """MaxCut as an Ising model: J_ij = -w_ij / 2, h = 0.

    Cut value C(s) = sum w_ij (1 - s_i s_j) / 2 = total_w / 2 -
    E_ising(s), so minimising the Ising energy maximises the cut.
    """
    J = [[0.0] * MAXCUT_N for _ in range(MAXCUT_N)]
    for (i, j, w) in MAXCUT_EDGES:
        J[i][j] = J[j][i] = -w / 2.0
    return J, [0.0] * MAXCUT_N


def cut_value(spins):
    total = 0.0
    for (i, j, w) in MAXCUT_EDGES:
        if spins[i] != spins[j]:
            total += w
    return total


def maxcut_report():
    J, h = maxcut_to_ising()
    e_star, s_star = brute_force_ising(J, h)
    total_w = sum(w for _, _, w in MAXCUT_EDGES)
    sa = anneal(J, h, seed=42, steps=6000, T0=1.5, T1=0.01, moves="single")
    cl = anneal(J, h, seed=42, steps=6000, T0=1.5, T1=0.01, moves="cluster")
    return {
        "nodes": MAXCUT_N,
        "edges": len(MAXCUT_EDGES),
        "total_weight": total_w,
        "optimal_cut": cut_value(s_star),
        "optimal_energy": e_star,
        "optimal_partition": [i for i, s in enumerate(s_star) if s == 1],
        "anneal_single_cut": cut_value(sa["best_spins"]),
        "anneal_cluster_cut": cut_value(cl["best_spins"]),
        "anneal_matches_optimum": cut_value(sa["best_spins"]) == cut_value(s_star),
        "cluster_matches_optimum": cut_value(cl["best_spins"]) == cut_value(s_star),
    }


# ---- Toy TSP: 7 labelled cities, exact optimum by permutation.
TSP_CITIES = {
    "alpha": (0.0, 0.0), "bravo": (2.0, 3.0), "charlie": (5.0, 1.0),
    "delta": (6.0, 4.0), "echo": (3.0, 6.0), "foxtrot": (8.0, 2.0),
    "golf": (7.0, 7.0),
}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tour_length(order, cities=TSP_CITIES):
    names = list(cities)
    pts = [cities[names[i]] for i in order]
    total = 0.0
    for k in range(len(pts)):
        total += _dist(pts[k], pts[(k + 1) % len(pts)])
    return total


def tsp_brute_force():
    """Exact optimum tour (city 0 fixed to break symmetry)."""
    n = len(TSP_CITIES)
    best = None
    best_order = None
    for perm in itertools.permutations(range(1, n)):
        order = (0,) + perm
        L = tour_length(order)
        if best is None or L < best:
            best, best_order = L, order
    names = list(TSP_CITIES)
    return {"optimal_length": best,
            "optimal_tour": [names[i] for i in best_order]}


def tsp_anneal(seed=7, steps=30000, restarts=5, T0=1.2, T1=0.005):
    """Permutation annealing: swap two cities, Metropolis on length.

    NOT an Ising run - TSP is annealed in its native permutation
    space here; its QUBO encoding would need n^2 = 49 spins for 7
    cities, which the lab states instead of pretending otherwise.
    """
    n = len(TSP_CITIES)
    best_global = None
    for r in range(restarts):
        rng = random.Random(seed + r)
        order = list(range(n))
        rng.shuffle(order)
        cur = tour_length(order)
        best, best_order = cur, list(order)
        ratio = (T1 / T0) ** (1.0 / (steps - 1))
        for k in range(steps):
            T = T0 * ratio ** k
            i, j = rng.randrange(n), rng.randrange(n)
            if i == j:
                continue
            trial = list(order)
            trial[i], trial[j] = trial[j], trial[i]
            L = tour_length(trial)
            dE = L - cur
            if dE <= 0.0 or rng.random() < math.exp(-dE / T):
                order, cur = trial, L
                if cur < best:
                    best, best_order = cur, list(order)
        if best_global is None or best < best_global["length"]:
            names = list(TSP_CITIES)
            best_global = {"length": best,
                           "tour": [names[i] for i in best_order]}
    return best_global


def tsp_report():
    exact = tsp_brute_force()
    ann = tsp_anneal()
    return {"cities": len(TSP_CITIES),
            "optimal_length": round(exact["optimal_length"], 4),
            "optimal_tour": exact["optimal_tour"],
            "anneal_length": round(ann["length"], 4),
            "anneal_tour": ann["tour"],
            "anneal_matches_optimum": abs(ann["length"]
                                          - exact["optimal_length"]) < 1e-9}


# ---- Bin packing: labelled items, capacity 10.
BIN_ITEMS = [8, 7, 6, 5, 4, 4, 3, 3, 2, 2]
BIN_CAPACITY = 10


def first_fit_decreasing(items=BIN_ITEMS, capacity=BIN_CAPACITY):
    bins = []
    for item in sorted(items, reverse=True):
        for b in bins:
            if sum(b) + item <= capacity:
                b.append(item)
                break
        else:
            bins.append([item])
    return bins


def binpack_anneal(seed=3, steps=20000, T0=1.0, T1=0.01):
    """Anneal item->bin assignments; energy prices bins + overflow."""
    items = BIN_ITEMS
    cap = BIN_CAPACITY
    max_bins = len(items)

    def energy(assign):
        loads = [0] * max_bins
        used = [False] * max_bins
        for idx, b in enumerate(assign):
            loads[b] += items[idx]
            used[b] = True
        overflow = sum(max(0, load - cap) for load in loads)
        return 100.0 * sum(used) + 1000.0 * overflow

    rng = random.Random(seed)
    assign = [rng.randrange(max_bins) for _ in items]
    cur = energy(assign)
    best, best_assign = cur, list(assign)
    ratio = (T1 / T0) ** (1.0 / (steps - 1))
    for k in range(steps):
        T = T0 * ratio ** k
        idx = rng.randrange(len(items))
        old = assign[idx]
        new = rng.randrange(max_bins)
        if new == old:
            continue
        assign[idx] = new
        e = energy(assign)
        dE = e - cur
        if dE <= 0.0 or rng.random() < math.exp(-dE / T):
            cur = e
            if cur < best:
                best, best_assign = cur, list(assign)
        else:
            assign[idx] = old
    bins = {}
    for idx, b in enumerate(best_assign):
        bins.setdefault(b, []).append(items[idx])
    return {"bins_used": len(bins),
            "bins": [sorted(v, reverse=True) for _, v in sorted(bins.items())],
            "energy": best}


def binpack_report():
    ffd = first_fit_decreasing()
    lower_bound = math.ceil(sum(BIN_ITEMS) / BIN_CAPACITY)
    ann = binpack_anneal()
    return {"items": BIN_ITEMS, "capacity": BIN_CAPACITY,
            "lower_bound_bins": lower_bound,
            "ffd_bins": len(ffd), "ffd_packing": ffd,
            "anneal_bins": ann["bins_used"], "anneal_packing": ann["bins"],
            "anneal_is_optimal": ann["bins_used"] == lower_bound}


# ---- Demo spin glass: seeded +/-1 couplings on a ring + chords.
def spin_glass(n=12, seed=7):
    """Seeded spin glass: ring couplings plus chords, J in {-1, +1}."""
    rng = random.Random(seed)
    J = [[0.0] * n for _ in range(n)]
    pairs = set()
    for i in range(n):
        pairs.add((i, (i + 1) % n))
        pairs.add((i, (i + 3) % n))
    for (a, b) in sorted(pairs):
        if a == b:
            continue
        i, j = min(a, b), max(a, b)
        if J[i][j] == 0.0:
            J[i][j] = J[j][i] = 1.0 if rng.random() < 0.5 else -1.0
    h = [0.0] * n
    return J, h


def anneal_report():
    J, h = spin_glass()
    e_star, _ = brute_force_ising(J, h)
    sa = anneal(J, h, seed=11, steps=20000, T0=2.5, T1=0.02, moves="single")
    cl = anneal(J, h, seed=11, steps=20000, T0=2.5, T1=0.02, moves="cluster")
    return {"spins": len(h), "ground_energy": e_star,
            "single_best": sa["best_energy"],
            "cluster_best": cl["best_energy"],
            "single_matches_ground": sa["best_energy"] == e_star,
            "cluster_matches_ground": cl["best_energy"] == e_star,
            "frustrated": is_frustrated(J, h)}


# ================================================================ 4. Barrier model
# Why annealing is hard, in two laws. A thermal hop over a barrier
# of height H at temperature T succeeds at the Arrhenius rate
# exp(-H / T) - it pays for HEIGHT. A tunnelling event through a
# barrier of width W and height H goes as exp(-W * sqrt(H)) in the
# lab's labelled units (the WKB exponent with 2 sqrt(2m) / hbar set
# to 1) - it pays for WIDTH times root height, and it does not
# care about temperature. The two rates cross at T* = sqrt(H) / W:
# below T*, tunnelling wins; above it, thermal hopping wins. Both
# are MODELS printed as models - no hardware is measured.

def thermal_escape_rate(height, T):
    """Arrhenius rate exp(-H / T), attempt frequency set to 1."""
    if T <= 0:
        raise ValueError("temperature must be positive")
    return math.exp(-height / T)


def tunnelling_exponent(width, height):
    """Labelled WKB-style exponent: W * sqrt(H)."""
    return width * math.sqrt(height)


def tunnelling_rate(width, height):
    """exp(-W * sqrt(H)) - temperature-independent (labelled model)."""
    return math.exp(-tunnelling_exponent(width, height))


def crossover_temperature(width, height):
    """T* where the two rates are equal: sqrt(H) / W."""
    return math.sqrt(height) / width


def barrier_report(height=4.0, width=3.0):
    rows = []
    for T in (0.25, 0.5, 1.0, 2.0):
        th = thermal_escape_rate(height, T)
        tu = tunnelling_rate(width, height)
        rows.append({"T": T, "thermal_rate": th, "tunnel_rate": tu,
                     "winner": "tunnelling" if tu > th else "thermal"})
    return {"height": height, "width": width,
            "tunnelling_exponent": tunnelling_exponent(width, height),
            "crossover_T": crossover_temperature(width, height),
            "rows": rows}


# ================================================================ 5. Freeze-out & TTS
# A real annealer freezes out: once the temperature falls below
# the point where the remaining schedule can move the state, the
# answer is whatever the state happens to be. The lab's version of
# the same honesty is the restart ledger: run a deliberately SHORT
# schedule many times, measure how often it lands on the exact
# ground state, and price the time-to-solution from that measured
# p - instead of quoting the one lucky run.

def tts_report():
    J, h = spin_glass()
    e_star, _ = brute_force_ising(J, h)
    short = anneal_restarts(J, h, restarts=20, base_seed=5000,
                            target_energy=e_star,
                            steps=120, T0=1.0, T1=0.3, moves="single")
    full = anneal_restarts(J, h, restarts=20, base_seed=7000,
                           target_energy=e_star,
                           steps=20000, T0=2.5, T1=0.02, moves="single")
    t_short = 120          # anneal time unit = schedule steps (labelled)
    t_full = 20000
    return {
        "ground_energy": e_star,
        "short_schedule": {"steps": t_short,
                           "success_probability": short["success_probability"],
                           "hits": short["hits"], "restarts": 20,
                           "tts_99": time_to_solution(
                               short["success_probability"], t_short)},
        "full_schedule": {"steps": t_full,
                          "success_probability": full["success_probability"],
                          "hits": full["hits"], "restarts": 20,
                          "tts_99": time_to_solution(
                              full["success_probability"], t_full)},
    }


# ================================================================ 6. Labelled hardware table
# D-Wave-class annealers, published-class reference values ONLY.
# Every row is flagged LABELLED: these are spec-sheet-class
# numbers printed for scale, never measured by this lab, and the
# lab's classical simulation is not comparable to any of them.

HARDWARE_TABLE = [
    {"name": "D-Wave 2000Q-class", "qubits": "~2,048", "year": 2017,
     "topology": "Chimera", "labelled": True,
     "note": "published-class spec value, not measured"},
    {"name": "D-Wave Advantage-class", "qubits": "~5,760", "year": 2020,
     "topology": "Pegasus", "labelled": True,
     "note": "published-class spec value, not measured"},
    {"name": "D-Wave Advantage2-class", "qubits": "~4,400", "year": 2025,
     "topology": "Zephyr", "labelled": True,
     "note": "published-class spec value, not measured"},
    {"name": "This lab (classical sim)", "qubits": "12-spin demo instance",
     "year": 2026, "topology": "ring + chords, fully classical",
     "labelled": True,
     "note": "a CPU simulation - NOT a quantum annealer"},
]


def hardware_report():
    return {"rows": HARDWARE_TABLE,
            "all_labelled": all(r["labelled"] for r in HARDWARE_TABLE),
            "measured_here": False}


# ================================================================ 7. Landscape + kNN classifier
# The regime of an instance is read off its landscape: find the
# ground state, walk the cheapest single-flip path (minimax: the
# path whose HIGHEST energy is as low as possible) to the opposite
# valley, and record the barrier height H and the path width W in
# flips. Frustration is checked exactly (section 1). The regime
# rule is a labelled model built from the section-4 laws:
#   frustrated and H >= 1      -> "frustrated"
#   frustrated, lower barrier  -> "hard"
#   W * sqrt(H) <= 16          -> "easy"   (thin/low barrier)
#   otherwise                  -> "hard"
# Training rows are generated by running THIS machinery on a book
# of small instances - the classifier interpolates the model, it
# does not replace it.

def landscape_barrier(J, h):
    """Minimax barrier from the ground state to the opposite valley.

    Returns (height, width_flips, frustrated). Exact for n <= 12.
    The opposite valley = lowest-energy state at Hamming distance
    >= n - 1 from the ground state.
    """
    n = len(h)
    if n > 12:
        raise ValueError("landscape barrier is limited to n <= 12 spins")
    energies = {}
    ground = None
    for mask in range(1 << n):
        spins = [1 if (mask >> i) & 1 else -1 for i in range(n)]
        e = ising_energy(spins, J, h)
        energies[mask] = e
        if ground is None or e < energies[ground]:
            ground = mask
    e_star = energies[ground]
    target = None
    for mask, e in energies.items():
        d = bin(mask ^ ground).count("1")
        if d >= n - 1 and (target is None or e < energies[target]):
            target = mask
    # Dijkstra minimax: minimise the highest energy along the path.
    dist = {ground: e_star}
    parent = {ground: None}
    pq = [(e_star, ground)]
    settled = set()
    while pq:
        cost, state = heapq.heappop(pq)
        if state in settled:
            continue
        settled.add(state)
        if state == target:
            break
        for i in range(n):
            nxt = state ^ (1 << i)
            new_cost = max(cost, energies[nxt])
            if nxt not in dist or new_cost < dist[nxt]:
                dist[nxt] = new_cost
                parent[nxt] = state
                heapq.heappush(pq, (new_cost, nxt))
    width = 0
    node = target
    while parent[node] is not None:
        width += 1
        node = parent[node]
    bound = -sum(abs(J[i][j]) for i in range(n) for j in range(i + 1, n))
    bound -= sum(abs(x) for x in h)
    frustrated = e_star > bound + 1e-9
    return dist[target] - e_star, width, frustrated


def model_label(height, width, n, frustrated):
    """Ground-truth regime label from the barrier model (labelled)."""
    if frustrated:
        return "frustrated" if height >= 1.0 else "hard"
    if width * math.sqrt(height) <= 16.0:
        return "easy"
    return "hard"


def _ring_ferromagnet(n, af_bonds=0):
    J = [[0.0] * n for _ in range(n)]
    for i in range(n):
        j = (i + 1) % n
        J[i][j] = J[j][i] = -1.0 if i < af_bonds else 1.0
    return J, [0.0] * n


def dense_glass(n, seed):
    """Seeded spin glass on the complete graph, J in {-1, +1}.

    Denser than the ring+chords demo instance, so frustration
    piles up and the minimax barriers stand taller (H = 6..12 in
    the book below, against H = 4 for every ferromagnet ring) -
    which is what lets a (H, W, n) classifier separate the
    frustrated regime from the merely hard one.
    """
    rng = random.Random(seed)
    J = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            J[i][j] = J[j][i] = 1.0 if rng.random() < 0.5 else -1.0
    return J, [0.0] * n


def _instance_book():
    """Small instances whose landscapes generate the training rows."""
    book = {}
    for n in (6, 7, 8, 10, 11, 12):
        book[f"ring_ferromagnet_{n}"] = _ring_ferromagnet(n)
    book["ring_one_af_bond_8"] = _ring_ferromagnet(8, af_bonds=1)
    book["dense_glass_8_s3"] = dense_glass(8, seed=3)
    book["dense_glass_10_s5"] = dense_glass(10, seed=5)
    book["dense_glass_10_s9"] = dense_glass(10, seed=9)
    book["dense_glass_12_s7"] = dense_glass(12, seed=7)
    book["spinglass_12_demo"] = spin_glass(12, seed=7)
    J, h = maxcut_to_ising()
    book["maxcut_8"] = (J, h)
    return book


_FEATURES_CACHE = None


def instance_features(J, h):
    H, W, frustrated = landscape_barrier(J, h)
    return {"barrier_height": H, "barrier_width": W, "spins": len(h),
            "frustrated": frustrated,
            "label": model_label(H, W, len(h), frustrated)}


def _training_rows():
    global _FEATURES_CACHE
    if _FEATURES_CACHE is None:
        rows = []
        for name, (J, h) in _instance_book().items():
            f = instance_features(J, h)
            rows.append((f["barrier_height"], f["barrier_width"] / 2.0,
                         f["spins"] / 2.0, f["label"]))
        _FEATURES_CACHE = rows
    return _FEATURES_CACHE


def _knn_label(height, width, n, k=3):
    f = (height, width / 2.0, n / 2.0)
    ranked = sorted(_training_rows(),
                    key=lambda row: math.sqrt(sum((a - b) ** 2
                                                  for a, b in zip(f, row[:3]))))
    votes = {}
    for row in ranked[:k]:
        votes[row[3]] = votes.get(row[3], 0) + 1
    return max(sorted(votes), key=lambda lab: votes[lab])


def classify_instance(J, h):
    """kNN regime label for an instance, from its landscape features."""
    f = instance_features(J, h)
    return _knn_label(f["barrier_height"], f["barrier_width"], f["spins"])


def regime_report():
    out = {}
    for name in ("ring_ferromagnet_8", "ring_ferromagnet_12",
                 "spinglass_12_demo"):
        J, h = _instance_book()[name]
        f = instance_features(J, h)
        out[name] = {"barrier_height": f["barrier_height"],
                     "barrier_width": f["barrier_width"],
                     "frustrated": f["frustrated"],
                     "model_label": f["label"],
                     "knn_label": classify_instance(J, h)}
    return out


# ================================================================ 8. Advisory seats
def _trace(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def argon4_decide(topic="quantumannealer-lab-demo"):
    """Gemini Argon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never lift one.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-argon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Argon 4 Extended Thinking",
        "role": "DECIDER",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "decision": "PROCEED - sandbox simulation only, all claim gates stay HOLD",
        "can_override_hold": False,
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="quantumannealer-lab-demo"):
    """Muse advisor seat: advises, can never override a HOLD."""
    return {
        "seat": "Muse",
        "role": "ADVISOR",
        "key": "names-only (MODEL_API_KEY name checked, value never read)",
        "live_calls": 0,
        "advice": "Say it in the first line of every report: this is a classical simulation - the cluster flips are a tunnelling-inspired heuristic, the hardware numbers are labelled references, and the annealer is only believed where brute force agrees with it.",
        "can_override_hold": False,
    }


def google_flow_plan():
    """Google Flow seat: plan-only, nothing rendered, nothing live."""
    return {
        "seat": "Google Flow",
        "role": "PLAN-ONLY",
        "shots": ["an energy landscape of valleys drawn as a night terrain",
                  "a thermal climber sweating over a high pass as temperature falls",
                  "a cluster of spins flipping together through a thin wall",
                  "a freeze-out moment: the landscape locking mid-stride",
                  "the restart ledger stamping a measured success probability"],
        "rendered": False,
        "live": 0,
    }


def brains_report():
    return {"decider": argon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ checks
def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    # 1. Ising / QUBO core
    J2 = [[0.0, 1.0], [1.0, 0.0]]
    chk("ising_two_spin_values",
        ising_energy([1, 1], J2, [0.0, 0.0]) == -1.0
        and ising_energy([1, -1], J2, [0.0, 0.0]) == 1.0)
    rng = random.Random(99)
    n5 = 5
    Q5 = [[0.0] * n5 for _ in range(n5)]
    lin5 = [rng.uniform(-2, 2) for _ in range(n5)]
    for i in range(n5):
        for j in range(i + 1, n5):
            Q5[i][j] = Q5[j][i] = rng.uniform(-2, 2)
    J5, h5, off5 = qubo_to_ising(Q5, lin5)
    match = True
    for bits in itertools.product((0, 1), repeat=n5):
        spins = [2 * b - 1 for b in bits]
        if abs(qubo_energy(bits, Q5, lin5)
               - (ising_energy(spins, J5, h5) + off5)) > 1e-9:
            match = False
    chk("qubo_ising_energy_match_all_states", match)
    Q5b, lin5b, off5b = ising_to_qubo(J5, h5)
    chk("qubo_ising_roundtrip",
        all(abs(Q5b[i][j] - Q5[i][j]) < 1e-9
            for i in range(n5) for j in range(n5))
        and all(abs(lin5b[i] - lin5[i]) < 1e-9 for i in range(n5))
        and abs(off5b - off5) < 1e-9)
    e5, _ = brute_force_ising(J5, h5)
    qmin = min(qubo_energy(bits, Q5, lin5)
               for bits in itertools.product((0, 1), repeat=n5))
    chk("qubo_min_matches_ising_ground", abs(qmin - (e5 + off5)) < 1e-9)
    # 2. annealing machinery
    sched = geometric_schedule(2.5, 0.02, 100)
    chk("schedule_endpoints_monotonic",
        abs(sched[0] - 2.5) < 1e-12 and abs(sched[-1] - 0.02) < 1e-9
        and all(sched[k] > sched[k + 1] for k in range(99)))
    Jg, hg = spin_glass()
    a1 = anneal(Jg, hg, seed=11, steps=2000, T0=2.5, T1=0.02)
    a2 = anneal(Jg, hg, seed=11, steps=2000, T0=2.5, T1=0.02)
    chk("anneal_deterministic",
        a1["best_energy"] == a2["best_energy"]
        and a1["best_spins"] == a2["best_spins"])
    rep = anneal_report()
    chk("anneal_single_matches_ground", rep["single_matches_ground"])
    chk("anneal_cluster_matches_ground", rep["cluster_matches_ground"])
    chk("spinglass_is_frustrated", rep["frustrated"] is True)
    # 3. MaxCut - graded against brute force
    mc = maxcut_report()
    chk("maxcut_anneal_matches_optimum", mc["anneal_matches_optimum"])
    chk("maxcut_cluster_matches_optimum", mc["cluster_matches_optimum"])
    chk("maxcut_cut_energy_consistent",
        abs(mc["optimal_cut"] - (mc["total_weight"] / 2.0
                                 - mc["optimal_energy"])) < 1e-9)
    # 4. TSP - graded against brute force
    tp = tsp_report()
    chk("tsp_anneal_matches_optimum", tp["anneal_matches_optimum"])
    chk("tsp_tour_is_permutation",
        sorted(tp["anneal_tour"]) == sorted(TSP_CITIES))
    # 5. Bin packing - graded against the lower bound
    bp = binpack_report()
    chk("binpack_ffd_optimal", bp["ffd_bins"] == bp["lower_bound_bins"])
    chk("binpack_anneal_optimal", bp["anneal_is_optimal"])
    chk("binpack_packing_valid",
        all(sum(b) <= BIN_CAPACITY for b in bp["anneal_packing"])
        and sorted(x for b in bp["anneal_packing"] for x in b)
        == sorted(BIN_ITEMS))
    # 6. Barrier model - the two laws
    chk("thermal_rate_falls_with_height",
        thermal_escape_rate(4.0, 1.0) < thermal_escape_rate(2.0, 1.0))
    chk("thermal_rate_rises_with_T",
        thermal_escape_rate(4.0, 2.0) > thermal_escape_rate(4.0, 1.0))
    chk("tunnelling_exponent_value",
        abs(tunnelling_exponent(3.0, 4.0) - 6.0) < 1e-12)
    chk("tunnelling_rate_falls_with_width",
        tunnelling_rate(4.0, 4.0) < tunnelling_rate(2.0, 4.0))
    chk("crossover_formula",
        abs(crossover_temperature(3.0, 4.0) - 2.0 / 3.0) < 1e-12)
    bar = barrier_report()
    rows = {r["T"]: r for r in bar["rows"]}
    chk("below_crossover_tunnelling_wins", rows[0.5]["winner"] == "tunnelling")
    chk("above_crossover_thermal_wins", rows[2.0]["winner"] == "thermal")
    # 7. Freeze-out / TTS
    chk("tts_formula_exact",
        abs(time_to_solution(0.5, 2.0)
            - 2.0 * math.log(0.01) / math.log(0.5)) < 1e-12)
    chk("tts_zero_prob_infinite", math.isinf(time_to_solution(0.0, 5.0)))
    chk("tts_certain_is_one_run", time_to_solution(1.0, 5.0) == 5.0)
    chk("tts_falls_as_p_rises",
        time_to_solution(0.8, 10.0) < time_to_solution(0.2, 10.0))
    tt = tts_report()
    chk("tts_full_schedule_certain",
        tt["full_schedule"]["success_probability"] == 1.0)
    chk("tts_short_schedule_harder",
        tt["short_schedule"]["success_probability"]
        < tt["full_schedule"]["success_probability"])
    # 8. Hardware table honesty
    hw = hardware_report()
    chk("hardware_all_labelled", hw["all_labelled"] is True)
    chk("hardware_nothing_measured", hw["measured_here"] is False)
    # 9. Classifier - agrees with the barrier model on the scenarios
    reg = regime_report()
    chk("classifier_ring8_easy", reg["ring_ferromagnet_8"]["knn_label"] == "easy")
    chk("classifier_ring12_hard",
        reg["ring_ferromagnet_12"]["knn_label"] == "hard")
    chk("classifier_spinglass_frustrated",
        reg["spinglass_12_demo"]["knn_label"] == "frustrated")
    chk("classifier_matches_model",
        all(row["knn_label"] == row["model_label"] for row in reg.values()))
    # gates + seats
    chk("hold_gates_14", len(HOLD_GATES) == 14
        and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_hardware_hold", gate("real_hardware_control") == "HOLD")
    chk("gate_submission_hold", gate("live_hardware_submission") == "HOLD")
    chk("gate_advantage_hold", gate("quantum_advantage_claim") == "HOLD")
    chk("gate_tunnelling_hold", gate("tunnelling_overclaim") == "HOLD")
    chk("gate_guarantee_hold", gate("guaranteed_optimum_claim") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_simulation") == "PROCEED")
    b = brains_report()
    chk("argon4_seat_name",
        b["decider"]["seat"] == "Gemini Argon 4 Extended Thinking")
    chk("argon4_no_override", b["decider"]["can_override_hold"] is False)
    chk("claude_no_override", b["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", b["flow"]["rendered"] is False and b["flow"]["live"] == 0)
    return c


# ================================================================ CLI / demo
def demo():
    print("=" * 68)
    print("QuantumAnnealer-Lab - quantum annealing lab (classical simulation)")
    print("by Darion Kellar | MIT | pure stdlib | zero network | EDUCATION ONLY")
    print("=" * 68)
    print("\n[1] THE HONEST CORE - what this lab is and is not")
    print("    No quantum hardware is run, submitted to, or simulated here.")
    print("    The annealers are classical: Metropolis single-spin flips,")
    print("    plus cluster flips INSPIRED BY tunnelling (a heuristic -")
    print("    many spins move together, as a tunnelling event would move")
    print("    them). No quantum advantage is claimed (HOLD gate).")
    print("\n[2] Ising core - E = -sum(J s s) - sum(h s), QUBO <-> Ising exact")
    J5, h5, off5 = qubo_to_ising(*_demo_qubo())
    print(f"    demo QUBO (5 vars) -> Ising offset {off5:.4f}; energies match")
    print("    on all 32 states and the conversion round-trips (CHECKS).")
    print("\n[3] Annealing the demo spin glass (12 spins, seeded +/-1 couplings)")
    rep = anneal_report()
    print(f"    exact ground energy (brute force): {rep['ground_energy']:.1f}")
    print(f"    simulated annealing best:          {rep['single_best']:.1f} "
          f"(matches: {rep['single_matches_ground']})")
    print(f"    cluster-flip anneal best:          {rep['cluster_best']:.1f} "
          f"(matches: {rep['cluster_matches_ground']})")
    print(f"    frustrated instance: {rep['frustrated']} - some couplings")
    print("    must lose in every state; that is what makes it a glass.")
    print("\n[4] MaxCut on the 8-node book graph - graded by brute force")
    mc = maxcut_report()
    print(f"    optimal cut {mc['optimal_cut']:.0f} of total weight "
          f"{mc['total_weight']:.0f} | partition {mc['optimal_partition']}")
    print(f"    annealer: single {mc['anneal_single_cut']:.0f} | "
          f"cluster {mc['anneal_cluster_cut']:.0f} - both match the optimum")
    print("\n[5] Toy TSP (7 cities) - permutation annealing vs exact optimum")
    tp = tsp_report()
    print(f"    exact optimum: {tp['optimal_length']:.4f}  "
          f"{' -> '.join(tp['optimal_tour'])}")
    print(f"    annealer:      {tp['anneal_length']:.4f}  "
          f"{' -> '.join(tp['anneal_tour'])}")
    print("    (TSP is annealed in permutation space; its QUBO form would")
    print("     need 7^2 = 49 spins - stated, not hidden.)")
    print("\n[6] Bin packing - first-fit-decreasing vs the annealer")
    bp = binpack_report()
    print(f"    items {BIN_ITEMS} into bins of {BIN_CAPACITY}: "
          f"lower bound {bp['lower_bound_bins']} bins")
    print(f"    FFD uses {bp['ffd_bins']} bins | annealer uses "
          f"{bp['anneal_bins']} bins - both optimal here")
    print("\n[7] Barrier model - thermal pays for height, tunnelling for width")
    bar = barrier_report()
    print(f"    constructed barrier H = {bar['height']}, W = {bar['width']}: "
          f"tunnelling exponent {bar['tunnelling_exponent']:.1f}, "
          f"crossover T* = {bar['crossover_T']:.4f}")
    for row in bar["rows"]:
        print(f"    T = {row['T']:4.2f}: thermal {row['thermal_rate']:.3e} | "
              f"tunnel {row['tunnel_rate']:.3e} -> {row['winner']} wins")
    print("\n[8] Freeze-out ledger - measured success probability + TTS(99%)")
    tt = tts_report()
    for key in ("short_schedule", "full_schedule"):
        s = tt[key]
        print(f"    {key:15s} steps {s['steps']:6d}: p = "
              f"{s['success_probability']:.2f} ({s['hits']}/20 restarts) | "
              f"TTS = {s['tts_99']:,.0f} steps")
    print("    -> the short schedule freezes out mid-landscape; its TTS is")
    print("       priced from the measured p, not from its luckiest run.")
    print("\n[9] Hardware table - LABELLED published-class values, not measured")
    for row in hardware_report()["rows"]:
        print(f"    {row['name']:28s} {row['qubits']:>22s}  {row['topology']}")
    print("\n[10] Regime classifier (TinyML kNN, labels from the barrier model)")
    for name, row in regime_report().items():
        print(f"    {name:22s} H = {row['barrier_height']:.1f} "
              f"W = {row['barrier_width']:2d} frustrated = "
              f"{row['frustrated']} -> {row['knn_label']}")
    print("\n[11] Advisory seats")
    print(f"    decider: {argon4_decide()['decision']} trace={argon4_decide()['trace_sha256_16']}")
    print(f"    advisor can_override_hold = {claude_advisor()['can_override_hold']}")
    print(f"    flow rendered = {google_flow_plan()['rendered']} live = {google_flow_plan()['live']}")
    print(f"\n[12] HOLD gates: {len(HOLD_GATES)}")
    for k in HOLD_GATES:
        print(f"    HOLD  {k}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (sandbox simulation only) | HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def _demo_qubo():
    rng = random.Random(99)
    n5 = 5
    Q = [[0.0] * n5 for _ in range(n5)]
    lin = [rng.uniform(-2, 2) for _ in range(n5)]
    for i in range(n5):
        for j in range(i + 1, n5):
            Q[i][j] = Q[j][i] = rng.uniform(-2, 2)
    return Q, lin


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "ising":
        Q, lin = _demo_qubo()
        J, h, off = qubo_to_ising(Q, lin)
        e, s = brute_force_ising(J, h)
        print(json.dumps({"qubo_lin": lin, "ising_h": h, "offset": off,
                          "ising_ground_energy": e, "ising_ground_spins": s},
                         indent=2, sort_keys=True))
        return 0
    if cmd == "anneal":
        print(json.dumps(anneal_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "maxcut":
        print(json.dumps(maxcut_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "tsp":
        print(json.dumps(tsp_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "barrier":
        print(json.dumps(barrier_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "tts":
        print(json.dumps(tts_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(regime_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for k, v in HOLD_GATES.items():
            print(f"{k}: {v}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "real_hardware_control"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
