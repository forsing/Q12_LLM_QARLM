#!/usr/bin/env python3
"""
Q12 LLM — tehnika: Quantum Autoregressive Language Model (QARLM)
(čisto kvantno, bez klasičnog treninga i bez hibrida).

Autoregresivno generisanje sedmice — broj po broj (t = 1..7):
  - Kontekst C_t = već izabrani brojevi (x_1..x_{t-1}).
  - Uslovni freq_vector f(·|C_t) = suma-by-row-preklopa nad CELIM CSV-om:
        f[v] = Σ_{row ∈ CSV}  (|row ∩ C_t|)^α · 1[v ∈ row, v ∉ C_t]
     (za t = 1, koristi marginalni freq_vector CSV-a).
  - Amplitude-encoding konteksta: amp_ctx = amp_from_freq(f(·|C_t)) (dim 2^nq).
  - Uslovni decoder PQC (nq qubit-a, L slojeva):
        početno stanje = StatePreparation(amp_ctx),
        po sloju l = 0..L-1: Ry(π·fn[(k+l·nq) mod 39]) + ring-CNOT,
        gde je fn = f(·|C_t)/Σf.
  - Egzaktni Statevector → p = |ψ|² → bias_39 nad 1..39.
  - Maska izabranih (C_t) → argmax → x_t. Ponavlja se do t=7 → NEXT = sort(x_1..x_7).

Sve deterministički: seed=39; svi amp-ovi i parametri iz CELOG CSV-a.
Deterministička grid-optimizacija (nq, L) po meri cos(bias_39 na praznom kontekstu, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39
ALPHA = 1.0

GRID_NQ = (5, 6)
GRID_L = (2, 3, 4)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


# =========================
# Uslovni freq (autoregressive kontekst)
# =========================
def conditional_freq(H: np.ndarray, ctx: List[int], alpha: float = ALPHA) -> np.ndarray:
    if not ctx:
        return freq_vector(H)
    ctx_set = set(int(c) for c in ctx)
    f = np.zeros(N_MAX, dtype=np.float64)
    for row in H:
        overlap = 0
        for v in row:
            if int(v) in ctx_set:
                overlap += 1
        if overlap == 0:
            continue
        w = float(overlap) ** float(alpha)
        for v in row:
            iv = int(v)
            if iv in ctx_set:
                continue
            if 1 <= iv <= N_MAX:
                f[iv - 1] += w
    if float(f.sum()) < 1e-12:
        f = freq_vector(H)
        for c in ctx_set:
            if 1 <= int(c) <= N_MAX:
                f[int(c) - 1] = 0.0
    return f


# =========================
# Amplitude-encoding (dim = 2^nq) iz freq-vektora
# =========================
def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# Uslovni decoder PQC
# =========================
def qarlm_dist(H: np.ndarray, ctx: List[int], nq: int, L: int, alpha: float = ALPHA) -> np.ndarray:
    """Vraća bias_39 raspodelu nad 1..39 iz egzaktnog Statevector-a."""
    f_cond = conditional_freq(H, ctx, alpha)
    s = float(f_cond.sum())
    fn = f_cond / s if s > 0 else np.ones(N_MAX, dtype=np.float64) / N_MAX

    amp_ctx = amp_from_freq(f_cond, nq)
    qc = QuantumCircuit(nq, name="qarlm")
    qc.append(StatePreparation(amp_ctx.tolist()), range(nq))
    for l in range(L):
        for k in range(nq):
            qc.ry(float(np.pi * fn[(k + l * nq) % N_MAX]), k)
        for k in range(nq - 1):
            qc.cx(k, k + 1)
        if nq > 1:
            qc.cx(nq - 1, 0)

    sv = Statevector(qc)
    p = np.abs(sv.data) ** 2
    s_p = float(p.sum())
    p = p / s_p if s_p > 0 else p
    return bias_39(p)


# =========================
# Autoregresivno generisanje NEXT
# =========================
def generate_next(H: np.ndarray, nq: int, L: int, alpha: float = ALPHA) -> Tuple[int, ...]:
    chosen: List[int] = []
    for _ in range(N_NUMBERS):
        b = qarlm_dist(H, chosen, nq, L, alpha).copy()
        for c in chosen:
            if 1 <= c <= N_MAX:
                b[c - 1] = -1.0
        idx = int(np.argmax(b))
        chosen.append(idx + 1)
    return tuple(sorted(chosen))


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =========================
# Determ. grid-optimizacija (nq, L) po meri cos(bias@∅, freq_csv)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        for L in GRID_L:
            try:
                b0 = qarlm_dist(H, [], nq, L, ALPHA)
                score = cosine(b0, f_csv_n)
            except Exception:
                continue
            key = (score, -nq, -L)
            if best is None or key > best[0]:
                best = (key, dict(nq=nq, L=L, score=float(score)))
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q12 LLM (QARLM — Quantum Autoregressive LM): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED, "| alpha:", ALPHA)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| L (slojeva):", best["L"],
        "| cos(bias@∅, freq_csv):", round(float(best["score"]), 6),
    )

    pred = generate_next(H, best["nq"], best["L"], ALPHA)
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q12 LLM (QARLM — Quantum Autoregressive LM): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39 | alpha: 1.0
BEST hparam: nq= 5 | L (slojeva): 2 | cos(bias@∅, freq_csv): 0.872039
predikcija NEXT: (7, 8, 13, 14, 19, 22, 27)
"""



"""
Q12_LLM_QARLM.py — tehnika: Quantum Autoregressive Language Model

Autoregresivno generisanje:
  t=1..7 — za svaki korak se iz CELOG CSV-a računa uslovni freq_vector
  f(·|C_t) (row-weighting po preklopu sa kontekstom, alpha = 1.0),
  amplitude-encodira u kvantno stanje, pa se primenjuje decoder PQC
  (Ry iz fn + ring-CNOT, L slojeva).
Egzaktni Statevector → bias_39 → mask(C_t) → argmax → x_t.
NEXT = sort(x_1..x_7).

Tehnike:
Amplitude encoding kondicionalne distribucije (StatePreparation).
Decoder PQC čiji su parametri deterministički iz uslovnih frekvencija.
Autoregressive inference — generiše NEXT broj-po-broj (za razliku od Q8–Q11
koji uzimaju TOP-7 iz jedne marginale).

Prednosti:
Čisto kvantno: nema klasičnog softmax-a, klasičnog treninga, gradient descent-a.
Uvodi zavisnost na prethodno izabrane brojeve kroz uslovni freq (decoder kontekst).
Mehanizam maske i argmax eliminiše duplikate u sedmici.

Nedostaci:
Uslovni freq je deterministička heuristika (row-overlap weighting), nije kvantna
distribucija nad sekvencama.
Budžet: nq ≤ 6, L ≤ 4 za brzu grid-pretragu; generisanje radi 7 PQC evaluacija.
Mera cos(bias@∅, freq_csv) ocenjuje samo marginalu (t=1), ne celu autoregressive
trajektoriju.
mod-39 readout meša stanja (dim 2^nq ≠ 39).
"""
