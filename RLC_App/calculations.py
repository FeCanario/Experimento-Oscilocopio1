# calculations.py
"""
Cálculos físicos do circuito RLC série.

Baseado na analogia elétrica do oscilador harmônico forçado amortecido
descrita no artigo:

    "Batimentos e Ressonância" — RBEF / SciELO
    https://www.scielo.br/j/rbef/a/D7k5Pxj7HcmmbpGZJMf4wNs/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ANALOGIA MECÂNICA ↔ ELÉTRICA  (artigo, Eq. 4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Grandeza mecânica            Grandeza elétrica
  ─────────────────────────    ─────────────────────────
  Massa           m        ↔   Indutância        L  [H]
  Amortecimento   b        ↔   Resistência        R  [Ω]
  Rigidez         k        ↔   1/Capacitância  1/C  [1/F]
  Força           F₀       ↔   Tensão de entrada  V_in  [V]
  Deslocamento    x        ↔   Carga             q  [C]

  Equação diferencial mecânica (artigo, Eq. 4):
      m·ẍ + b·ẋ + k·x = F₀·cos(ωt)

  Equivalente elétrico (RLC série):
      L·q̈ + R·q̇ + (1/C)·q = V_in·cos(ωt)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AMPLITUDE DE RESSONÂNCIA  (artigo, Eq. 6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mecânica:
      x₀(ω) = F₀ / √[ (k − mω²)² + (bω)² ]

  Elétrica — tensão sobre R (resposta passa-faixa):
      V_out(ω) = V_in · R / √[ R² + (ωL − 1/ωC)² ]

  Ganho (função de transferência em módulo):
      |H(jω)| = V_out / V_in = R / √[ R² + (ωL − 1/ωC)² ]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FREQUÊNCIA NATURAL / RESSONÂNCIA  (artigo, Eq. 7)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mecânica:    ω₀ = √(k/m)
  Elétrica:    ω₀ = 1/√(LC)   →   f₀ = 1 / (2π√(LC))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FATOR DE QUALIDADE E LARGURA DE BANDA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q  = (1/R) · √(L/C)   =  ω₀·L / R   =  1/(ω₀·R·C)
  BW = f₀ / Q           =  R / (2π·L)          [Hz]

  Frequências de meia-potência (-3 dB):
      α   = R / (2L)
      f₁  = (−α + √(α² + ω₀²)) / (2π)
      f₂  = ( α + √(α² + ω₀²)) / (2π)
      BW  = f₂ − f₁

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BATIMENTOS  (artigo, Eq. 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  f_bat = |f₁ − f₂|   (frequência de batimento)
  f_med = (f₁ + f₂)/2 (frequência média / portadora)
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import curve_fit

    _SCIPY = True
except ImportError:
    _SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Função de transferência
# ─────────────────────────────────────────────────────────────────────────────

def transfer_function(
    f: np.ndarray,
    R: float,
    L: float,
    C: float,
) -> np.ndarray:
    """
    Módulo da função de transferência  |H(jω)| = V_out / V_in
    medida sobre o resistor R do circuito RLC série.

    Derivada diretamente da Eq. 6 do artigo (amplitude do oscilador forçado):

        |H(jω)| = R / √[ R² + (ωL − 1/ωC)² ]

    Parâmetros
    ----------
    f   : frequências em Hz
    R   : resistência em Ω
    L   : indutância em H
    C   : capacitância em F
    """
    f = np.asarray(f, dtype=float)
    w = 2.0 * np.pi * f
    Xl = w * L
    Xc = 1.0 / (w * C + 1e-30)          # evita divisão por zero
    Z  = np.sqrt(R**2 + (Xl - Xc)**2)
    return R / Z


# ─────────────────────────────────────────────────────────────────────────────
# Modelo passa-faixa normalizado (para curve fitting)
# ─────────────────────────────────────────────────────────────────────────────

def _bandpass_model(
    f: np.ndarray,
    A: float,
    f0: float,
    Q: float,
) -> np.ndarray:
    """
    Forma normalizada do oscilador forçado amortecido (artigo, Eq. 6)
    reescrita em termos de f₀ e Q:

        H(f) = A · (f/f₀)/Q / √[ (1 − (f/f₀)²)² + ((f/f₀)/Q)² ]

    Em f = f₀  →  H ≈ A  (ganho de pico).

    Usada internamente no ajuste Levenberg–Marquardt.
    """
    x = np.asarray(f, dtype=float) / f0
    num = x / Q
    den = np.sqrt((1.0 - x**2) ** 2 + (x / Q) ** 2)
    den = np.where(den == 0.0, np.finfo(float).eps, den)
    return A * num / den


# ─────────────────────────────────────────────────────────────────────────────
# Métricas teóricas (a partir dos valores dos componentes)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_from_components(R: float, L: float, C: float) -> dict:
    """
    Calcula as grandezas características do circuito RLC série
    a partir dos valores conhecidos dos componentes.

    Equações (analogia com o artigo):

        f₀ = 1 / (2π√(LC))     ← artigo Eq. 7: ω₀ = √(k/m)
        Q  = (1/R)·√(L/C)
        BW = f₀ / Q  =  R/(2πL)
        f₁, f₂  (frequências de meia-potência, solução exata)

    Retorna
    -------
    dict com chaves: f0, Q, BW, f1, f2  (todos em Hz, adimensional para Q)
    """
    if R <= 0 or L <= 0 or C <= 0:
        return {"f0": 0.0, "Q": 0.0, "BW": 0.0, "f1": 0.0, "f2": 0.0}

    w0 = 1.0 / np.sqrt(L * C)
    f0 = w0 / (2.0 * np.pi)
    Q  = (1.0 / R) * np.sqrt(L / C)
    BW = f0 / Q

    alpha   = R / (2.0 * L)
    w0_sq   = 1.0 / (L * C)
    w1      = -alpha + np.sqrt(alpha**2 + w0_sq)
    w2      =  alpha + np.sqrt(alpha**2 + w0_sq)
    f1      = w1 / (2.0 * np.pi)
    f2      = w2 / (2.0 * np.pi)

    return {"f0": float(f0), "Q": float(Q), "BW": float(BW),
            "f1": float(f1), "f2": float(f2)}


# ─────────────────────────────────────────────────────────────────────────────
# Ajuste de curva (Levenberg–Marquardt)
# ─────────────────────────────────────────────────────────────────────────────

def _initial_guess(
    freqs: np.ndarray,
    gains: np.ndarray,
) -> tuple[float, float, float]:
    """Chutes iniciais (A₀, f₀₀, Q₀) a partir dos dados medidos."""
    idx  = int(np.nanargmax(gains))
    A0   = float(gains[idx])
    f0_0 = float(freqs[idx])

    target = A0 / np.sqrt(2.0)
    mask   = gains >= target
    if mask.sum() >= 2:
        f_bw = freqs[mask]
        BW0  = max(float(f_bw[-1]) - float(f_bw[0]), 1e-9)
        Q0   = float(np.clip(f0_0 / BW0, 0.1, 1000.0))
    else:
        Q0 = 1.0

    return A0, f0_0, Q0


def fit_experimental_curve(
    freqs: np.ndarray,
    gains: np.ndarray,
    n_smooth: int = 600,
) -> dict | None:
    """
    Ajuste Levenberg–Marquardt do modelo passa-faixa (Eq. 6 do artigo)
    aos dados experimentais (ganho V_out/V_in vs. frequência).

    Parâmetros
    ----------
    freqs    : frequências em Hz (array 1-D)
    gains    : ganho medido V_out/V_in (adimensional ou em Vpp se V_in fixo)
    n_smooth : pontos da curva ajustada suavizada

    Retorna
    -------
    dict com:
        A, f0, Q, BW, f1, f2,
        freq_smooth (Hz), gain_smooth (mesmas unidades de gains)
    None se dados insuficientes.
    """
    freqs = np.asarray(freqs, dtype=float)
    gains = np.asarray(gains, dtype=float)

    mask = np.isfinite(freqs) & np.isfinite(gains) & (freqs > 0) & (gains >= 0)
    if mask.sum() < 5:
        return None

    f, g = freqs[mask], gains[mask]
    A0, f0_0, Q0 = _initial_guess(f, g)

    if _SCIPY:
        try:
            popt, _ = curve_fit(
                _bandpass_model,
                f, g,
                p0=[A0, f0_0, Q0],
                bounds=([0.0, f.min() * 0.5, 0.01], [np.inf, f.max() * 2.0, 2000.0]),
                maxfev=20_000,
            )
            A, f0, Q = float(popt[0]), float(popt[1]), float(popt[2])
        except Exception:
            A, f0, Q = A0, f0_0, Q0
    else:
        A, f0, Q = A0, f0_0, Q0

    BW = f0 / Q if Q > 0 else 0.0

    # Frequências de meia-potência via fórmula exata
    # (equivalente ao resultado do artigo para o oscilador amortecido)
    half_bw = BW / 2.0
    f1 = max(f0 - half_bw, 0.0)
    f2 = f0 + half_bw

    # Curva suavizada
    f_smooth = np.logspace(np.log10(f.min()), np.log10(f.max()), n_smooth)
    g_smooth = _bandpass_model(f_smooth, A, f0, Q)

    return {
        "A":           A,
        "f0":          f0,
        "Q":           Q,
        "BW":          BW,
        "f1":          f1,
        "f2":          f2,
        "freq_smooth": f_smooth,
        "gain_smooth": g_smooth,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batimentos (artigo, Eq. 3)
# ─────────────────────────────────────────────────────────────────────────────

def beat_frequency(f1: float, f2: float) -> tuple[float, float]:
    """
    Calcula frequência de batimento e frequência média.

    Artigo, Eq. 3:
        f_bat = |f₁ − f₂|
        f_med = (f₁ + f₂) / 2

    Retorna (f_bat, f_med) em Hz.
    """
    return abs(f1 - f2), (f1 + f2) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de formatação
# ─────────────────────────────────────────────────────────────────────────────

def fmt_hz(hz: float) -> str:
    """Formata frequência com unidade automática (Hz / kHz / MHz)."""
    if hz >= 1e6:
        return f"{hz / 1e6:.4f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:.4f} kHz"
    return f"{hz:.4f} Hz"


def fmt_metric(value: float, unit: str = "", decimals: int = 4) -> str:
    """Formata grandeza genérica."""
    return f"{value:.{decimals}f} {unit}".strip()
