# calculations.py
"""
Cálculos do experimento de Batimentos e Ressonância.

Baseado no artigo:
    "Batimentos e Ressonância" — RBEF / SciELO
    https://www.scielo.br/j/rbef/a/D7k5Pxj7HcmmbpGZJMf4wNs/

Equações implementadas
──────────────────────
  (1)  f = 1 / T                     frequência fundamental
  (2)  f_bat = |f₁ − f₂|            frequência de batimento (artigo Eq. 3)
  (3)  f_med = (f₁ + f₂) / 2        frequência média / portadora
  (4)  FFT                           confirmação espectral (artigo usa FFT
                                     para validar os valores medidos)
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Período e frequência  (artigo Eq. 1)
# ─────────────────────────────────────────────────────────────────────────────

def period_from_zero_crossings(time: np.ndarray, voltage: np.ndarray) -> float:
    """
    Estima o período T medindo o tempo médio entre cruzamentos
    ascendentes pelo zero.

    Equivale à leitura de T diretamente na tela do osciloscópio.

    Retorna T em segundos (0.0 se não encontrar cruzamentos suficientes).
    """
    # Detecta cruzamentos ascendentes (sinal vai de negativo para positivo)
    signs      = np.sign(voltage)
    crossings  = np.where((signs[:-1] < 0) & (signs[1:] >= 0))[0]

    if len(crossings) < 2:
        return 0.0

    # Interpola linearmente o instante exato de cada cruzamento
    times_cross = []
    for i in crossings:
        v0, v1 = voltage[i], voltage[i + 1]
        t0, t1 = time[i],    time[i + 1]
        t_cross = t0 - v0 * (t1 - t0) / (v1 - v0)
        times_cross.append(t_cross)

    # Período médio entre cruzamentos consecutivos
    periods = np.diff(times_cross)
    return float(np.mean(periods))


def frequency_from_period(T: float) -> float:
    """
    f = 1 / T   (artigo Eq. 1)
    Retorna 0.0 se T for zero ou inválido.
    """
    if T > 0:
        return 1.0 / T
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FFT  (artigo usa FFT para confirmar valores)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fft(
    time: np.ndarray,
    voltage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula o espectro de frequência (FFT) da forma de onda.

    Retorna
    -------
    freqs : np.ndarray — frequências em Hz (apenas lado positivo)
    amps  : np.ndarray — amplitudes normalizadas (0 a 1)
    """
    n    = len(voltage)
    dt   = float(np.mean(np.diff(time))) if n > 1 else 1.0

    # FFT e frequências
    fft_vals = np.fft.rfft(voltage * np.hanning(n))
    freqs    = np.fft.rfftfreq(n, d=dt)
    amps     = np.abs(fft_vals)

    # Normaliza para 0–1
    max_amp = amps.max()
    if max_amp > 0:
        amps = amps / max_amp

    return freqs, amps


def find_peaks_fft(
    freqs: np.ndarray,
    amps:  np.ndarray,
    n_peaks: int = 2,
    min_freq: float = 50.0,
) -> list[float]:
    """
    Encontra os `n_peaks` picos dominantes no espectro FFT
    acima de `min_freq` Hz.

    Retorna lista de frequências dos picos em Hz (ordenados por amplitude).
    """
    # Ignora componente DC e frequências muito baixas
    mask   = freqs >= min_freq
    f_filt = freqs[mask]
    a_filt = amps[mask]

    if len(f_filt) == 0:
        return []

    # Detecta picos locais simples (maior que os vizinhos)
    peak_idx = []
    for i in range(1, len(a_filt) - 1):
        if a_filt[i] > a_filt[i - 1] and a_filt[i] > a_filt[i + 1]:
            peak_idx.append(i)

    if not peak_idx:
        # Fallback: pega os maiores valores
        peak_idx = np.argsort(a_filt)[::-1][:n_peaks].tolist()

    # Ordena por amplitude decrescente e pega os n_peaks maiores
    peak_idx_sorted = sorted(peak_idx, key=lambda i: a_filt[i], reverse=True)
    top_idx         = peak_idx_sorted[:n_peaks]

    # Retorna frequências ordenadas por valor (menor primeiro)
    peak_freqs = sorted([float(f_filt[i]) for i in top_idx])
    return peak_freqs


# ─────────────────────────────────────────────────────────────────────────────
# Batimentos  (artigo Eq. 3)
# ─────────────────────────────────────────────────────────────────────────────

def beat_frequency(f1: float, f2: float) -> float:
    """
    f_bat = |f₁ − f₂|

    Frequência de batimento — taxa em que a amplitude oscila
    quando dois sons de frequências próximas se somam.
    """
    return abs(f1 - f2)


def mean_frequency(f1: float, f2: float) -> float:
    """
    f_med = (f₁ + f₂) / 2

    Frequência média — corresponde à portadora audível
    quando dois sons se somam (artigo Eq. 3).
    """
    return (f1 + f2) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Análise completa de uma captura
# ─────────────────────────────────────────────────────────────────────────────

def analyze_waveform(
    time: np.ndarray,
    voltage: np.ndarray,
) -> dict:
    """
    Recebe a forma de onda (tempo, tensão) e retorna todas as
    grandezas calculadas pelo artigo.

    Retorna dict com:
        T       — período (s)
        f       — frequência fundamental (Hz)
        f1, f2  — dois picos FFT dominantes (Hz)
        f_bat   — frequência de batimento = |f1 − f2| (Hz)
        f_med   — frequência média = (f1 + f2)/2 (Hz)
        freqs   — vetor de frequências FFT (Hz)
        amps    — vetor de amplitudes FFT normalizadas
        v_max   — tensão de pico positivo (V)
        v_min   — tensão de pico negativo (V)
    """
    # Período e frequência via cruzamentos de zero
    T = period_from_zero_crossings(time, voltage)
    f = frequency_from_period(T)

    # FFT
    freqs_fft, amps_fft = compute_fft(time, voltage)

    # Dois picos dominantes
    peaks = find_peaks_fft(freqs_fft, amps_fft, n_peaks=2, min_freq=50.0)

    f1 = peaks[0] if len(peaks) > 0 else 0.0
    f2 = peaks[1] if len(peaks) > 1 else 0.0

    f_bat = beat_frequency(f1, f2)
    f_med = mean_frequency(f1, f2)

    return {
        "T":     T,
        "f":     f,
        "f1":    f1,
        "f2":    f2,
        "f_bat": f_bat,
        "f_med": f_med,
        "freqs": freqs_fft,
        "amps":  amps_fft,
        "v_max": float(np.max(voltage)),
        "v_min": float(np.min(voltage)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatação
# ─────────────────────────────────────────────────────────────────────────────

def fmt_hz(hz: float) -> str:
    """Formata frequência com unidade automática."""
    if hz <= 0:
        return "—"
    if hz >= 1e3:
        return f"{hz/1e3:.3f} kHz"
    return f"{hz:.2f} Hz"


def fmt_time(s: float) -> str:
    """Formata tempo com unidade automática."""
    if s <= 0:
        return "—"
    if s < 1e-3:
        return f"{s*1e6:.2f} µs"
    if s < 1.0:
        return f"{s*1e3:.3f} ms"
    return f"{s:.4f} s"
