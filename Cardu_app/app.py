# app.py
"""
Analisador de Ondas Sonoras — Layout Sidebar (tipo IDE/DevTools)

Layout
──────
┌──────────────────────────────────────────────────────────┐
│  ⚙ Analisador de Ondas      [DPO: desconectado]         │ <- header
├────────────────┬──────────────────────────────────────────┤
│                │                                          │
│   SIDEBAR      │  PLOTS                                  │
│   Esquerdo     │  ┌─────────────────────────────────────┐│
│   (280px)      │  │ FORMA DE ONDA                        ││
│                │  │                                     ││
│  [Botões]      │  │                                     ││
│  [Ch/Int]      │  ├─────────────────────────────────────┤│
│  [Métricas]    │  │ ESPECTRO FFT                        ││
│  [Log]         │  │  f1: ...Hz  f2: ...Hz              ││
│                │  │  bat: ...Hz  med: ...Hz            ││
│                │  │                                     ││
└────────────────┴──────────────────────────────────────────┘
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import customtkinter as ctk
import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

import calculations as calc
from instruments import ConnectionManager
from worker import CaptureWorker

matplotlib.use("TkAgg")

# ── Paleta: azul escuro + gráficos claros + botões coloridos ─────────────────
APP_BG    = "#1a1a2e"
PANEL_BG  = "#16213e"
CARD_BG   = "#0f3460"
CARD_HL   = "#16355c"
BORDER    = "#2a4158"
ACCENT    = "#00d4ff"
WAVE_C    = "#ff6b6b"
FFT_C     = "#4ecdc4"
PEAK1_C   = "#95e1d3"
PEAK2_C   = "#f38181"
GREEN_C   = "#2ecc71"
RED_C     = "#e74c3c"
AMBER_C   = "#f39c12"
TEXT_H    = "#ecf0f1"
TEXT_S    = "#95a5a6"
TEXT_D    = "#7f8c8d"

PLOT_BG   = "#f8f9fa"
PLOT_GRID = "#e9ecef"
BTN_CONNECT  = "#3498db"
BTN_PLAY     = "#2ecc71"
BTN_STOP     = "#e74c3c"
BTN_EXPORT   = "#9b59b6"

# ── Fontes ────────────────────────────────────────────────────────────────────
F_TITLE  = ("Segoe UI", 15, "bold")
F_NUM    = ("Segoe UI", 18, "bold")
F_LABEL  = ("Segoe UI", 9)
F_SEC    = ("Segoe UI", 10, "bold")
F_MONO   = ("Consolas",  10)
F_EQ     = ("Consolas",   8)


class MetricValue(ctk.CTkFrame):
    """Pequeno componente de métrica estilo label-value."""
    def __init__(self, parent, label: str, color: str = TEXT_H, **kw):
        super().__init__(parent, fg_color="transparent", **kw)

        ctk.CTkLabel(self, text=label,
                     font=F_LABEL, text_color=TEXT_S,
                     anchor="w").pack(fill="x", pady=(0, 2))

        self._val = ctk.CTkLabel(self, text="—",
                                  font=("Segoe UI", 16, "bold"),
                                  text_color=color, anchor="w")
        self._val.pack(fill="x")

    def set(self, text: str):
        self._val.configure(text=text)

    def pulse(self):
        self._val.configure(text_color=ACCENT)
        self.after(200, lambda: self._val.configure(text_color=self._val._text_color if hasattr(self._val, '_text_color') else TEXT_H))


class BatimentosApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=APP_BG)
        self.title("Analisador de Ondas Sonoras")
        self.minsize(1000, 650)
        self.after(0, lambda: self.state("zoomed"))

        self._conn          = ConnectionManager()
        self._worker: Optional[CaptureWorker] = None
        self._last_time:    Optional[np.ndarray] = None
        self._last_voltage: Optional[np.ndarray] = None
        self._last_metrics: dict = {}
        self._log_hist:     list[str] = []

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self.after(400, self._scan_thread)

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0)    # sidebar fixo
        self.grid_columnconfigure(1, weight=1)    # plots expande

        self._build_header()
        self._build_sidebar()
        self._build_plots_area()

    # ─── Header minimal ───────────────────────────────────────────────────────

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=0, height=56)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(header, text="⚙  Analisador de Ondas Sonoras",
                     font=F_TITLE, text_color=ACCENT).grid(row=0, column=0,
                                                            padx=18, pady=12, sticky="w")

        self._status_lbl = ctk.CTkLabel(header, text="● Osciloscópio  ·  desconectado",
                                         font=F_LABEL, text_color=RED_C)
        self._status_lbl.grid(row=0, column=2, padx=18, pady=12, sticky="e")

    # ─── Sidebar esquerdo (tipo IDE) ──────────────────────────────────────────

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=0, width=280)
        sidebar.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        sidebar.grid_propagate(False)

        # Usando pack interno para organização vertical
        scroll = ctk.CTkScrollableFrame(sidebar, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=8, pady=8)

        # ── Seção: CAPTURA ────────────────────────────────────────────────────
        ctk.CTkLabel(scroll, text="CAPTURA", font=F_SEC,
                     text_color=ACCENT).pack(fill="x", pady=(6, 4))

        self._btn_conn = ctk.CTkButton(scroll, text="Conectar",
                                        fg_color=BTN_CONNECT, hover_color="#2980b9",
                                        text_color=TEXT_H, corner_radius=6,
                                        font=F_SEC, height=36,
                                        command=self._scan_thread)
        self._btn_conn.pack(fill="x", pady=4)

        self._btn_cap = ctk.CTkButton(scroll, text="▶  Iniciar Gravação",
                                       fg_color=BTN_PLAY, hover_color="#27ae60",
                                       text_color=TEXT_H, corner_radius=6,
                                       font=F_SEC, height=36,
                                       command=self._start_capture)
        self._btn_cap.pack(fill="x", pady=4)

        self._btn_stop = ctk.CTkButton(scroll, text="■  Parar Gravação",
                                        fg_color=BTN_STOP, hover_color="#c0392b",
                                        text_color=TEXT_H, corner_radius=6,
                                        font=F_SEC, height=36,
                                        state="disabled",
                                        command=self._stop_capture)
        self._btn_stop.pack(fill="x", pady=4)

        # ── Seção: CONFIGURAÇÃO ────────────────────────────────────────────────
        ctk.CTkLabel(scroll, text="CONFIGURAÇÃO", font=F_SEC,
                     text_color=ACCENT).pack(fill="x", pady=(12, 4))

        ch_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        ch_frame.pack(fill="x", pady=4)
        ctk.CTkLabel(ch_frame, text="Canal", font=F_LABEL,
                     text_color=TEXT_S).pack(side="left")
        self._ch_var = ctk.StringVar(value="CH1")
        ctk.CTkOptionMenu(ch_frame, values=["CH1", "CH2", "CH3", "CH4"],
                          variable=self._ch_var,
                          width=150, height=28,
                          fg_color=CARD_BG, button_color=BORDER,
                          dropdown_fg_color=CARD_BG, font=F_MONO
                          ).pack(side="right", padx=(4, 0))

        int_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        int_frame.pack(fill="x", pady=4)
        ctk.CTkLabel(int_frame, text="Intervalo (s)", font=F_LABEL,
                     text_color=TEXT_S).pack(side="left")
        self._interval = ctk.CTkEntry(int_frame, width=80, height=28,
                                       fg_color=CARD_BG, border_color=BORDER,
                                       text_color=ACCENT, font=F_MONO)
        self._interval.insert(0, "0.5")
        self._interval.pack(side="right", padx=(4, 0))

        # ── Seção: DADOS ───────────────────────────────────────────────────────
        ctk.CTkLabel(scroll, text="DADOS", font=F_SEC,
                     text_color=ACCENT).pack(fill="x", pady=(12, 4))

        self._btn_csv = ctk.CTkButton(scroll, text="💾  Exportar CSV",
                                       fg_color=BTN_EXPORT, hover_color="#8e44ad",
                                       text_color=TEXT_H, corner_radius=6,
                                       font=F_SEC, height=32,
                                       state="disabled",
                                       command=self._save_csv)
        self._btn_csv.pack(fill="x", pady=4)

        # ── Seção: MÉTRICAS ────────────────────────────────────────────────────
        ctk.CTkLabel(scroll, text="MEDIÇÕES", font=F_SEC,
                     text_color=ACCENT).pack(fill="x", pady=(12, 4))

        self._c_T   = MetricValue(scroll, "Período (T)", WAVE_C)
        self._c_T.pack(fill="x", pady=4)

        self._c_f   = MetricValue(scroll, "Frequência (f)", WAVE_C)
        self._c_f.pack(fill="x", pady=4)

        self._c_f1  = MetricValue(scroll, "Pico f₁", PEAK1_C)
        self._c_f1.pack(fill="x", pady=4)

        self._c_f2  = MetricValue(scroll, "Pico f₂", PEAK2_C)
        self._c_f2.pack(fill="x", pady=4)

        self._c_bat = MetricValue(scroll, "Batimento (f_bat)", ACCENT)
        self._c_bat.pack(fill="x", pady=4)

        self._c_med = MetricValue(scroll, "Freq. Média (f_med)", FFT_C)
        self._c_med.pack(fill="x", pady=4)

        # ── Seção: LOG ─────────────────────────────────────────────────────────
        ctk.CTkLabel(scroll, text="LOG", font=F_SEC,
                     text_color=ACCENT).pack(fill="x", pady=(12, 4))

        self._log_box = ctk.CTkTextbox(scroll, height=120,
                                        fg_color=CARD_BG, text_color=TEXT_S,
                                        font=F_MONO, corner_radius=6,
                                        wrap="word", state="disabled")
        self._log_box.pack(fill="both", expand=True, pady=4)

    # ─── Área de Plots (direita) ──────────────────────────────────────────────

    def _build_plots_area(self):
        area = ctk.CTkFrame(self, fg_color=APP_BG, corner_radius=0)
        area.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)
        area.grid_rowconfigure(0, weight=3)   # waveform 60%
        area.grid_rowconfigure(1, weight=2)   # fft 40%
        area.grid_columnconfigure(0, weight=1)

        # ── Plots com matplotlib ───────────────────────────────────────────
        self._fig = Figure(facecolor=PLOT_BG, figsize=(9, 6), dpi=90)
        gs = GridSpec(2, 1, figure=self._fig, height_ratios=[3, 2],
                      hspace=0.15, left=0.08, right=0.98,
                      top=0.97, bottom=0.07)

        self._ax_w = self._fig.add_subplot(gs[0])
        self._style_ax(self._ax_w)
        self._ax_w.set_ylabel("Tensão (V)", color=TEXT_D, fontsize=9)
        self._ax_w.text(0.01, 0.95, "FORMA DE ONDA", transform=self._ax_w.transAxes,
                        color=WAVE_C, fontsize=10, fontweight="bold",
                        fontfamily="monospace", va="top")
        self._ax_w.axhline(0, color=PLOT_GRID, lw=0.8)
        self._ax_w.set_xticklabels([])
        self._ln_w, = self._ax_w.plot([], [], color=WAVE_C, lw=1.3, antialiased=True)

        self._ax_f = self._fig.add_subplot(gs[1])
        self._style_ax(self._ax_f)
        self._ax_f.set_xlabel("Frequência (Hz)", color=TEXT_D, fontsize=9)
        self._ax_f.set_ylabel("Amplitude", color=TEXT_D, fontsize=9)
        self._ax_f.text(0.01, 0.95, "ESPECTRO FFT", transform=self._ax_f.transAxes,
                        color=FFT_C, fontsize=10, fontweight="bold",
                        fontfamily="monospace", va="top")
        self._ln_f,  = self._ax_f.plot([], [], color=FFT_C, lw=1.0, antialiased=True)
        self._vl_f1  = self._ax_f.axvline(np.nan, color=PEAK1_C, lw=1.5, ls="--", alpha=0.9)
        self._vl_f2  = self._ax_f.axvline(np.nan, color=PEAK2_C, lw=1.5, ls="--", alpha=0.9)
        self._ann_f1 = self._ax_f.annotate("", xy=(0, 0.85), xycoords=("data", "axes fraction"),
                                            color=PEAK1_C, fontsize=8, ha="center",
                                            fontfamily="monospace")
        self._ann_f2 = self._ax_f.annotate("", xy=(0, 0.65), xycoords=("data", "axes fraction"),
                                            color=PEAK2_C, fontsize=8, ha="center",
                                            fontfamily="monospace")

        self._canvas = FigureCanvasTkAgg(self._fig, master=area)
        self._canvas.get_tk_widget().grid(row=0, column=0, rowspan=2, sticky="nsew")
        self._canvas.draw()

        # ── Info rápido overlay (direita dos plots) ────────────────────────
        info = ctk.CTkFrame(area, fg_color=CARD_BG, corner_radius=10,
                             border_width=1, border_color=BORDER)
        info.grid(row=0, column=1, sticky="nsew", padx=(8, 0), rowspan=2)
        info.grid_propagate(False)

        ctk.CTkLabel(info, text="RESUMO", font=F_SEC,
                     text_color=ACCENT).pack(fill="x", padx=10, pady=(10, 6))

        self._info_text = ctk.CTkTextbox(info, fg_color=PANEL_BG,
                                          text_color=TEXT_S, font=F_MONO,
                                          corner_radius=6, wrap="word",
                                          state="disabled")
        self._info_text.pack(fill="both", expand=True, padx=8, pady=8)

    def _style_ax(self, ax):
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=TEXT_D, labelsize=8, length=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(PLOT_GRID)
        ax.grid(True, color=PLOT_GRID, lw=0.6, linestyle="-")

    # ─── Conexão ──────────────────────────────────────────────────────────────

    def _scan_thread(self):
        self._status_lbl.configure(text="◌ Osciloscópio  ·  procurando…", text_color=AMBER_C)
        self._log("Escaneando recursos VISA…")
        threading.Thread(target=self._do_scan, daemon=True).start()

    def _do_scan(self):
        s = self._conn.scan_and_connect()
        self.after(0, lambda: self._apply_scan(s))

    def _apply_scan(self, s: dict):
        if s["dpo_connected"]:
            self._status_lbl.configure(text=f"● DPO  ·  {s['dpo_name'][:35]}", text_color=GREEN_C)
            self._log(f"DPO conectado: {s['dpo_name']}")
        else:
            self._status_lbl.configure(text="● Osciloscópio  ·  não encontrado", text_color=RED_C)
            self._log("DPO não encontrado — verifique o cabo.", warn=True)
        for e in s["errors"]:
            self._log(f"⚠  {e}", warn=True)

    # ─── Captura ──────────────────────────────────────────────────────────────

    def _start_capture(self):
        if not self._conn.ready:
            self._log("Conecte o osciloscópio antes de capturar.", err=True)
            return
        try:
            interval = max(float(self._interval.get()), 0.1)
        except ValueError:
            interval = 0.5
        ch = int(self._ch_var.get().replace("CH", ""))

        self._btn_cap.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._btn_csv.configure(state="disabled")
        self._log(f"Iniciando captura  CH{ch}  ·  {interval:.1f} s/ciclo")

        self._worker = CaptureWorker(
            conn_manager=self._conn,
            channel=ch, interval=interval,
            on_capture=lambda t, v, m: self.after(0, lambda: self._gui_update(t, v, m)),
            on_error=lambda msg: self.after(0, lambda: self._log(f"Erro: {msg}", err=True)),
        )
        self._worker.start()

    def _stop_capture(self):
        if self._worker and self._worker.is_alive():
            self._worker.stop()
        self._btn_cap.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        if self._last_voltage is not None:
            self._btn_csv.configure(state="normal")
        self._log("Captura encerrada.")

    # ─── Atualização da GUI ───────────────────────────────────────────────────

    def _gui_update(self, t: np.ndarray, v: np.ndarray, m: dict):
        self._last_time    = t
        self._last_voltage = v
        self._last_metrics = m

        # Atualiza métricas no sidebar
        self._c_T  .set(calc.fmt_time(m["T"]));   self._c_T.pulse()
        self._c_f  .set(calc.fmt_hz(m["f"]));     self._c_f.pulse()
        self._c_f1 .set(calc.fmt_hz(m["f1"]));    self._c_f1.pulse()
        self._c_f2 .set(calc.fmt_hz(m["f2"]));    self._c_f2.pulse()
        self._c_bat.set(calc.fmt_hz(m["f_bat"])); self._c_bat.pulse()
        self._c_med.set(calc.fmt_hz(m["f_med"])); self._c_med.pulse()

        # Info rápido no painel direito
        info = f"""T = {calc.fmt_time(m["T"])}
f = {calc.fmt_hz(m["f"])}

f₁ = {calc.fmt_hz(m["f1"])}
f₂ = {calc.fmt_hz(m["f2"])}

f_bat = {calc.fmt_hz(m["f_bat"])}
f_med = {calc.fmt_hz(m["f_med"])}

Vpp = {m.get("v_max", 0) - m.get("v_min", 0):.4f} V"""

        self._info_text.configure(state="normal")
        self._info_text.delete("1.0", "end")
        self._info_text.insert("1.0", info)
        self._info_text.configure(state="disabled")

        # Plots
        self._ln_w.set_data(t * 1e3, v)
        self._ax_w.relim(); self._ax_w.autoscale_view()
        self._ax_w.set_xlabel("Tempo (ms)", color=TEXT_D, fontsize=9)

        freqs, amps = m["freqs"], m["amps"]
        mask = freqs <= 2000
        self._ln_f.set_data(freqs[mask], amps[mask])
        self._ax_f.set_xlim(0, 2000)
        self._ax_f.set_ylim(0, 1.1)

        if m["f1"] > 0:
            self._vl_f1.set_xdata([m["f1"], m["f1"]])
            self._ann_f1.set_text(f"f₁={calc.fmt_hz(m['f1'])}")
            self._ann_f1.xy = (m["f1"], 0.85)
        if m["f2"] > 0:
            self._vl_f2.set_xdata([m["f2"], m["f2"]])
            self._ann_f2.set_text(f"f₂={calc.fmt_hz(m['f2'])}")
            self._ann_f2.xy = (m["f2"], 0.65)

        self._canvas.draw_idle()

    # ─── Salvar CSV ───────────────────────────────────────────────────────────

    def _save_csv(self):
        if self._last_time is None:
            return
        m         = self._last_metrics
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_pretty = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        ch        = self._ch_var.get()
        path      = Path(f"captura_{ts}.csv")

        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["# ============================================================"])
            w.writerow(["# EXPERIMENTO: BATIMENTOS E RESSONÂNCIA"])
            w.writerow(["# ============================================================"])
            w.writerow([f"# Data/Hora:             {ts_pretty}"])
            w.writerow([f"# Canal:                 {ch}"])
            w.writerow([f"# Pontos capturados:     {len(self._last_time)}"])
            w.writerow(["#"])
            w.writerow(["# MÉTRICAS CALCULADAS"])
            w.writerow([f"# Período (T):           {calc.fmt_time(m.get('T', 0))}"])
            w.writerow([f"# Frequência (f=1/T):    {calc.fmt_hz(m.get('f', 0))}"])
            w.writerow([f"# Pico f1 (FFT):         {calc.fmt_hz(m.get('f1', 0))}"])
            w.writerow([f"# Pico f2 (FFT):         {calc.fmt_hz(m.get('f2', 0))}"])
            w.writerow([f"# Batimento |f1-f2|:     {calc.fmt_hz(m.get('f_bat', 0))}"])
            w.writerow([f"# Freq. Média (f1+f2)/2: {calc.fmt_hz(m.get('f_med', 0))}"])
            w.writerow([f"# Tensão máxima:         {m.get('v_max', 0):.4f} V"])
            w.writerow([f"# Tensão mínima:         {m.get('v_min', 0):.4f} V"])
            w.writerow(["#"])
            w.writerow(["# Ref: Batimentos e Ressonância — RBEF/SciELO"])
            w.writerow(["# https://www.scielo.br/j/rbef/a/D7k5Pxj7HcmmbpGZJMf4wNs/"])
            w.writerow(["# ============================================================"])
            w.writerow([])
            w.writerow(["# SEÇÃO 1 — FORMA DE ONDA"])
            w.writerow(["tempo_s", "tensao_v"])
            for t, v in zip(self._last_time, self._last_voltage):
                w.writerow([f"{t:.9f}", f"{v:.6f}"])
            w.writerow([])
            w.writerow(["# SEÇÃO 2 — ESPECTRO FFT"])
            w.writerow(["frequencia_hz", "amplitude_norm"])
            for f, a in zip(m.get("freqs", []), m.get("amps", [])):
                if f <= 5000:
                    w.writerow([f"{f:.4f}", f"{a:.6f}"])

        self._log(f"CSV salvo: {path.resolve()}")

    # ─── Log ──────────────────────────────────────────────────────────────────

    def _log(self, msg: str, warn: bool = False, err: bool = False):
        ts     = datetime.now().strftime("%H:%M:%S")
        prefix = "✗ " if err else ("⚠ " if warn else "  ")
        line   = f"[{ts}] {prefix}{msg}"
        self._log_hist.append(line)

        self._log_box.configure(state="normal")
        self._log_box.insert("end", line + "\n")
        self._log_box.configure(state="disabled")
        self._log_box.see("end")

    # ─── Fechar ───────────────────────────────────────────────────────────────

    def _on_close(self):
        if self._worker and self._worker.is_alive():
            self._worker.stop()
        self._conn.close_all()
        self.destroy()
