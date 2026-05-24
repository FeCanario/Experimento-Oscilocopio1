# app.py
"""
Experimento de Batimentos e Ressonância
Interface redesenhada — mais limpa e organizada.

Layout
──────
┌──────────────────────────────────────────────────────────────┐
│  TOPBAR   logo | DPO status | canal + intervalo | botões     │
│           ════════════════════════ (linha accent ciano)      │
├────────────────────────┬─────────────────────────────────────┤
│                        │  ┌─────────────────────────────┐   │
│  PAINEL DE MÉTRICAS    │  │   FORMA DE ONDA             │   │
│                        │  │   (verde · estilo scope)    │   │
│  T    f               │  └─────────────────────────────┘   │
│  f₁   f₂             │  ┌─────────────────────────────┐   │
│  f_bat  f_med         │  │   ESPECTRO FFT               │   │
│                        │  │   (ciano · picos marcados)  │   │
│  [equações]            │  └─────────────────────────────┘   │
├────────────────────────┴─────────────────────────────────────┤
│  LOG  — última mensagem                            [📋 hist] │
└──────────────────────────────────────────────────────────────┘
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

# ── Paleta ────────────────────────────────────────────────────────────────────
APP_BG      = "#0d1117"
PANEL_BG    = "#111820"
CARD_BG     = "#161c26"
CARD_HL     = "#1c2535"
BORDER      = "#2a3545"
ACCENT      = "#00b4d8"       # ciano accent
WAVE_C      = "#39d353"       # verde da onda (estilo GitHub contrib)
FFT_C       = "#58a6ff"       # azul FFT
PEAK1_C     = "#ff7b72"       # vermelho f₁
PEAK2_C     = "#ffa657"       # laranja f₂
GREEN_C     = "#3fb950"
RED_C       = "#f85149"
AMBER_C     = "#d29922"
TEXT_H      = "#e6edf3"       # texto principal
TEXT_S      = "#8b949e"       # texto secundário
TEXT_D      = "#3d4f65"       # texto muito apagado

SCOPE_BG    = "#060e0f"
SCOPE_GRID  = "#0f2020"

# ── Fontes ────────────────────────────────────────────────────────────────────
F_NUM   = ("Courier New", 26, "bold")   # número grande do card
F_UNIT  = ("Courier New", 11)           # unidade pequena do card
F_LABEL = ("Arial", 10)
F_SEC   = ("Arial", 11, "bold")
F_MONO  = ("Courier New", 11)
F_EQ    = ("Courier New", 10)


# ── Componentes reutilizáveis ─────────────────────────────────────────────────

class MetricCard(ctk.CTkFrame):
    """
    Card de métrica com:
      - Label descritivo (topo, cinza)
      - Número grande colorido (centro)
      - Equação de referência (base, muito apagada)
    """
    def __init__(self, parent, label: str, equation: str,
                 color: str = TEXT_H, **kw):
        super().__init__(parent,
                         fg_color=CARD_BG,
                         corner_radius=10,
                         border_width=1,
                         border_color=BORDER,
                         **kw)
        self._color = color

        ctk.CTkLabel(self, text=label,
                     font=F_LABEL, text_color=TEXT_S,
                     anchor="w").pack(fill="x", padx=12, pady=(10, 0))

        self._num = ctk.CTkLabel(self, text="—",
                                 font=F_NUM, text_color=color,
                                 anchor="w")
        self._num.pack(fill="x", padx=12, pady=(2, 0))

        ctk.CTkLabel(self, text=equation,
                     font=F_EQ, text_color=TEXT_D,
                     anchor="w").pack(fill="x", padx=12, pady=(0, 10))

    def set(self, text: str):
        self._num.configure(text=text)

    def pulse(self):
        """Pisca o card brevemente quando recebe novo valor."""
        self.configure(fg_color=CARD_HL)
        self.after(120, lambda: self.configure(fg_color=CARD_BG))


class StatusBadge(ctk.CTkFrame):
    """Badge de instrumento: LED colorido + nome."""
    def __init__(self, parent, label: str, **kw):
        super().__init__(parent, fg_color="#1a2030",
                         corner_radius=8, **kw)
        self._dot  = ctk.CTkLabel(self, text="⬤",
                                  text_color=RED_C,
                                  font=("Arial", 13), width=20)
        self._dot.pack(side="left", padx=(10, 4), pady=6)
        self._lbl  = ctk.CTkLabel(self,
                                  text=f"{label}: desconectado",
                                  font=F_LABEL, text_color=TEXT_S,
                                  anchor="w", width=210)
        self._lbl.pack(side="left", padx=(0, 10), pady=6)

    def ok(self, name: str):
        self._dot.configure(text_color=GREEN_C)
        self._lbl.configure(text=name[:32], text_color=TEXT_H)

    def err(self, label: str = "desconectado"):
        self._dot.configure(text_color=RED_C)
        self._lbl.configure(text=f"{label}", text_color=TEXT_S)

    def searching(self):
        self._dot.configure(text_color=AMBER_C)
        self._lbl.configure(text="procurando…", text_color=AMBER_C)


def _btn(parent, text, color, hover, text_color=APP_BG, **kw) -> ctk.CTkButton:
    return ctk.CTkButton(parent, text=text,
                         fg_color=color, hover_color=hover,
                         text_color=text_color,
                         corner_radius=8, font=F_SEC, **kw)


def _ghost_btn(parent, text, border_color, text_color, **kw) -> ctk.CTkButton:
    return ctk.CTkButton(parent, text=text,
                         fg_color="transparent",
                         hover_color="#1a2535",
                         border_width=1,
                         border_color=border_color,
                         text_color=text_color,
                         corner_radius=8, font=F_SEC, **kw)


# ── App principal ─────────────────────────────────────────────────────────────

class BatimentosApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=APP_BG)
        self.title("Batimentos & Ressonância")
        self.geometry("1360x860")
        self.minsize(1100, 700)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._conn          = ConnectionManager()
        self._worker: Optional[CaptureWorker] = None
        self._last_time:    Optional[np.ndarray] = None
        self._last_voltage: Optional[np.ndarray] = None
        self._last_metrics: dict = {}
        self._log_hist:     list[str] = []

        self._build_ui()
        self.after(400, self._scan_thread)

    # ─── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._build_topbar()
        self._build_body()
        self._build_logbar()

    # ─── Topbar ───────────────────────────────────────────────────────────────

    def _build_topbar(self):
        top = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=0, height=58)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(2, weight=1)
        top.grid_propagate(False)

        # Logo
        ctk.CTkLabel(top,
                     text="  ⚡  Batimentos & Ressonância",
                     font=("Arial", 16, "bold"),
                     text_color=ACCENT,
                     ).grid(row=0, column=0, padx=18, pady=12, sticky="w")

        # Status DPO
        self._badge = StatusBadge(top, "DPO")
        self._badge.grid(row=0, column=1, padx=12, pady=8)

        # Controles (canal + intervalo)
        ctrl = ctk.CTkFrame(top, fg_color="transparent")
        ctrl.grid(row=0, column=3, padx=12)

        ctk.CTkLabel(ctrl, text="Canal", font=F_LABEL,
                     text_color=TEXT_S).grid(row=0, column=0, padx=(0,4))
        self._ch_var = ctk.StringVar(value="CH1")
        ctk.CTkOptionMenu(ctrl, values=["CH1","CH2","CH3","CH4"],
                          variable=self._ch_var,
                          width=72, height=30,
                          fg_color="#1a2535",
                          button_color=BORDER,
                          dropdown_fg_color=CARD_BG,
                          ).grid(row=0, column=1, padx=4)

        ctk.CTkLabel(ctrl, text="Intervalo", font=F_LABEL,
                     text_color=TEXT_S).grid(row=0, column=2, padx=(12,4))
        self._interval = ctk.CTkEntry(ctrl, width=52, height=30,
                                      fg_color="#1a2535",
                                      border_color=BORDER,
                                      text_color=ACCENT,
                                      font=F_MONO)
        self._interval.insert(0, "0.5")
        self._interval.grid(row=0, column=3, padx=4)
        ctk.CTkLabel(ctrl, text="s", font=F_LABEL,
                     text_color=TEXT_S).grid(row=0, column=4)

        # Botões
        btns = ctk.CTkFrame(top, fg_color="transparent")
        btns.grid(row=0, column=4, padx=16)

        _ghost_btn(btns, "CONECTAR", ACCENT, ACCENT,
                   width=105, height=32,
                   command=self._scan_thread,
                   ).pack(side="left", padx=4)

        self._btn_cap = _btn(btns, "▶  CAPTURAR",
                             GREEN_C, "#2ea043", APP_BG,
                             width=130, height=32,
                             command=self._start_capture)
        self._btn_cap.pack(side="left", padx=4)

        self._btn_stop = _ghost_btn(btns, "■  PARAR",
                                    RED_C, RED_C,
                                    width=100, height=32,
                                    state="disabled",
                                    command=self._stop_capture)
        self._btn_stop.pack(side="left", padx=4)

        self._btn_csv = _ghost_btn(btns, "💾  CSV",
                                   BORDER, TEXT_S,
                                   width=80, height=32,
                                   state="disabled",
                                   command=self._save_csv)
        self._btn_csv.pack(side="left", padx=4)

        # Linha accent embaixo da topbar
        ctk.CTkFrame(self, fg_color=ACCENT, height=2,
                     corner_radius=0).grid(row=0, column=0,
                                           sticky="sew", padx=0)

    # ─── Corpo (métricas + gráficos) ──────────────────────────────────────────

    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0)   # painel métricas fixo
        body.grid_columnconfigure(1, weight=1)   # gráficos expandem

        self._build_metric_panel(body)
        self._build_plot_panel(body)

    # ── Painel de métricas (esquerda) ─────────────────────────────────────────

    def _build_metric_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=PANEL_BG,
                             corner_radius=0, width=280)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.grid_propagate(False)
        panel.grid_columnconfigure(0, weight=1)

        # Título do painel
        ctk.CTkLabel(panel, text="MEDIÇÕES",
                     font=("Arial", 11, "bold"),
                     text_color=ACCENT,
                     anchor="w").pack(fill="x", padx=16, pady=(16, 8))

        # Cards
        self._c_T   = MetricCard(panel, "Período",
                                 "T  =  1 / f",   WAVE_C)
        self._c_f   = MetricCard(panel, "Frequência",
                                 "f  =  1 / T",   WAVE_C)
        self._c_f1  = MetricCard(panel, "Pico  f₁  (FFT)",
                                 "maior pico do espectro", PEAK1_C)
        self._c_f2  = MetricCard(panel, "Pico  f₂  (FFT)",
                                 "segundo pico do espectro", PEAK2_C)

        for c in (self._c_T, self._c_f, self._c_f1, self._c_f2):
            c.pack(fill="x", padx=12, pady=5)

        # Separador
        ctk.CTkFrame(panel, fg_color=BORDER, height=1,
                     corner_radius=0).pack(fill="x", padx=12, pady=8)

        ctk.CTkLabel(panel, text="BATIMENTOS",
                     font=("Arial", 11, "bold"),
                     text_color=ACCENT,
                     anchor="w").pack(fill="x", padx=16, pady=(0, 8))

        self._c_bat = MetricCard(panel, "Frequência de Batimento",
                                 "f_bat  =  | f₁ − f₂ |",  "#c084fc")
        self._c_med = MetricCard(panel, "Frequência Média",
                                 "f_med  =  (f₁ + f₂) / 2", "#67e8f9")

        for c in (self._c_bat, self._c_med):
            c.pack(fill="x", padx=12, pady=5)

        # Rodapé ref
        ctk.CTkFrame(panel, fg_color=BORDER, height=1,
                     corner_radius=0).pack(fill="x", padx=12, pady=(12, 8))
        ctk.CTkLabel(panel,
                     text="Ref: RBEF / SciELO\nBatimentos e Ressonância",
                     font=("Arial", 9), text_color=TEXT_D,
                     justify="left", anchor="w",
                     ).pack(fill="x", padx=16, pady=(0, 12))

    # ── Painel de gráficos (direita) ──────────────────────────────────────────

    def _build_plot_panel(self, parent):
        outer = ctk.CTkFrame(parent, fg_color=SCOPE_BG, corner_radius=0)
        outer.grid(row=0, column=1, sticky="nsew")
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        self._fig = Figure(facecolor=SCOPE_BG)
        gs = GridSpec(2, 1, figure=self._fig,
                      height_ratios=[2, 1],
                      hspace=0.06,
                      left=0.07, right=0.98,
                      top=0.97, bottom=0.07)

        # ── Forma de onda ──────────────────────────────────────────────────
        self._ax_w = self._fig.add_subplot(gs[0])
        self._style_ax(self._ax_w)
        self._ax_w.set_ylabel("Tensão  (V)", color=TEXT_S, fontsize=9)
        self._ax_w.text(0.01, 0.96, "FORMA DE ONDA",
                        transform=self._ax_w.transAxes,
                        color=WAVE_C, fontsize=9, va="top",
                        fontfamily="monospace")
        self._ax_w.axhline(0, color=SCOPE_GRID, lw=0.8)
        self._ax_w.set_xticklabels([])

        self._ln_w, = self._ax_w.plot([], [], color=WAVE_C,
                                       lw=1.2, antialiased=True)

        # ── FFT ────────────────────────────────────────────────────────────
        self._ax_f = self._fig.add_subplot(gs[1])
        self._style_ax(self._ax_f)
        self._ax_f.set_xlabel("Frequência  (Hz)", color=TEXT_S, fontsize=9)
        self._ax_f.set_ylabel("Amplitude", color=TEXT_S, fontsize=9)
        self._ax_f.text(0.01, 0.96, "ESPECTRO  FFT",
                        transform=self._ax_f.transAxes,
                        color=FFT_C, fontsize=9, va="top",
                        fontfamily="monospace")

        self._ln_f,  = self._ax_f.plot([], [], color=FFT_C,
                                         lw=1.0, antialiased=True)
        self._vl_f1  = self._ax_f.axvline(np.nan, color=PEAK1_C,
                                           lw=1.5, ls="--", alpha=0.9)
        self._vl_f2  = self._ax_f.axvline(np.nan, color=PEAK2_C,
                                           lw=1.5, ls="--", alpha=0.9)
        self._ann_f1 = self._ax_f.annotate(
            "", xy=(0, 0.8), xycoords=("data", "axes fraction"),
            color=PEAK1_C, fontsize=8, fontfamily="monospace",
            ha="center")
        self._ann_f2 = self._ax_f.annotate(
            "", xy=(0, 0.6), xycoords=("data", "axes fraction"),
            color=PEAK2_C, fontsize=8, fontfamily="monospace",
            ha="center")

        self._canvas = FigureCanvasTkAgg(self._fig, master=outer)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._canvas.draw()

    def _style_ax(self, ax):
        ax.set_facecolor(SCOPE_BG)
        ax.tick_params(colors=TEXT_D, labelsize=8, length=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(SCOPE_GRID)
        ax.grid(True, color=SCOPE_GRID, lw=0.6, linestyle="-")

    # ─── Barra de log ─────────────────────────────────────────────────────────

    def _build_logbar(self):
        bar = ctk.CTkFrame(self, fg_color=PANEL_BG,
                           height=28, corner_radius=0)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)
        bar.grid_propagate(False)

        ctk.CTkLabel(bar, text="  LOG",
                     font=F_LABEL, text_color=TEXT_D,
                     width=45).grid(row=0, column=0, sticky="ns")

        ctk.CTkFrame(bar, fg_color=BORDER, width=1,
                     corner_radius=0).grid(row=0, column=0,
                                           sticky="nse", padx=(44,0))

        self._log_lbl = ctk.CTkLabel(bar, text="Iniciando…",
                                     font=F_MONO, text_color=TEXT_S,
                                     anchor="w")
        self._log_lbl.grid(row=0, column=1, sticky="ew", padx=12)

        hist_btn = ctk.CTkLabel(bar, text="📋 histórico",
                                font=F_LABEL, text_color=TEXT_D,
                                cursor="hand2")
        hist_btn.grid(row=0, column=2, padx=12)
        hist_btn.bind("<Button-1>", lambda _: self._show_log())
        self._log_lbl.bind("<Button-1>", lambda _: self._show_log())

    def _show_log(self):
        pop = ctk.CTkToplevel(self)
        pop.title("Histórico de Log")
        pop.geometry("720x380")
        pop.configure(fg_color=APP_BG)
        pop.lift(); pop.focus_force()
        txt = ctk.CTkTextbox(pop, fg_color=CARD_BG,
                             text_color=TEXT_H, font=F_MONO)
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", "\n".join(self._log_hist))
        txt.configure(state="disabled")
        txt.see("end")

    # ─── Conexão ──────────────────────────────────────────────────────────────

    def _scan_thread(self):
        self._badge.searching()
        self._log("Escaneando recursos VISA…")
        threading.Thread(target=self._do_scan, daemon=True).start()

    def _do_scan(self):
        s = self._conn.scan_and_connect()
        self.after(0, lambda: self._apply_scan(s))

    def _apply_scan(self, s: dict):
        if s["dpo_connected"]:
            self._badge.ok(f"DPO  ·  {s['dpo_name'][:26]}")
            self._log(f"DPO ✓  {s['dpo_name']}")
        else:
            self._badge.err("DPO: não encontrado")
            self._log("DPO ✗  verifique o cabo USB/GPIB.", warn=True)
        for e in s["errors"]:
            self._log(f"⚠  {e}", warn=True)

    # ─── Captura ──────────────────────────────────────────────────────────────

    def _start_capture(self):
        if not self._conn.ready:
            self._log("✗  Conecte o osciloscópio antes de capturar.", err=True)
            return
        try:
            interval = max(float(self._interval.get()), 0.1)
        except ValueError:
            interval = 0.5
        ch = int(self._ch_var.get().replace("CH", ""))

        self._btn_cap.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._btn_csv.configure(state="disabled")
        self._log(f"▶  Capturando CH{ch}  ·  intervalo {interval:.1f} s")

        self._worker = CaptureWorker(
            conn_manager=self._conn,
            channel=ch, interval=interval,
            on_capture=lambda t, v, m: self.after(0, lambda: self._gui_update(t, v, m)),
            on_error=lambda msg: self.after(0, lambda: self._log(f"✗  {msg}", err=True)),
        )
        self._worker.start()

    def _stop_capture(self):
        if self._worker and self._worker.is_alive():
            self._worker.stop()
        self._btn_cap.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        if self._last_voltage is not None:
            self._btn_csv.configure(state="normal")
        self._log("■  Captura encerrada.")

    # ─── Atualização da GUI ───────────────────────────────────────────────────

    def _gui_update(self, t: np.ndarray, v: np.ndarray, m: dict):
        self._last_time    = t
        self._last_voltage = v
        self._last_metrics = m

        # Cards
        self._c_T  .set(calc.fmt_time(m["T"]));   self._c_T.pulse()
        self._c_f  .set(calc.fmt_hz(m["f"]));     self._c_f.pulse()
        self._c_f1 .set(calc.fmt_hz(m["f1"]));    self._c_f1.pulse()
        self._c_f2 .set(calc.fmt_hz(m["f2"]));    self._c_f2.pulse()
        self._c_bat.set(calc.fmt_hz(m["f_bat"])); self._c_bat.pulse()
        self._c_med.set(calc.fmt_hz(m["f_med"])); self._c_med.pulse()

        # Forma de onda
        self._ln_w.set_data(t * 1e3, v)
        self._ax_w.relim(); self._ax_w.autoscale_view()
        self._ax_w.set_xlabel("Tempo  (ms)", color=TEXT_S, fontsize=9)

        # FFT
        freqs, amps = m["freqs"], m["amps"]
        mask = freqs <= 2000
        self._ln_f.set_data(freqs[mask], amps[mask])
        self._ax_f.set_xlim(0, 2000)
        self._ax_f.set_ylim(0, 1.1)

        # Linhas de pico + anotações
        if m["f1"] > 0:
            self._vl_f1.set_xdata([m["f1"], m["f1"]])
            self._ann_f1.set_text(f"f₁={calc.fmt_hz(m['f1'])}")
            self._ann_f1.xy = (m["f1"], 0.8)
        if m["f2"] > 0:
            self._vl_f2.set_xdata([m["f2"], m["f2"]])
            self._ann_f2.set_text(f"f₂={calc.fmt_hz(m['f2'])}")
            self._ann_f2.xy = (m["f2"], 0.6)

        self._canvas.draw_idle()

    # ─── Salvar CSV ───────────────────────────────────────────────────────────

    def _save_csv(self):
        if self._last_time is None:
            return
        m          = self._last_metrics
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_pretty  = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        ch         = self._ch_var.get()
        path       = Path(f"captura_{ts}.csv")

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
            w.writerow([f"# Período (T):           {calc.fmt_time(m.get('T',0))}"])
            w.writerow([f"# Frequência (f=1/T):    {calc.fmt_hz(m.get('f',0))}"])
            w.writerow([f"# Pico f1 (FFT):         {calc.fmt_hz(m.get('f1',0))}"])
            w.writerow([f"# Pico f2 (FFT):         {calc.fmt_hz(m.get('f2',0))}"])
            w.writerow([f"# Batimento |f1-f2|:     {calc.fmt_hz(m.get('f_bat',0))}"])
            w.writerow([f"# Freq. Média (f1+f2)/2: {calc.fmt_hz(m.get('f_med',0))}"])
            w.writerow([f"# Tensão máxima:         {m.get('v_max',0):.4f} V"])
            w.writerow([f"# Tensão mínima:         {m.get('v_min',0):.4f} V"])
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

        self._log(f"💾  Salvo: {path.resolve()}")

    # ─── Log ──────────────────────────────────────────────────────────────────

    def _log(self, msg: str, warn: bool = False, err: bool = False):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {msg}"
        self._log_hist.append(line)
        color = RED_C if err else (AMBER_C if warn else TEXT_S)
        self._log_lbl.configure(text=line[-120:], text_color=color)

    # ─── Fechar ───────────────────────────────────────────────────────────────

    def _on_close(self):
        if self._worker and self._worker.is_alive():
            self._worker.stop()
        self._conn.close_all()
        self.destroy()
