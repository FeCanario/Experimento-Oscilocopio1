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

    # ─── Corpo (abas) ─────────────────────────────────────────────────────────

    def _build_body(self):
        self._tabs = ctk.CTkTabview(
            self,
            fg_color=APP_BG,
            segmented_button_fg_color=PANEL_BG,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color="#0090b0",
            segmented_button_unselected_color=PANEL_BG,
            segmented_button_unselected_hover_color=CARD_BG,
            text_color=TEXT_H,
            corner_radius=0,
        )
        self._tabs.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self._tabs.grid_columnconfigure(0, weight=1)
        self._tabs.grid_rowconfigure(0, weight=1)

        self._tabs.add("  Medição  ")
        self._tabs.add("  Configurar Scope  ")

        # Aba 1 — Medição
        tab1 = self._tabs.tab("  Medição  ")
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=0)
        tab1.grid_columnconfigure(1, weight=1)
        self._build_metric_panel(tab1)
        self._build_plot_panel(tab1)

        # Aba 2 — Configuração do scope
        tab2 = self._tabs.tab("  Configurar Scope  ")
        tab2.grid_rowconfigure(0, weight=1)
        tab2.grid_columnconfigure(0, weight=1)
        self._build_scope_config(tab2)

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

    # ─── Aba 2: Configurar Scope (DPO 2012B) ─────────────────────────────────

    def _build_scope_config(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color=APP_BG, corner_radius=0)
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure((0, 1, 2), weight=1, uniform="col")

        def section(text, row, col, colspan=1):
            f = ctk.CTkFrame(scroll, fg_color=PANEL_BG,
                             corner_radius=10, border_width=1,
                             border_color=BORDER)
            f.grid(row=row, column=col, columnspan=colspan,
                   sticky="nsew", padx=8, pady=8)
            ctk.CTkLabel(f, text=text, font=("Arial", 12, "bold"),
                         text_color=ACCENT, anchor="w"
                         ).pack(fill="x", padx=14, pady=(12, 6))
            ctk.CTkFrame(f, fg_color=BORDER, height=1,
                         corner_radius=0).pack(fill="x", padx=14, pady=(0, 10))
            return f

        def row_field(parent, label, widget_factory):
            r = ctk.CTkFrame(parent, fg_color="transparent")
            r.pack(fill="x", padx=14, pady=3)
            ctk.CTkLabel(r, text=label, font=F_LABEL,
                         text_color=TEXT_S, width=150, anchor="w"
                         ).pack(side="left")
            w = widget_factory(r)
            w.pack(side="left", padx=(8, 0))
            return w

        def entry(parent, default="", width=100):
            e = ctk.CTkEntry(parent, width=width, height=28,
                             fg_color=CARD_BG, border_color=BORDER,
                             text_color=ACCENT, font=F_MONO)
            e.insert(0, default)
            return e

        def optmenu(parent, values, width=130):
            return ctk.CTkOptionMenu(parent, values=values, width=width,
                                     height=28, fg_color=CARD_BG,
                                     button_color=BORDER,
                                     dropdown_fg_color=CARD_BG,
                                     text_color=ACCENT, font=F_MONO)

        # ── Cabeçalho do modelo ────────────────────────────────────────────
        hdr = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=10,
                           border_width=1, border_color=BORDER)
        hdr.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(10, 0))
        ctk.CTkLabel(hdr,
                     text="Tektronix DPO 2012B  —  Digital Phosphor Oscilloscope  "
                          "·  100 MHz  ·  1 GS/s  ·  2 canais",
                     font=("Courier New", 12, "bold"), text_color=TEXT_H,
                     ).pack(padx=16, pady=10)

        # ── Seção Horizontal (Timebase) ────────────────────────────────────
        s_tb = section("⏱  Horizontal  (Timebase)", row=1, col=0)

        TB_SCALES = ["1ns", "2ns", "5ns", "10ns", "20ns", "50ns",
                     "100ns", "200ns", "500ns",
                     "1µs", "2µs", "5µs", "10µs", "20µs", "50µs",
                     "100µs", "200µs", "500µs",
                     "1ms", "2ms", "5ms", "10ms", "20ms", "50ms",
                     "100ms", "200ms", "500ms",
                     "1s", "2s", "5s", "10s", "25s", "50s"]
        TB_MAP = {
            "1ns":1e-9,"2ns":2e-9,"5ns":5e-9,"10ns":10e-9,"20ns":20e-9,
            "50ns":50e-9,"100ns":100e-9,"200ns":200e-9,"500ns":500e-9,
            "1µs":1e-6,"2µs":2e-6,"5µs":5e-6,"10µs":10e-6,"20µs":20e-6,
            "50µs":50e-6,"100µs":100e-6,"200µs":200e-6,"500µs":500e-6,
            "1ms":1e-3,"2ms":2e-3,"5ms":5e-3,"10ms":10e-3,"20ms":20e-3,
            "50ms":50e-3,"100ms":100e-3,"200ms":200e-3,"500ms":500e-3,
            "1s":1.0,"2s":2.0,"5s":5.0,"10s":10.0,"25s":25.0,"50s":50.0,
        }
        self._tb_map = TB_MAP

        self._tb_scale = row_field(s_tb, "Escala  (s/div)",
                                   lambda p: optmenu(p, TB_SCALES, 130))
        self._tb_scale.set("1ms")

        self._rec_len = row_field(s_tb, "Record Length",
                                  lambda p: optmenu(p, ["1000","10000",
                                                        "100000","1000000"], 130))
        self._rec_len.set("10000")

        # ── Seção Canal 1 ──────────────────────────────────────────────────
        s_ch1 = section("📡  Canal 1  (CH1)", row=1, col=1)
        self._ch1_scale, self._ch1_coup, self._ch1_bw = \
            self._build_channel_section(s_ch1, row_field, entry, optmenu)

        # ── Seção Canal 2 ──────────────────────────────────────────────────
        s_ch2 = section("📡  Canal 2  (CH2)", row=1, col=2)
        self._ch2_scale, self._ch2_coup, self._ch2_bw = \
            self._build_channel_section(s_ch2, row_field, entry, optmenu)

        # ── Seção Trigger ──────────────────────────────────────────────────
        s_tr = section("⚡  Trigger", row=2, col=0)

        self._trig_src = row_field(s_tr, "Fonte",
                                   lambda p: optmenu(p, ["CH1","CH2","EXT"], 100))
        self._trig_src.set("CH1")

        self._trig_slope = row_field(s_tr, "Borda",
                                     lambda p: optmenu(p, ["RISE","FALL"], 100))
        self._trig_slope.set("RISE")

        self._trig_level = row_field(s_tr, "Nível  (V)",
                                     lambda p: entry(p, "0.0", 80))

        self._trig_mode = row_field(s_tr, "Modo",
                                    lambda p: optmenu(p, ["AUTO","NORMal"], 100))
        self._trig_mode.set("AUTO")

        # ── Seção Aquisição ────────────────────────────────────────────────
        s_aq = section("🔬  Aquisição", row=2, col=1)

        self._acq_mode = row_field(s_aq, "Modo",
                                   lambda p: optmenu(p,
                                       ["SAMple","PEAKdetect","AVErage","HIRes"], 130))
        self._acq_mode.set("SAMple")

        self._acq_avg = row_field(s_aq, "Nº Médias (Average)",
                                  lambda p: optmenu(p,
                                      ["2","4","8","16","32","64","128","256","512"], 80))
        self._acq_avg.set("16")

        # ── Botões de ação ─────────────────────────────────────────────────
        s_ac = section("🎛  Ações", row=2, col=2)

        _btn(s_ac, "✅  Aplicar Configurações",
             ACCENT, "#0090b0", APP_BG,
             width=220, height=36,
             command=self._apply_scope_config,
             ).pack(padx=14, pady=6, anchor="w")

        _btn(s_ac, "📥  Ler do Scope",
             "#2d6a4f", "#1b4332", TEXT_H,
             width=220, height=36,
             command=self._read_scope_config,
             ).pack(padx=14, pady=6, anchor="w")

        _ghost_btn(s_ac, "🔄  AutoSet",
                   BORDER, TEXT_S,
                   width=220, height=36,
                   command=self._do_autoset,
                   ).pack(padx=14, pady=6, anchor="w")

        # Status da última operação
        ctk.CTkFrame(s_ac, fg_color=BORDER, height=1,
                     corner_radius=0).pack(fill="x", padx=14, pady=(10, 6))
        self._cfg_status = ctk.CTkLabel(s_ac, text="Aguardando…",
                                         font=F_MONO, text_color=TEXT_D,
                                         anchor="w")
        self._cfg_status.pack(fill="x", padx=14, pady=(0, 12))

    def _build_channel_section(self, frame, row_field, entry, optmenu):
        """Campos comuns de configuração de canal (escala, coupling, BW)."""
        V_SCALES = ["1mV","2mV","5mV","10mV","20mV","50mV",
                    "100mV","200mV","500mV",
                    "1V","2V","5V","10V"]
        V_MAP = {
            "1mV":1e-3,"2mV":2e-3,"5mV":5e-3,"10mV":10e-3,"20mV":20e-3,
            "50mV":50e-3,"100mV":100e-3,"200mV":200e-3,"500mV":500e-3,
            "1V":1.0,"2V":2.0,"5V":5.0,"10V":10.0,
        }
        if not hasattr(self, "_vscale_map"):
            self._vscale_map = V_MAP

        scale = row_field(frame, "Escala  (V/div)",
                          lambda p: optmenu(p, V_SCALES, 110))
        scale.set("1V")

        coup = row_field(frame, "Acoplamento",
                         lambda p: optmenu(p, ["DC","AC","GND"], 100))
        coup.set("DC")

        bw = row_field(frame, "Largura de Banda",
                       lambda p: optmenu(p, ["FULl (100MHz)","TWEnty (20MHz)"], 160))
        bw.set("FULl (100MHz)")

        return scale, coup, bw

    def _apply_scope_config(self):
        """Envia todas as configurações para o DPO 2012B."""
        if not self._conn.ready:
            self._cfg_status.configure(
                text="✗  Scope não conectado.", text_color=RED_C)
            return

        dpo = self._conn.dpo
        try:
            # Timebase
            tb_str = self._tb_scale.get()
            tb_val = self._tb_map.get(tb_str, 1e-3)
            dpo.set_timebase_scale(tb_val)
            dpo.set_record_length(int(self._rec_len.get()))

            # CH1
            ch1_v = self._vscale_map.get(self._ch1_scale.get(), 1.0)
            dpo.set_ch_scale(1, ch1_v)
            dpo.set_ch_coupling(1, self._ch1_coup.get())
            dpo.set_ch_bandwidth(1, self._ch1_bw.get().split()[0])

            # CH2
            ch2_v = self._vscale_map.get(self._ch2_scale.get(), 1.0)
            dpo.set_ch_scale(2, ch2_v)
            dpo.set_ch_coupling(2, self._ch2_coup.get())
            dpo.set_ch_bandwidth(2, self._ch2_bw.get().split()[0])

            # Trigger
            dpo.set_trigger_source(self._trig_src.get())
            dpo.set_trigger_slope(self._trig_slope.get())
            dpo.set_trigger_level(float(self._trig_level.get() or 0))
            dpo.set_trigger_mode(self._trig_mode.get())

            # Aquisição
            dpo.set_acquire_mode(self._acq_mode.get())
            if self._acq_mode.get() == "AVErage":
                dpo.set_acquire_numavg(int(self._acq_avg.get()))

            self._cfg_status.configure(
                text=f"✅  Configurações aplicadas  [{datetime.now().strftime('%H:%M:%S')}]",
                text_color=GREEN_C)
            self._log("✅  Configurações do scope aplicadas com sucesso.")

        except Exception as exc:
            self._cfg_status.configure(
                text=f"✗  Erro: {exc}", text_color=RED_C)
            self._log(f"✗  Erro ao configurar scope: {exc}", err=True)

    def _read_scope_config(self):
        """Lê as configurações atuais do scope e preenche os campos."""
        if not self._conn.ready:
            self._cfg_status.configure(
                text="✗  Scope não conectado.", text_color=RED_C)
            return

        try:
            s = self._conn.dpo.read_all_settings()

            # Timebase — encontra a chave mais próxima no mapa
            tb_closest = min(self._tb_map, key=lambda k: abs(self._tb_map[k] - s["tb_scale"]))
            self._tb_scale.set(tb_closest)
            self._rec_len.set(str(s["rec_length"]))

            # CH1
            v1_closest = min(self._vscale_map, key=lambda k: abs(self._vscale_map[k] - s["ch1_scale"]))
            self._ch1_scale.set(v1_closest)
            self._ch1_coup.set(s["ch1_coup"][:2])

            # CH2
            v2_closest = min(self._vscale_map, key=lambda k: abs(self._vscale_map[k] - s["ch2_scale"]))
            self._ch2_scale.set(v2_closest)
            self._ch2_coup.set(s["ch2_coup"][:2])

            # Trigger
            self._trig_src.set(s["trig_src"])
            self._trig_level.delete(0, "end")
            self._trig_level.insert(0, f"{s['trig_lvl']:.3f}")
            self._trig_mode.set(s["trig_mode"][:4])

            # Aquisição
            self._acq_mode.set(s["acq_mode"][:3])

            self._cfg_status.configure(
                text=f"📥  Lido do scope  [{datetime.now().strftime('%H:%M:%S')}]",
                text_color=ACCENT)
            self._log("📥  Configurações lidas do scope.")

        except Exception as exc:
            self._cfg_status.configure(
                text=f"✗  Erro: {exc}", text_color=RED_C)
            self._log(f"✗  Erro ao ler scope: {exc}", err=True)

    def _do_autoset(self):
        """Executa AutoSet no scope."""
        if not self._conn.ready:
            self._cfg_status.configure(
                text="✗  Scope não conectado.", text_color=RED_C)
            return
        self._cfg_status.configure(text="🔄  Executando AutoSet…", text_color=AMBER_C)
        self._log("🔄  AutoSet executando (aguarde ~3s)…")

        def _run():
            try:
                self._conn.dpo.autoset()
                self.after(0, lambda: self._cfg_status.configure(
                    text="✅  AutoSet concluído.", text_color=GREEN_C))
                self.after(0, lambda: self._log("✅  AutoSet concluído."))
            except Exception as exc:
                self.after(0, lambda: self._cfg_status.configure(
                    text=f"✗  {exc}", text_color=RED_C))

        threading.Thread(target=_run, daemon=True).start()

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
