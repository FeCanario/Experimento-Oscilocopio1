# app.py
"""
Interface do Experimento de Batimentos e Ressonância.

Layout
──────
┌──────────────────────────────────────────────────────────┐
│  TOPBAR — status DPO | canal | intervalo | botões        │
├──────────────────────────────────────────────────────────┤
│  DADOS:  T= │ f= │ f₁= │ f₂= │ f_bat= │ f_med=          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   FORMA DE ONDA  (verde sobre preto, igual ao scope)    │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   ESPECTRO FFT   (picos f₁ e f₂ marcados)               │
│                                                          │
└──────────────────────────────────────────────────────────┘
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
SCOPE_BG   = "#0a0f00"       # fundo do osciloscópio (preto-esverdeado)
SCOPE_GRID = "#1a2a0a"       # grade
WAVE_COLOR = "#00ff41"       # verde fosforescente da onda
FFT_COLOR  = "#00ccff"       # azul ciano do espectro
PEAK_COLOR = "#ff4444"       # vermelho dos picos marcados
PEAK2_COLOR= "#ffaa00"       # laranja do segundo pico

APP_BG     = "#0d1117"       # fundo geral
CARD_BG    = "#161b22"       # fundo dos cards de dados
BORDER     = "#30363d"       # bordas
TEXT_DIM   = "#8b949e"       # texto secundário
TEXT_MAIN  = "#e6edf3"       # texto principal
CYAN       = "#58a6ff"       # azul destaque
GREEN      = "#3fb950"       # verde OK
RED        = "#f85149"       # vermelho erro
AMBER      = "#d29922"       # amarelo aviso

FONT_DATA  = ("Courier New", 20, "bold")
FONT_LABEL = ("Arial", 10)
FONT_SEC   = ("Arial", 11, "bold")
FONT_MONO  = ("Courier New", 11)


# ── Card de dado numérico ─────────────────────────────────────────────────────

class DataCard(ctk.CTkFrame):
    """Card compacto: label em cima, valor grande embaixo."""

    def __init__(self, parent, label: str, color: str = TEXT_MAIN, **kw):
        super().__init__(parent, fg_color=CARD_BG, corner_radius=6,
                         border_width=1, border_color=BORDER, **kw)
        ctk.CTkLabel(self, text=label, font=FONT_LABEL,
                     text_color=TEXT_DIM).pack(pady=(6, 0), padx=10)
        self._val = ctk.CTkLabel(self, text="—", font=FONT_DATA,
                                 text_color=color)
        self._val.pack(pady=(0, 6), padx=10)

    def update(self, text: str):
        self._val.configure(text=text)


# ── App principal ─────────────────────────────────────────────────────────────

class BatimentosApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=APP_BG)
        self.title("Experimento — Batimentos e Ressonância")
        self.geometry("1280x860")
        self.minsize(1000, 700)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._conn:   ConnectionManager         = ConnectionManager()
        self._worker: Optional[CaptureWorker]   = None
        self._last_time:    Optional[np.ndarray] = None
        self._last_voltage: Optional[np.ndarray] = None
        self._last_metrics: dict = {}
        self._log_history: list[str] = []

        self._build_ui()
        self.after(400, self._scan_thread)

    # ─── Construção da UI ─────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_topbar()
        self._build_data_bar()
        self._build_plots()
        self._build_logbar()

    # ── Topbar ────────────────────────────────────────────────────────────────

    def _build_topbar(self):
        top = ctk.CTkFrame(self, fg_color=CARD_BG, height=52,
                           corner_radius=0, border_width=0)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(2, weight=1)

        # Logo
        ctk.CTkLabel(top, text="  ⚡ BATIMENTOS & RESSONÂNCIA",
                     font=("Arial", 15, "bold"),
                     text_color=CYAN).grid(row=0, column=0, padx=16, pady=10)

        # Status DPO
        self._dpo_frame = ctk.CTkFrame(top, fg_color="#1c2128",
                                       corner_radius=6, width=260, height=32)
        self._dpo_frame.grid(row=0, column=1, padx=10)
        self._dpo_frame.grid_propagate(False)
        self._dpo_dot  = ctk.CTkLabel(self._dpo_frame, text="●",
                                      text_color=RED, font=("Arial", 14), width=20)
        self._dpo_dot.pack(side="left", padx=(8, 2))
        self._dpo_name = ctk.CTkLabel(self._dpo_frame,
                                      text="DPO: Desconectado",
                                      font=FONT_LABEL, text_color=TEXT_DIM)
        self._dpo_name.pack(side="left", padx=2)

        # Controles direita
        ctrl = ctk.CTkFrame(top, fg_color="transparent")
        ctrl.grid(row=0, column=3, padx=10)

        # Canal
        ctk.CTkLabel(ctrl, text="Canal:", font=FONT_LABEL,
                     text_color=TEXT_DIM).pack(side="left", padx=(0, 4))
        self._ch_var = ctk.StringVar(value="CH1")
        ctk.CTkOptionMenu(ctrl, values=["CH1", "CH2", "CH3", "CH4"],
                          variable=self._ch_var, width=70, height=28,
                          fg_color="#1c2128", button_color="#30363d",
                          ).pack(side="left", padx=4)

        # Intervalo
        ctk.CTkLabel(ctrl, text="Intervalo:", font=FONT_LABEL,
                     text_color=TEXT_DIM).pack(side="left", padx=(8, 4))
        self._interval = ctk.CTkEntry(ctrl, width=55, height=28,
                                      fg_color="#1c2128", border_color=BORDER,
                                      text_color=CYAN, font=FONT_MONO)
        self._interval.pack(side="left", padx=4)
        self._interval.insert(0, "0.5")
        ctk.CTkLabel(ctrl, text="s", font=FONT_LABEL,
                     text_color=TEXT_DIM).pack(side="left")

        # Botão conectar
        ctk.CTkButton(ctrl, text="CONECTAR", width=100, height=32,
                      fg_color="#1c2128", border_color=CYAN, border_width=1,
                      text_color=CYAN, hover_color="#1c2128",
                      font=FONT_SEC,
                      command=self._scan_thread,
                      ).pack(side="left", padx=8)

        # Botão capturar
        self._btn_cap = ctk.CTkButton(
            ctrl, text="▶  CAPTURAR", width=120, height=32,
            fg_color="#0d2818", border_color=GREEN, border_width=1,
            text_color=GREEN, hover_color="#0d2818",
            font=FONT_SEC, command=self._start_capture,
        )
        self._btn_cap.pack(side="left", padx=4)

        # Botão parar
        self._btn_stop = ctk.CTkButton(
            ctrl, text="■  PARAR", width=90, height=32,
            fg_color="#2d0d0d", border_color=RED, border_width=1,
            text_color=RED, hover_color="#2d0d0d",
            font=FONT_SEC, state="disabled",
            command=self._stop_capture,
        )
        self._btn_stop.pack(side="left", padx=4)

        # Botão salvar CSV
        self._btn_csv = ctk.CTkButton(
            ctrl, text="💾", width=42, height=32,
            fg_color="#1c2128", border_color=BORDER, border_width=1,
            text_color=TEXT_DIM, hover_color="#1c2128",
            state="disabled", command=self._save_csv,
        )
        self._btn_csv.pack(side="left", padx=4)

    # ── Barra de dados ────────────────────────────────────────────────────────

    def _build_data_bar(self):
        bar = ctk.CTkFrame(self, fg_color=APP_BG, height=90, corner_radius=0)
        bar.grid(row=1, column=0, sticky="ew", padx=0, pady=4)
        bar.grid_columnconfigure((0,1,2,3,4,5), weight=1, uniform="cards")

        self._card_T    = DataCard(bar, "Período  T",      WAVE_COLOR)
        self._card_f    = DataCard(bar, "Frequência  f",   WAVE_COLOR)
        self._card_f1   = DataCard(bar, "Pico  f₁",        PEAK_COLOR)
        self._card_f2   = DataCard(bar, "Pico  f₂",        PEAK2_COLOR)
        self._card_bat  = DataCard(bar, "Batimento  |f₁−f₂|", "#ff88ff")
        self._card_med  = DataCard(bar, "Freq. Média  (f₁+f₂)/2", CYAN)

        for col, card in enumerate((self._card_T, self._card_f,
                                    self._card_f1, self._card_f2,
                                    self._card_bat, self._card_med)):
            card.grid(row=0, column=col, sticky="nsew", padx=4, pady=4)

    # ── Gráficos ──────────────────────────────────────────────────────────────

    def _build_plots(self):
        plot_frame = ctk.CTkFrame(self, fg_color=SCOPE_BG, corner_radius=0)
        plot_frame.grid(row=2, column=0, sticky="nsew", padx=0, pady=0)
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        # Figura com dois subplots empilhados (70% / 30%)
        self._fig = Figure(facecolor=SCOPE_BG)
        gs = GridSpec(2, 1, figure=self._fig,
                      height_ratios=[2, 1],
                      hspace=0.08,
                      left=0.07, right=0.97, top=0.96, bottom=0.08)

        # ── Plot 1: Forma de onda ──────────────────────────────────────────
        self._ax_wave = self._fig.add_subplot(gs[0])
        self._style_ax(self._ax_wave,
                       ylabel="Tensão  (V)",
                       title="FORMA DE ONDA")
        self._ax_wave.set_xticklabels([])       # oculta eixo X no gráfico de cima

        self._ln_wave, = self._ax_wave.plot(
            [], [], color=WAVE_COLOR, linewidth=1.2, antialiased=True)

        # ── Plot 2: Espectro FFT ───────────────────────────────────────────
        self._ax_fft = self._fig.add_subplot(gs[1])
        self._style_ax(self._ax_fft,
                       xlabel="Frequência  (Hz)",
                       ylabel="Amplitude")
        self._ax_fft.set_title("ESPECTRO FFT", color=TEXT_DIM,
                               fontsize=9, pad=3, loc="left")

        self._ln_fft, = self._ax_fft.plot(
            [], [], color=FFT_COLOR, linewidth=1.0)
        self._vl_f1 = self._ax_fft.axvline(
            np.nan, color=PEAK_COLOR,  lw=1.4, ls="--", alpha=0.9)
        self._vl_f2 = self._ax_fft.axvline(
            np.nan, color=PEAK2_COLOR, lw=1.4, ls="--", alpha=0.9)

        # Labels dos picos (criados uma vez, reposicionados a cada update)
        self._txt_f1 = self._ax_fft.text(
            0, 0.9, "", color=PEAK_COLOR,
            fontsize=8, transform=self._ax_fft.get_yaxis_transform())
        self._txt_f2 = self._ax_fft.text(
            0, 0.75, "", color=PEAK2_COLOR,
            fontsize=8, transform=self._ax_fft.get_yaxis_transform())

        # Canvas
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._canvas.draw()

    def _style_ax(self, ax, xlabel="", ylabel="", title=""):
        ax.set_facecolor(SCOPE_BG)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        if xlabel: ax.set_xlabel(xlabel, labelpad=4)
        if ylabel: ax.set_ylabel(ylabel, labelpad=4)
        if title:  ax.set_title(title, color=TEXT_DIM, fontsize=9, pad=3, loc="left")
        for spine in ax.spines.values():
            spine.set_edgecolor(SCOPE_GRID)
        ax.grid(True, color=SCOPE_GRID, linewidth=0.6)

    # ── Barra de log ──────────────────────────────────────────────────────────

    def _build_logbar(self):
        bar = ctk.CTkFrame(self, fg_color=CARD_BG, height=22, corner_radius=0)
        bar.grid(row=3, column=0, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)
        bar.grid_propagate(False)

        ctk.CTkLabel(bar, text=" LOG ", font=FONT_LABEL,
                     text_color=TEXT_DIM,
                     fg_color="#0d1117").grid(row=0, column=0, sticky="ns")
        self._log_lbl = ctk.CTkLabel(bar, text="Iniciando…",
                                     font=FONT_MONO, text_color=TEXT_MAIN,
                                     anchor="w")
        self._log_lbl.grid(row=0, column=1, sticky="ew", padx=8)
        self._log_lbl.bind("<Button-1>", lambda _: self._show_log())
        bar.bind("<Button-1>",           lambda _: self._show_log())

    def _show_log(self):
        pop = ctk.CTkToplevel(self)
        pop.title("Log")
        pop.geometry("700x360")
        pop.configure(fg_color=APP_BG)
        txt = ctk.CTkTextbox(pop, fg_color=CARD_BG, text_color=TEXT_MAIN,
                             font=FONT_MONO)
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", "\n".join(self._log_history))
        txt.configure(state="disabled")

    # ─── Conexão ──────────────────────────────────────────────────────────────

    def _scan_thread(self):
        self._dpo_dot.configure(text_color=AMBER)
        self._dpo_name.configure(text="DPO: Procurando…", text_color=AMBER)
        self._log("Escaneando recursos VISA…")
        threading.Thread(target=self._do_scan, daemon=True).start()

    def _do_scan(self):
        s = self._conn.scan_and_connect()
        self.after(0, lambda: self._apply_scan(s))

    def _apply_scan(self, s: dict):
        if s["dpo_connected"]:
            self._dpo_dot.configure(text_color=GREEN)
            self._dpo_name.configure(
                text=f"DPO: {s['dpo_name'][:28]}", text_color=TEXT_MAIN)
            self._log(f"DPO ✓  {s['dpo_name']}")
        else:
            self._dpo_dot.configure(text_color=RED)
            self._dpo_name.configure(text="DPO: Não encontrado", text_color=TEXT_DIM)
            self._log("DPO ✗  não encontrado. Verifique o cabo USB/GPIB.", warn=True)
        for e in s["errors"]:
            self._log(f"⚠  {e}", warn=True)

    # ─── Captura ──────────────────────────────────────────────────────────────

    def _start_capture(self):
        if not self._conn.ready:
            self._log("✗  Conecte o osciloscópio antes de capturar.", err=True)
            return

        try:
            interval = float(self._interval.get())
            if interval < 0.1:
                interval = 0.1
        except ValueError:
            interval = 0.5

        ch = int(self._ch_var.get().replace("CH", ""))

        self._btn_cap.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._btn_csv.configure(state="disabled")
        self._log(f"▶  Capturando CH{ch} a cada {interval:.1f} s…")

        self._worker = CaptureWorker(
            conn_manager=self._conn,
            channel=ch,
            interval=interval,
            on_capture=self._cb_capture,
            on_error=self._cb_error,
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

    # ── Callbacks do worker ───────────────────────────────────────────────────

    def _cb_capture(self, time_arr, voltage_arr, metrics):
        self.after(0, lambda: self._gui_update(time_arr, voltage_arr, metrics))

    def _cb_error(self, msg):
        self.after(0, lambda: self._log(f"✗  {msg}", err=True))

    # ── Atualização da GUI ────────────────────────────────────────────────────

    def _gui_update(self, t, v, m: dict):
        self._last_time    = t
        self._last_voltage = v
        self._last_metrics = m

        # ── Cards de dados ──────────────────────────────────────────────────
        self._card_T  .update(calc.fmt_time(m["T"]))
        self._card_f  .update(calc.fmt_hz(m["f"]))
        self._card_f1 .update(calc.fmt_hz(m["f1"]))
        self._card_f2 .update(calc.fmt_hz(m["f2"]))
        self._card_bat.update(calc.fmt_hz(m["f_bat"]))
        self._card_med.update(calc.fmt_hz(m["f_med"]))

        # ── Forma de onda ───────────────────────────────────────────────────
        self._ln_wave.set_data(t * 1e3, v)      # tempo em ms no eixo X
        self._ax_wave.relim()
        self._ax_wave.autoscale_view()
        self._ax_wave.set_xlabel("Tempo  (ms)", color=TEXT_DIM, labelpad=4)

        # Grade de referência em y=0
        self._ax_wave.axhline(0, color=SCOPE_GRID, linewidth=0.8, linestyle="-")

        # ── FFT ─────────────────────────────────────────────────────────────
        freqs = m["freqs"]
        amps  = m["amps"]

        # Limita exibição a 0–2000 Hz (faixa dos diapasões)
        mask = freqs <= 2000
        self._ln_fft.set_data(freqs[mask], amps[mask])
        self._ax_fft.relim()
        self._ax_fft.autoscale_view()
        self._ax_fft.set_xlim(0, 2000)

        # Linhas verticais dos picos
        if m["f1"] > 0:
            self._vl_f1.set_xdata([m["f1"], m["f1"]])
            self._txt_f1.set_text(f"f₁={calc.fmt_hz(m['f1'])}")
            self._txt_f1.set_x(m["f1"] / 2000)
        if m["f2"] > 0:
            self._vl_f2.set_xdata([m["f2"], m["f2"]])
            self._txt_f2.set_text(f"f₂={calc.fmt_hz(m['f2'])}")
            self._txt_f2.set_x(m["f2"] / 2000)

        self._canvas.draw_idle()

    # ─── Salvar CSV ───────────────────────────────────────────────────────────

    def _save_csv(self):
        if self._last_time is None:
            return

        m   = self._last_metrics
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_pretty = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        ch  = self._ch_var.get()
        path = Path(f"captura_{ts}.csv")

        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)

            # ── Cabeçalho / Metadados ─────────────────────────────────────
            w.writerow(["# ============================================================"])
            w.writerow(["# EXPERIMENTO: BATIMENTOS E RESSONÂNCIA"])
            w.writerow(["# ============================================================"])
            w.writerow([f"# Data/Hora:             {ts_pretty}"])
            w.writerow([f"# Canal:                 {ch}"])
            w.writerow([f"# Pontos capturados:     {len(self._last_time)}"])
            w.writerow(["#"])
            w.writerow(["# MÉTRICAS CALCULADAS"])
            w.writerow([f"# Período (T):           {calc.fmt_time(m['T'])}"])
            w.writerow([f"# Frequência (f=1/T):    {calc.fmt_hz(m['f'])}"])
            w.writerow([f"# Pico f1 (FFT):         {calc.fmt_hz(m['f1'])}"])
            w.writerow([f"# Pico f2 (FFT):         {calc.fmt_hz(m['f2'])}"])
            w.writerow([f"# Batimento |f1-f2|:     {calc.fmt_hz(m['f_bat'])}"])
            w.writerow([f"# Freq. Média (f1+f2)/2: {calc.fmt_hz(m['f_med'])}"])
            w.writerow([f"# Tensão máxima:         {m['v_max']:.4f} V"])
            w.writerow([f"# Tensão mínima:         {m['v_min']:.4f} V"])
            w.writerow(["#"])
            w.writerow(["# Ref: Batimentos e Ressonância — RBEF/SciELO"])
            w.writerow(["# https://www.scielo.br/j/rbef/a/D7k5Pxj7HcmmbpGZJMf4wNs/"])
            w.writerow(["# ============================================================"])
            w.writerow([])

            # ── Seção 1: Forma de Onda ────────────────────────────────────
            w.writerow(["# SEÇÃO 1 — FORMA DE ONDA"])
            w.writerow(["tempo_s", "tensao_v"])
            for t, v in zip(self._last_time, self._last_voltage):
                w.writerow([f"{t:.9f}", f"{v:.6f}"])

            w.writerow([])

            # ── Seção 2: Espectro FFT ─────────────────────────────────────
            w.writerow(["# SEÇÃO 2 — ESPECTRO FFT"])
            w.writerow(["frequencia_hz", "amplitude_norm"])
            for f, a in zip(m["freqs"], m["amps"]):
                if f <= 5000:                      # salva até 5 kHz
                    w.writerow([f"{f:.4f}", f"{a:.6f}"])

        self._log(f"💾  Salvo: {path.resolve()}")

    # ─── Log ──────────────────────────────────────────────────────────────────

    def _log(self, msg: str, warn: bool = False, err: bool = False):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {msg}"
        self._log_history.append(line)
        color = RED if err else (AMBER if warn else TEXT_MAIN)
        self._log_lbl.configure(text=line[-115:], text_color=color)

    # ─── Fechar ───────────────────────────────────────────────────────────────

    def _on_close(self):
        if self._worker and self._worker.is_alive():
            self._worker.stop()
        self._conn.close_all()
        self.destroy()
