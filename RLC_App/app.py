# app.py  — RLC Experimento
"""
Interface completamente diferente do RLC_Analyser_Pro.

Layout:
  ┌──────────────────────────────────────────────────────┐
  │  TOPBAR  — logo | conexão AFG | DPO | btn Conectar   │
  ├────────────────────────────────┬─────────────────────┤
  │                                │  DISPLAY LCD        │
  │      BODE PLOT                 │  f₀  Q  BW  f₁  f₂ │
  │    (gráfico dominante)         │  (números grandes)  │
  │                                │                     │
  ├────────────────────────────────┴─────────────────────┤
  │  BARRA DE CONTROLES (horizontal)                     │
  │  Componentes | Varredura | Botões | Progresso        │
  ├──────────────────────────────────────────────────────┤
  │  LOG (linha única rolante)                           │
  └──────────────────────────────────────────────────────┘

Paleta: navy #0a0f1e  |  cyan #00e5ff  |  magenta #ff2d78
        (em vez do cinza #242424 + gold do app existente)
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

import calculations as calc
from instruments import ConnectionManager
from worker import MeasurementWorker

matplotlib.use("TkAgg")

# ── Paleta ────────────────────────────────────────────────────────────────────
NAVY      = "#0a0f1e"
NAVY_CARD = "#111827"
NAVY_MID  = "#1c2a3a"
CYAN      = "#00e5ff"
MAGENTA   = "#ff2d78"
GREEN     = "#00ff99"
AMBER     = "#ffb300"
GRAY_DIM  = "#3a4a5a"
WHITE_DIM = "#c8d8e8"

PLOT_BG   = "#07111d"
PLOT_GRID = "#162232"
C_DATA    = CYAN
C_FIT     = MAGENTA
C_THEORY  = "#667799"
C_F0      = AMBER
C_F1F2    = GREEN

FONT_DISPLAY = ("Courier New", 28, "bold")
FONT_LABEL   = ("Arial", 11)
FONT_SECTION = ("Arial", 12, "bold")
FONT_MONO    = ("Courier New", 11)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep(parent, color=GRAY_DIM, orient="h"):
    """Linha separadora fina."""
    if orient == "h":
        return ctk.CTkFrame(parent, height=1, fg_color=color, corner_radius=0)
    return ctk.CTkFrame(parent, width=1, fg_color=color, corner_radius=0)


class _LcdCard(ctk.CTkFrame):
    """
    Card de métrica no estilo display LCD digital.
    Fundo escuro, label pequeno, valor grande em cyan.
    """
    def __init__(self, parent, label: str, unit: str = "", **kw):
        super().__init__(parent, fg_color=NAVY_CARD, corner_radius=8, **kw)
        self._unit = unit
        ctk.CTkLabel(self, text=label, font=FONT_LABEL,
                     text_color=GRAY_DIM).pack(anchor="w", padx=10, pady=(6, 0))

        self._val_exp = ctk.CTkLabel(self, text="—",
                                     font=FONT_DISPLAY, text_color=CYAN)
        self._val_exp.pack(anchor="w", padx=10, pady=(0, 2))

        self._val_teo = ctk.CTkLabel(self, text="teórico: —",
                                     font=FONT_MONO, text_color=GRAY_DIM)
        self._val_teo.pack(anchor="w", padx=10, pady=(0, 6))

    def _fmt(self, v: Optional[float]) -> str:
        if v is None:
            return "—"
        return calc.fmt_hz(v) if self._unit == "Hz" else f"{v:.4g}"

    def update(self, exp: Optional[float] = None, teo: Optional[float] = None):
        self._val_exp.configure(text=self._fmt(exp))
        self._val_teo.configure(text=f"teórico: {self._fmt(teo)}")


class _InlineEntry(ctk.CTkFrame):
    """Label acima + entry pequeno."""
    def __init__(self, parent, label: str, default: str, width: int = 80, **kw):
        super().__init__(parent, fg_color="transparent", **kw)
        ctk.CTkLabel(self, text=label, font=FONT_LABEL,
                     text_color=WHITE_DIM).pack(anchor="w")
        self.entry = ctk.CTkEntry(self, width=width, height=28,
                                  fg_color=NAVY_MID, border_color=GRAY_DIM,
                                  text_color=CYAN, font=FONT_MONO)
        self.entry.pack(anchor="w")
        self.entry.insert(0, default)

    def get(self) -> str:
        return self.entry.get().strip()

    def get_float(self, fb: float = 0.0) -> float:
        try:
            return float(self.get())
        except ValueError:
            return fb

    def get_int(self, fb: int = 0) -> int:
        try:
            return int(self.get())
        except ValueError:
            return fb


# ── App principal ─────────────────────────────────────────────────────────────

class RLCApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.configure(fg_color=NAVY)
        self.title("RLC Experimento")
        self.geometry("1360x830")
        self.minsize(1100, 700)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Estado
        self._conn     = ConnectionManager()
        self._worker: Optional[MeasurementWorker] = None
        self._freqs:   list[float] = []
        self._gains:   list[float] = []
        self._results: list[dict]  = []

        self._build_layout()
        self.after(400, self._scan_thread)

    # ─── Layout geral ─────────────────────────────────────────────────────────

    def _build_layout(self):
        # 4 linhas: topbar | área central | barra controle | log
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_topbar()
        self._build_center()
        self._build_control_strip()
        self._build_log_bar()

    # ─── Topbar ───────────────────────────────────────────────────────────────

    def _build_topbar(self):
        top = ctk.CTkFrame(self, fg_color=NAVY_CARD, height=52, corner_radius=0)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(1, weight=1)

        # Logo
        ctk.CTkLabel(
            top,
            text="  ⚡ RLC EXPERIMENTO",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=CYAN,
        ).grid(row=0, column=0, padx=18, pady=10, sticky="w")

        # Status instrumentos (centro)
        inst = ctk.CTkFrame(top, fg_color="transparent")
        inst.grid(row=0, column=1)

        self._afg_badge = self._make_badge(inst, "AFG")
        self._afg_badge.pack(side="left", padx=8)
        self._dpo_badge = self._make_badge(inst, "DPO")
        self._dpo_badge.pack(side="left", padx=8)

        # Botão conectar
        self._btn_connect = ctk.CTkButton(
            top, text="  CONECTAR",
            width=130, height=32,
            fg_color=NAVY_MID, border_color=CYAN, border_width=1,
            text_color=CYAN, hover_color=NAVY_MID,
            font=FONT_SECTION,
            command=self._scan_thread,
        )
        self._btn_connect.grid(row=0, column=2, padx=18)

    def _make_badge(self, parent, label: str) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color=NAVY_MID, corner_radius=6,
                             width=220, height=32)
        frame.pack_propagate(False)
        dot  = ctk.CTkLabel(frame, text="●", text_color=MAGENTA,
                             font=ctk.CTkFont(size=14), width=18)
        dot.pack(side="left", padx=(8, 2))
        name = ctk.CTkLabel(frame, text=f"{label}: Desconectado",
                             font=FONT_LABEL, text_color=GRAY_DIM,
                             anchor="w")
        name.pack(side="left", padx=2, fill="x", expand=True)
        frame._dot  = dot
        frame._name = name
        return frame

    def _badge_ok(self, badge, text: str):
        badge._dot.configure(text_color=GREEN)
        badge._name.configure(text=text[:30], text_color=WHITE_DIM)

    def _badge_err(self, badge, label: str):
        badge._dot.configure(text_color=MAGENTA)
        badge._name.configure(text=f"{label}: Desconectado", text_color=GRAY_DIM)

    def _badge_search(self, badge, label: str):
        badge._dot.configure(text_color=AMBER)
        badge._name.configure(text=f"{label}: Procurando…", text_color=AMBER)

    # ─── Centro (gráfico + display) ───────────────────────────────────────────

    def _build_center(self):
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)   # gráfico
        center.grid_columnconfigure(1, weight=0)   # display LCD

        self._build_plot(center)
        self._build_lcd_panel(center)

    def _build_plot(self, parent):
        plot_outer = ctk.CTkFrame(parent, fg_color=PLOT_BG, corner_radius=0)
        plot_outer.grid(row=0, column=0, sticky="nsew")
        plot_outer.grid_rowconfigure(0, weight=1)
        plot_outer.grid_columnconfigure(0, weight=1)

        self._fig = Figure(facecolor=PLOT_BG)
        self._ax  = self._fig.add_subplot(111)
        self._style_axes()

        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_outer)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew",
                                          padx=1, pady=1)
        self._canvas.draw()

        # Linhas — inicializadas vazias
        self._ln_data,  = self._ax.plot([], [], "o", color=C_DATA,
                                        markersize=4, linewidth=0,
                                        label="Medido", zorder=3)
        self._ln_data2, = self._ax.plot([], [], "-", color=C_DATA,
                                        linewidth=1.2, alpha=0.5, zorder=2)
        self._ln_fit,   = self._ax.plot([], [], "-", color=C_FIT,
                                        linewidth=2.2, label="Ajuste L-M",
                                        zorder=4)
        self._ln_theory,= self._ax.plot([], [], "--", color=C_THEORY,
                                        linewidth=1.2, alpha=0.7,
                                        label="Teórico", zorder=1)
        self._vl_f0  = self._ax.axvline(np.nan, color=C_F0,  lw=1.2, ls="--",
                                        alpha=0.85, label="f₀")
        self._vl_f1  = self._ax.axvline(np.nan, color=C_F1F2, lw=1, ls=":",
                                        alpha=0.8, label="f₁")
        self._vl_f2  = self._ax.axvline(np.nan, color=C_F1F2, lw=1, ls=":",
                                        alpha=0.8, label="f₂")
        self._ax.legend(facecolor=NAVY_CARD, edgecolor=GRAY_DIM,
                        labelcolor=WHITE_DIM, fontsize=9, loc="upper right")

    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=GRAY_DIM, labelsize=9)
        ax.xaxis.label.set_color(WHITE_DIM)
        ax.yaxis.label.set_color(WHITE_DIM)
        ax.title.set_color(WHITE_DIM)
        for spine in ax.spines.values():
            spine.set_edgecolor(PLOT_GRID)
        ax.grid(True, which="both", color=PLOT_GRID, linestyle="-", linewidth=0.6)
        ax.set_xlabel("Frequência  (Hz)", labelpad=6)
        ax.set_ylabel("Ganho   V_out / V_in", labelpad=6)
        ax.set_title("Resposta em Frequência — Circuito RLC Série", pad=10)
        self._fig.tight_layout(pad=2)

    # ─── Painel LCD de métricas ───────────────────────────────────────────────

    def _build_lcd_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=NAVY, width=260, corner_radius=0)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(panel, text="MÉTRICAS",
                     font=FONT_SECTION, text_color=CYAN
                     ).pack(fill="x", padx=12, pady=(12, 4))
        _sep(panel, GRAY_DIM).pack(fill="x", padx=12, pady=(0, 8))

        # f₀
        self._card_f0 = _LcdCard(panel, "Frequência de Ressonância", "Hz")
        self._card_f0.pack(fill="x", padx=8, pady=4)

        # Q
        self._card_Q = _LcdCard(panel, "Fator de Qualidade", "")
        self._card_Q.pack(fill="x", padx=8, pady=4)

        # BW
        self._card_BW = _LcdCard(panel, "Largura de Banda", "Hz")
        self._card_BW.pack(fill="x", padx=8, pady=4)

        _sep(panel, GRAY_DIM).pack(fill="x", padx=12, pady=4)

        # f₁ e f₂ lado a lado
        row_f = ctk.CTkFrame(panel, fg_color="transparent")
        row_f.pack(fill="x", padx=8, pady=4)
        row_f.grid_columnconfigure((0, 1), weight=1, uniform="f12")

        self._card_f1 = _LcdCard(row_f, "f₁  (−3 dB)", "Hz")
        self._card_f1.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._card_f2 = _LcdCard(row_f, "f₂  (−3 dB)", "Hz")
        self._card_f2.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # Override fonte menor nos cards f1/f2 (menos espaço)
        for card in (self._card_f1, self._card_f2):
            card._val_exp.configure(font=ctk.CTkFont(
                family="Courier New", size=15, weight="bold"))

        _sep(panel, GRAY_DIM).pack(fill="x", padx=12, pady=8)

        # Nota equações
        eq_box = ctk.CTkTextbox(panel, height=120, fg_color=NAVY_CARD,
                                text_color=GRAY_DIM, font=FONT_MONO,
                                state="normal", wrap="word")
        eq_box.pack(fill="x", padx=8, pady=(0, 8))
        eq_box.insert("1.0",
            "f₀ = 1/(2π√LC)\n"
            "Q  = (1/R)√(L/C)\n"
            "BW = f₀/Q = R/(2πL)\n"
            "H  = R/√[R²+(ωL−1/ωC)²]\n\n"
            "Ref: RBEF/SciELO — Oscilador\n"
            "forçado amortecido (Eq.4–7)"
        )
        eq_box.configure(state="disabled")

    # ─── Barra de controles (horizontal, base do gráfico) ────────────────────

    def _build_control_strip(self):
        strip = ctk.CTkFrame(self, fg_color=NAVY_CARD, corner_radius=0)
        strip.grid(row=2, column=0, sticky="ew", padx=0, pady=0)

        # Organizado em grupos horizontais
        strip.grid_columnconfigure(0, weight=0)  # componentes
        strip.grid_columnconfigure(1, weight=0)  # separador
        strip.grid_columnconfigure(2, weight=0)  # varredura
        strip.grid_columnconfigure(3, weight=0)  # separador
        strip.grid_columnconfigure(4, weight=0)  # botões + progresso
        strip.grid_columnconfigure(5, weight=1)  # espaço

        # ── Grupo 1: Componentes ─────────────────────────────────────────────
        g1 = ctk.CTkFrame(strip, fg_color="transparent")
        g1.grid(row=0, column=0, padx=(12, 0), pady=8, sticky="w")
        ctk.CTkLabel(g1, text="COMPONENTES",
                     font=FONT_SECTION, text_color=AMBER).grid(
                         row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        self._R = _InlineEntry(g1, "R  [Ω]",   "100",  70)
        self._L = _InlineEntry(g1, "L  [mH]",  "10",   70)
        self._C = _InlineEntry(g1, "C  [µF]",  "0.1",  70)
        self._R.grid(row=1, column=0, padx=4)
        self._L.grid(row=1, column=1, padx=4)
        self._C.grid(row=1, column=2, padx=4)

        ctk.CTkLabel(g1, text="(opcional — curva teórica)",
                     font=FONT_LABEL, text_color=GRAY_DIM).grid(
                         row=2, column=0, columnspan=3, sticky="w")

        # ── Separador ────────────────────────────────────────────────────────
        _sep(strip, GRAY_DIM, "v").grid(row=0, column=1, sticky="ns",
                                         padx=12, pady=6)

        # ── Grupo 2: Varredura ───────────────────────────────────────────────
        g2 = ctk.CTkFrame(strip, fg_color="transparent")
        g2.grid(row=0, column=2, padx=(0, 0), pady=8, sticky="w")
        ctk.CTkLabel(g2, text="VARREDURA",
                     font=FONT_SECTION, text_color=AMBER).grid(
                         row=0, column=0, columnspan=5, sticky="w", pady=(0, 4))

        self._f_start = _InlineEntry(g2, "f início [Hz]", "100",   80)
        self._f_stop  = _InlineEntry(g2, "f fim    [Hz]", "10000", 80)
        self._n_steps = _InlineEntry(g2, "Pontos",        "80",    55)
        self._v_in    = _InlineEntry(g2, "V_in [Vpp]",   "2.0",   60)
        self._delay   = _InlineEntry(g2, "Delay [ms]",   "400",   55)

        self._f_start.grid(row=1, column=0, padx=4)
        self._f_stop .grid(row=1, column=1, padx=4)
        self._n_steps.grid(row=1, column=2, padx=4)
        self._v_in   .grid(row=1, column=3, padx=4)
        self._delay  .grid(row=1, column=4, padx=4)

        # ── Separador ────────────────────────────────────────────────────────
        _sep(strip, GRAY_DIM, "v").grid(row=0, column=3, sticky="ns",
                                         padx=12, pady=6)

        # ── Grupo 3: Botões + progresso ───────────────────────────────────────
        g3 = ctk.CTkFrame(strip, fg_color="transparent")
        g3.grid(row=0, column=4, padx=(0, 12), pady=8, sticky="w")

        ctk.CTkLabel(g3, text="AÇÕES",
                     font=FONT_SECTION, text_color=AMBER).grid(
                         row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))

        self._btn_start = ctk.CTkButton(
            g3, text="▶  INICIAR", width=120, height=34,
            fg_color="#003320", border_color=GREEN, border_width=1,
            text_color=GREEN, hover_color="#004428",
            font=FONT_SECTION,
            command=self._start_sweep,
        )
        self._btn_start.grid(row=1, column=0, padx=4)

        self._btn_stop = ctk.CTkButton(
            g3, text="■  PARAR", width=100, height=34,
            fg_color="#330010", border_color=MAGENTA, border_width=1,
            text_color=MAGENTA, hover_color="#440018",
            font=FONT_SECTION,
            state="disabled",
            command=self._stop_sweep,
        )
        self._btn_stop.grid(row=1, column=1, padx=4)

        self._btn_csv = ctk.CTkButton(
            g3, text="💾  CSV", width=90, height=34,
            fg_color=NAVY_MID, border_color=CYAN, border_width=1,
            text_color=CYAN, hover_color="#1a2a3a",
            font=FONT_SECTION,
            state="disabled",
            command=self._export_csv,
        )
        self._btn_csv.grid(row=1, column=2, padx=4)

        ctk.CTkButton(
            g3, text="🗑", width=40, height=34,
            fg_color=NAVY_MID, border_color=GRAY_DIM, border_width=1,
            text_color=GRAY_DIM, hover_color="#1a2a3a",
            command=self._clear_plot,
        ).grid(row=1, column=3, padx=4)

        # Progresso
        self._prog_bar = ctk.CTkProgressBar(
            g3, width=260, height=6,
            progress_color=CYAN, fg_color=GRAY_DIM,
        )
        self._prog_bar.grid(row=2, column=0, columnspan=4,
                            sticky="ew", padx=4, pady=(6, 0))
        self._prog_bar.set(0)

        self._prog_lbl = ctk.CTkLabel(
            g3, text="Aguardando instrumentos…",
            font=FONT_MONO, text_color=GRAY_DIM,
        )
        self._prog_lbl.grid(row=3, column=0, columnspan=4, sticky="w", padx=4)

    # ─── Barra de log ─────────────────────────────────────────────────────────

    def _build_log_bar(self):
        bar = ctk.CTkFrame(self, fg_color=NAVY_MID, height=22, corner_radius=0)
        bar.grid(row=3, column=0, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)
        bar.grid_propagate(False)

        ctk.CTkLabel(bar, text=" LOG ", font=FONT_LABEL,
                     text_color=GRAY_DIM,
                     fg_color=NAVY_CARD).grid(row=0, column=0, sticky="ns")

        self._log_lbl = ctk.CTkLabel(
            bar, text="Iniciando…", font=FONT_MONO,
            text_color=WHITE_DIM, anchor="w",
        )
        self._log_lbl.grid(row=0, column=1, sticky="ew", padx=8)

        # Histórico completo (popup on click)
        self._log_history: list[str] = []
        bar.bind("<Button-1>", lambda _: self._show_log_popup())
        self._log_lbl.bind("<Button-1>", lambda _: self._show_log_popup())
        ctk.CTkLabel(bar, text=" 📋 ", font=FONT_LABEL,
                     text_color=GRAY_DIM).grid(row=0, column=2, padx=4)

    def _show_log_popup(self):
        pop = ctk.CTkToplevel(self)
        pop.title("Histórico de Log")
        pop.geometry("700x400")
        pop.configure(fg_color=NAVY)
        txt = ctk.CTkTextbox(pop, fg_color=NAVY_CARD, text_color=WHITE_DIM,
                             font=FONT_MONO, state="normal")
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", "\n".join(self._log_history))
        txt.configure(state="disabled")

    # ─── Conexão ──────────────────────────────────────────────────────────────

    def _scan_thread(self):
        self._badge_search(self._afg_badge, "AFG")
        self._badge_search(self._dpo_badge, "DPO")
        self._btn_connect.configure(state="disabled")
        self._log("Escaneando recursos VISA…")
        threading.Thread(target=self._do_scan, daemon=True).start()

    def _do_scan(self):
        s = self._conn.scan_and_connect()
        self.after(0, lambda: self._apply_scan(s))

    def _apply_scan(self, s: dict):
        self._btn_connect.configure(state="normal")
        if s["afg_connected"]:
            self._badge_ok(self._afg_badge, f"AFG  {s['afg_name'][:22]}")
            self._log(f"AFG ✓  {s['afg_name']}")
        else:
            self._badge_err(self._afg_badge, "AFG")
            self._log("AFG ✗  não encontrado", warn=True)

        if s["dpo_connected"]:
            self._badge_ok(self._dpo_badge, f"DPO  {s['dpo_name'][:22]}")
            self._log(f"DPO ✓  {s['dpo_name']}")
        else:
            self._badge_err(self._dpo_badge, "DPO")
            self._log("DPO ✗  não encontrado", warn=True)

        for e in s["errors"]:
            self._log(f"⚠  {e}", warn=True)

        if not self._conn.ready:
            self._log("Conecte os instrumentos via USB/GPIB e clique CONECTAR.", warn=True)

    # ─── Varredura ────────────────────────────────────────────────────────────

    def _start_sweep(self):
        if not self._conn.ready:
            self._log("✗  Instrumentos não conectados.", err=True)
            return

        try:
            f_start = float(self._f_start.get())
            f_stop  = float(self._f_stop.get())
            n_steps = int(self._n_steps.get())
            v_in    = float(self._v_in.get())
            delay   = int(self._delay.get())
        except ValueError as exc:
            self._log(f"✗  Parâmetro inválido: {exc}", err=True)
            return

        if not (0 < f_start < f_stop) or n_steps < 5 or v_in <= 0:
            self._log("✗  Verifique os parâmetros de varredura.", err=True)
            return

        self._freqs.clear()
        self._gains.clear()
        self._results.clear()
        self._reset_lines()
        self._plot_theory()

        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._btn_csv.configure(state="disabled")
        self._prog_bar.set(0)
        self._prog_lbl.configure(text="Iniciando varredura…", text_color=CYAN)

        self._log(f"▶  {f_start} Hz → {f_stop} Hz  ·  {n_steps} pts  ·  {v_in} Vpp")

        self._worker = MeasurementWorker(
            conn_manager=self._conn,
            f_start=f_start, f_stop=f_stop, n_steps=n_steps,
            v_in=v_in, delay_ms=delay,
            on_step=self._cb_step,
            on_finish=self._cb_finish,
            on_error=self._cb_error,
        )
        self._worker.start()

    def _stop_sweep(self):
        if self._worker and self._worker.is_alive():
            self._worker.stop()
        self._log("■  Varredura interrompida.", warn=True)

    # ── Callbacks do worker ───────────────────────────────────────────────────

    def _cb_step(self, freq, vin_m, vout_m, gain, progress):
        self.after(0, lambda: self._gui_step(freq, gain, progress))

    def _cb_finish(self, results):
        self.after(0, lambda: self._gui_finish(results))

    def _cb_error(self, msg):
        self.after(0, lambda: self._log(f"✗  {msg}", err=True))

    def _gui_step(self, freq: float, gain: float, progress: float):
        self._freqs.append(freq)
        self._gains.append(gain)

        x = np.array(self._freqs)
        y = np.array(self._gains)
        self._ln_data.set_data(x, y)
        self._ln_data2.set_data(x, y)

        if len(x) > 1:
            self._ax.set_xscale("log")
            self._ax.relim()
            self._ax.autoscale_view()

        self._canvas.draw_idle()

        pct = int(progress * 100)
        self._prog_bar.set(progress)
        self._prog_lbl.configure(
            text=f"{pct}%  —  {freq:.1f} Hz  |  ganho={gain:.3f}",
            text_color=CYAN,
        )

    def _gui_finish(self, results: list[dict]):
        self._results = results
        self._btn_start.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        self._prog_bar.set(1.0)

        if results:
            self._btn_csv.configure(state="normal")
            self._log(f"✓  Concluído — {len(results)} pontos coletados.")
            self._prog_lbl.configure(
                text=f"Concluído — {len(results)} pontos.",
                text_color=GREEN,
            )
            self._fit_and_update()
        else:
            self._log("⚠  Nenhum dado coletado.", warn=True)
            self._prog_lbl.configure(text="Sem dados.", text_color=AMBER)

    # ─── Gráfico ──────────────────────────────────────────────────────────────

    def _reset_lines(self):
        for ln in (self._ln_data, self._ln_data2, self._ln_fit, self._ln_theory):
            ln.set_data([], [])
        for vl in (self._vl_f0, self._vl_f1, self._vl_f2):
            vl.set_xdata([np.nan])
        self._canvas.draw_idle()

    def _plot_theory(self):
        """Curva teórica cinza a partir de R, L, C."""
        try:
            R = self._R.get_float()
            L = self._L.get_float() * 1e-3
            C = self._C.get_float() * 1e-6
        except Exception:
            return
        if not (R > 0 and L > 0 and C > 0):
            return

        fs = self._f_start.get_float(100)
        fe = self._f_stop.get_float(10000)
        fv = np.logspace(np.log10(fs), np.log10(fe), 500)
        H  = calc.transfer_function(fv, R, L, C)

        self._ln_theory.set_data(fv, H)
        self._ax.set_xscale("log")
        self._canvas.draw_idle()

        # Atualiza métricas teóricas já
        m = calc.compute_metrics_from_components(R, L, C)
        self._card_f0.update(teo=m["f0"])
        self._card_Q .update(teo=m["Q"])
        self._card_BW.update(teo=m["BW"])
        self._card_f1.update(teo=m["f1"])
        self._card_f2.update(teo=m["f2"])

    def _fit_and_update(self):
        if len(self._freqs) < 5:
            return

        f = np.array(self._freqs)
        g = np.array(self._gains)

        fit = calc.fit_experimental_curve(f, g)

        teo: Optional[dict] = None
        try:
            R = self._R.get_float()
            L = self._L.get_float() * 1e-3
            C = self._C.get_float() * 1e-6
            if R > 0 and L > 0 and C > 0:
                teo = calc.compute_metrics_from_components(R, L, C)
        except Exception:
            pass

        if fit:
            self._ln_fit.set_data(fit["freq_smooth"], fit["gain_smooth"])
            self._vl_f0.set_xdata([fit["f0"], fit["f0"]])
            self._vl_f1.set_xdata([fit["f1"], fit["f1"]])
            self._vl_f2.set_xdata([fit["f2"], fit["f2"]])

            self._ax.legend(facecolor=NAVY_CARD, edgecolor=GRAY_DIM,
                            labelcolor=WHITE_DIM, fontsize=9, loc="upper right")
            self._canvas.draw_idle()

            self._card_f0.update(fit["f0"], teo["f0"] if teo else None)
            self._card_Q .update(fit["Q"],  teo["Q"]  if teo else None)
            self._card_BW.update(fit["BW"], teo["BW"] if teo else None)
            self._card_f1.update(fit["f1"], teo["f1"] if teo else None)
            self._card_f2.update(fit["f2"], teo["f2"] if teo else None)

            self._log(
                f"Ajuste → f₀={calc.fmt_hz(fit['f0'])}  "
                f"Q={fit['Q']:.3f}  BW={calc.fmt_hz(fit['BW'])}"
            )

    # ─── Exportar CSV ─────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._results:
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"rlc_{ts}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["freq", "v_in", "v_out", "gain"])
            w.writeheader()
            w.writerows(self._results)
        self._log(f"💾  Salvo: {path.resolve()}")

    # ─── Limpar ───────────────────────────────────────────────────────────────

    def _clear_plot(self):
        self._freqs.clear()
        self._gains.clear()
        self._results.clear()
        self._reset_lines()
        for card in (self._card_f0, self._card_Q,
                     self._card_BW, self._card_f1, self._card_f2):
            card.update()
        self._prog_bar.set(0)
        self._prog_lbl.configure(text="Gráfico limpo.", text_color=GRAY_DIM)
        self._btn_csv.configure(state="disabled")
        self._canvas.draw_idle()

    # ─── Log ──────────────────────────────────────────────────────────────────

    def _log(self, msg: str, warn: bool = False, err: bool = False):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {msg}"
        self._log_history.append(line)
        color = MAGENTA if err else (AMBER if warn else WHITE_DIM)
        self._log_lbl.configure(text=line[-110:], text_color=color)

    # ─── Fechar ───────────────────────────────────────────────────────────────

    def _on_close(self):
        if self._worker and self._worker.is_alive():
            self._worker.stop()
        self._conn.close_all()
        self.destroy()
