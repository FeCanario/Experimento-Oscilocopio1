# worker.py
"""
Thread de varredura em frequência do experimento RLC.

Fluxo de cada ponto:
  1. AFG define a frequência f e a amplitude V_in.
  2. DPO ajusta a timebase (~3 ciclos visíveis na tela).
  3. DPO lê V_in (CH1) e V_out (CH2) em Vpp.
  4. Calcula ganho G = V_out / V_in.
  5. Notifica a GUI via callback on_step.

Ao terminar (ou abortar), chama on_finish com a lista completa de pontos.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np


class MeasurementWorker(threading.Thread):
    """
    Thread daemon que varre o espectro de frequências e coleta dados.

    Parâmetros do construtor
    ------------------------
    conn_manager : ConnectionManager
        Instância já conectada com .afg e .dpo disponíveis.
    f_start, f_stop : float
        Limites da varredura em Hz.
    n_steps : int
        Número de pontos (escala logarítmica).
    v_in : float
        Amplitude do sinal de entrada em Vpp.
    delay_ms : int
        Tempo de espera entre setar a frequência e ler (ms).
        Padrão = 400 ms — suficiente para o scope sincronizar o trigger.
    on_step : Callable(freq, v_in_meas, v_out_meas, gain, progress)
        Chamado após cada ponto. Executado na thread do worker — use
        `root.after()` na GUI para atualizações de UI.
    on_finish : Callable(results: list[dict])
        Chamado ao final (normal ou abortado). Lista de dicts com chaves:
        freq, v_in, v_out, gain.
    on_error : Callable(msg: str)
        Chamado se ocorrer exceção não tratada durante a varredura.
    """

    def __init__(
        self,
        conn_manager,
        f_start: float,
        f_stop: float,
        n_steps: int,
        v_in: float,
        on_step: Callable,
        on_finish: Callable,
        on_error: Callable,
        delay_ms: int = 400,
    ):
        super().__init__(daemon=True, name="MeasurementWorker")
        self.mgr      = conn_manager
        self.f_start  = float(f_start)
        self.f_stop   = float(f_stop)
        self.n_steps  = int(n_steps)
        self.v_in     = float(v_in)
        self.delay_ms = int(delay_ms)
        self.on_step   = on_step
        self.on_finish = on_finish
        self.on_error  = on_error
        self._stop_event = threading.Event()

    # ── Controle externo ──────────────────────────────────────────────────────

    def stop(self):
        """Solicita parada segura da varredura."""
        self._stop_event.set()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    # ── Lógica principal ──────────────────────────────────────────────────────

    def run(self):
        afg = self.mgr.afg
        dpo = self.mgr.dpo
        results: list[dict] = []

        try:
            # ── 1. Gera vetor de frequências (escala log) ──────────────────
            freqs = np.logspace(
                np.log10(self.f_start),
                np.log10(self.f_stop),
                self.n_steps,
            )

            # ── 2. Configura AFG ───────────────────────────────────────────
            afg.set_high_z()
            afg.set_waveform(shape="SIN", freq=self.f_start, amplitude=self.v_in)
            afg.output_on()

            # ── 3. Configura DPO ───────────────────────────────────────────
            #   CH1 = V_in  (monitoramento do sinal do gerador)
            #   CH2 = V_out (tensão sobre o resistor R)
            scale_inicial = max(self.v_in / 4.0, 0.01)
            dpo.configure_channel(ch=1, scale=scale_inicial)
            dpo.configure_channel(ch=2, scale=scale_inicial)
            dpo.setup_vpp_measurement(ch=1, meas_num=1)   # slot 1 → V_in
            dpo.setup_vpp_measurement(ch=2, meas_num=2)   # slot 2 → V_out

            time.sleep(2.0)   # estabilização inicial

            # ── 4. Loop de varredura ───────────────────────────────────────
            for i, freq in enumerate(freqs):
                if self.stopped:
                    break

                # Seta frequência no gerador
                afg.set_frequency(freq)

                # Ajusta timebase: ~3 ciclos em 10 divisões → s/div = (3/f)/10
                period   = 1.0 / freq
                tb_scale = (period * 3.0) / 10.0
                dpo.configure_timebase(scale=tb_scale)

                # Aguarda estabilização do trigger
                time.sleep(self.delay_ms / 1000.0)

                # Lê tensões Vpp
                vin_meas  = dpo.get_vpp(meas_num=1)
                vout_meas = dpo.get_vpp(meas_num=2)

                # Ganho G = V_out / V_in
                # Se CH1 não estiver disponível, usa o V_in nominal do AFG
                vin_ref = vin_meas if vin_meas > 1e-6 else self.v_in
                gain    = vout_meas / vin_ref if vin_ref > 0 else 0.0

                point = {
                    "freq":  float(freq),
                    "v_in":  float(vin_meas),
                    "v_out": float(vout_meas),
                    "gain":  float(gain),
                }
                results.append(point)

                progress = (i + 1) / self.n_steps
                self.on_step(freq, vin_meas, vout_meas, gain, progress)

        except Exception as exc:
            self.on_error(str(exc))

        finally:
            # Garante que o AFG é desligado ao final (sucesso ou erro)
            try:
                afg.output_off()
            except Exception:
                pass

            self.on_finish(results)
