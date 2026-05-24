# worker.py
"""
Thread de captura contínua da forma de onda do osciloscópio.

A cada ciclo:
  1. Lê a waveform do canal escolhido via CURVE?
  2. Calcula T, f, FFT, f_bat, f_med
  3. Notifica a GUI via callback on_capture
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import calculations as calc


class CaptureWorker(threading.Thread):
    """
    Thread daemon que captura e analisa formas de onda continuamente.

    Parâmetros
    ----------
    conn_manager : ConnectionManager
        Instância já conectada com .dpo disponível.
    channel : int
        Canal do osciloscópio a capturar (1 ou 2).
    interval : float
        Intervalo entre capturas em segundos (padrão 0.5 s).
    on_capture : Callable(time, voltage, metrics)
        Chamado a cada captura bem-sucedida.
    on_error : Callable(msg: str)
        Chamado se ocorrer exceção.
    """

    def __init__(
        self,
        conn_manager,
        channel:    int,
        interval:   float,
        on_capture: Callable,
        on_error:   Callable,
    ):
        super().__init__(daemon=True, name="CaptureWorker")
        self.mgr        = conn_manager
        self.channel    = channel
        self.interval   = interval
        self.on_capture = on_capture
        self.on_error   = on_error
        self._stop      = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        dpo = self.mgr.dpo

        while not self._stop.is_set():
            try:
                # Lê forma de onda completa
                time_arr, voltage_arr = dpo.read_waveform(ch=self.channel)

                # Calcula todas as grandezas do artigo
                metrics = calc.analyze_waveform(time_arr, voltage_arr)

                # Notifica GUI
                self.on_capture(time_arr, voltage_arr, metrics)

            except Exception as exc:
                self.on_error(str(exc))

            # Aguarda intervalo antes da próxima captura
            self._stop.wait(self.interval)
