# instruments.py
"""
Drivers VISA para Gerador de Funções (AFG) e Osciloscópio (DPO).
Extraído e simplificado do RLC_Analyser_Pro.

Suporta:  Tektronix AFG1022 / AFG3000 / AFG31000
          Tektronix DPO2000 / DPO4000 / MSO / TDS
"""

from __future__ import annotations
import time
import pyvisa


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentDriver:
    def __init__(self, resource_name: str, rm: pyvisa.ResourceManager):
        self.resource_name = resource_name
        self.rm = rm
        self.instrument = None
        self.connected = False

    def connect(self):
        self.instrument = self.rm.open_resource(self.resource_name)
        self.instrument.timeout = 5000
        self.connected = True
        print(f"[VISA] Conectado: {self.get_idn()}")

    def disconnect(self):
        if self.instrument:
            try:
                self.instrument.close()
            except Exception:
                pass
            self.connected = False

    def write(self, cmd: str):
        if self.connected and self.instrument:
            self.instrument.write(cmd)

    def query(self, cmd: str) -> str:
        if self.connected and self.instrument:
            return self.instrument.query(cmd).strip()
        return ""

    def get_idn(self) -> str:
        return self.query("*IDN?")

    def reset(self):
        self.write("*RST")
        self.write("*CLS")


# ─────────────────────────────────────────────────────────────────────────────
# Gerador de Funções Arbitrárias — Tektronix AFG
# ─────────────────────────────────────────────────────────────────────────────

class TektronixAFG(InstrumentDriver):
    """
    Controla o gerador de sinais via comandos SCPI.
    Canal 1 em modo senoidal, impedância High-Z.
    """

    def set_high_z(self):
        """Configura saída para carga de alta impedância (>1 kΩ)."""
        self.write("OUTPut1:IMPedance INFinity")

    def set_waveform(
        self,
        shape: str = "SIN",
        freq: float = 1000.0,
        amplitude: float = 1.0,
        offset: float = 0.0,
    ):
        """
        Define a forma de onda completa.
        amplitude : Vpp (pico a pico)
        """
        self.write(f"SOURce1:FUNCtion:SHAPe {shape}")
        self.write(f"SOURce1:FREQuency:FIXed {freq:.6f}")
        self.write(f"SOURce1:VOLTage:LEVel:IMMediate:AMPlitude {amplitude:.4f}")
        self.write(f"SOURce1:VOLTage:LEVel:IMMediate:OFFSet {offset:.4f}")

    def set_frequency(self, freq: float):
        self.write(f"SOURce1:FREQuency:FIXed {freq:.6f}")

    def output_on(self):
        self.write("OUTPut1:STATe ON")

    def output_off(self):
        self.write("OUTPut1:STATe OFF")


# ─────────────────────────────────────────────────────────────────────────────
# Osciloscópio — Tektronix DPO / MSO / TDS
# ─────────────────────────────────────────────────────────────────────────────

class TektronixDPO(InstrumentDriver):
    """
    Controla o osciloscópio via comandos SCPI.

    Canais usados no experimento RLC:
        CH1 → V_in  (tensão na saída do gerador)
        CH2 → V_out (tensão medida sobre o resistor R)
    """

    def configure_channel(self, ch: int = 1, scale: float = 1.0):
        """
        Configura acoplamento DC e escala vertical (V/div).
        scale ≈ Vpp / 4  para visualizar bem a onda.
        """
        self.write(f"CH{ch}:COUPling DC")
        self.write(f"CH{ch}:SCAle {scale:.6f}")
        self.write(f"CH{ch}:POSition 0")

    def configure_timebase(self, scale: float = 1e-3):
        """Escala horizontal em segundos/divisão."""
        self.write(f"HORizontal:SCAle {scale:.10f}")

    def setup_vpp_measurement(self, ch: int = 1, meas_num: int = 1):
        """
        Configura slot de medição automática de Vpp (pico a pico).
        meas_num : slot 1 → V_in (CH1),  slot 2 → V_out (CH2)
        """
        self.write(f"MEASUrement:MEAS{meas_num}:SOUrce1 CH{ch}")
        self.write(f"MEASUrement:MEAS{meas_num}:TYPe PK2Pk")
        self.write(f"MEASUrement:MEAS{meas_num}:STATE ON")

    def get_vpp(self, meas_num: int = 1) -> float:
        """Lê o valor de Vpp do slot especificado."""
        try:
            return float(self.query(f"MEASUrement:MEAS{meas_num}:VALue?"))
        except Exception:
            return 0.0

    def autoset(self):
        """Executa AutoSet (usar apenas no setup inicial, demora ~3 s)."""
        self.write("AUTOSet EXECute")
        time.sleep(3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Gerenciador de Conexão
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """
    Varre todos os recursos VISA disponíveis, identifica os instrumentos
    pelo *IDN? e instancia os drivers correspondentes.
    """

    def __init__(self):
        self.rm = pyvisa.ResourceManager()
        self.afg: TektronixAFG | None = None
        self.dpo: TektronixDPO | None = None

    def scan_and_connect(self) -> dict:
        """
        Escaneia e conecta instrumentos disponíveis.

        Retorna::

            {
              "afg_connected": bool,  "afg_name": str,
              "dpo_connected": bool,  "dpo_name": str,
              "errors": [str, ...]
            }
        """
        status = {
            "afg_connected": False,
            "afg_name": "Não encontrado",
            "dpo_connected": False,
            "dpo_name": "Não encontrado",
            "errors": [],
        }

        try:
            resources = self.rm.list_resources()
        except Exception as exc:
            status["errors"].append(f"Erro ao listar recursos VISA: {exc}")
            return status

        print(f"[VISA] Recursos encontrados: {resources}")

        for res in resources:
            try:
                tmp = self.rm.open_resource(res)
                tmp.timeout = 3000
                idn = tmp.query("*IDN?").strip()
                tmp.close()
                idn_up = idn.upper()

                if "AFG" in idn_up and not status["afg_connected"]:
                    self.afg = TektronixAFG(res, self.rm)
                    self.afg.connect()
                    status["afg_connected"] = True
                    status["afg_name"] = idn

                elif any(k in idn_up for k in ("DPO", "MSO", "TDS")) and not status["dpo_connected"]:
                    self.dpo = TektronixDPO(res, self.rm)
                    self.dpo.connect()
                    status["dpo_connected"] = True
                    status["dpo_name"] = idn

            except Exception as exc:
                status["errors"].append(f"{res}: {exc}")

        return status

    def close_all(self):
        """Fecha todas as conexões VISA abertas."""
        for inst in (self.afg, self.dpo):
            if inst:
                try:
                    inst.disconnect()
                except Exception:
                    pass

    @property
    def ready(self) -> bool:
        """True se AFG e DPO estiverem conectados e prontos."""
        return (
            self.afg is not None
            and self.afg.connected
            and self.dpo is not None
            and self.dpo.connected
        )
