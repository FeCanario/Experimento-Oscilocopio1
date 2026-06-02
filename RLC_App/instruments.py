# instruments.py
"""
Driver do Osciloscópio Tektronix (DPO/MSO/TDS) via VISA.

Função principal: ler a forma de onda completa do canal escolhido
e retornar arrays de tempo (s) e tensão (V) prontos para plotar e calcular.
"""

from __future__ import annotations
import numpy as np
import pyvisa


# ─────────────────────────────────────────────────────────────────────────────
# Base VISA
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentDriver:
    def __init__(self, resource_name: str, rm: pyvisa.ResourceManager):
        self.resource_name = resource_name
        self.rm            = rm
        self.instrument    = None
        self.connected     = False

    def connect(self):
        self.instrument         = self.rm.open_resource(self.resource_name)
        self.instrument.timeout = 10_000   # 10 s — leitura de waveform pode demorar
        self.connected          = True
        print(f"[VISA] Conectado: {self.get_idn()}")

    def disconnect(self):
        if self.instrument:
            try: self.instrument.close()
            except: pass
            self.connected = False

    def write(self, cmd: str):
        if self.connected:
            self.instrument.write(cmd)

    def query(self, cmd: str) -> str:
        if self.connected:
            return self.instrument.query(cmd).strip()
        return ""

    def get_idn(self) -> str:
        return self.query("*IDN?")


# ─────────────────────────────────────────────────────────────────────────────
# Osciloscópio Tektronix DPO / MSO / TDS
# ─────────────────────────────────────────────────────────────────────────────

class TektronixDPO(InstrumentDriver):
    """
    Controla o osciloscópio e lê formas de onda via SCPI.

    Fluxo de leitura de waveform:
      1. DATA:SOURCE CHx        → seleciona o canal
      2. DATA:ENC ASCII         → dados em texto (simples e portável)
      3. DATA:START / STOP      → pontos a transferir
      4. WFMPRE:XINCR / XZERO  → escala de tempo
      5. WFMPRE:YMULT / YOFF / YZERO → escala de tensão
      6. CURVE?                 → lê os dados brutos
    """

    # ── Configuração básica de canal ─────────────────────────────────────────

    def configure_channel(self, ch: int = 1, scale: float = 1.0):
        """Acoplamento DC e escala vertical (V/div)."""
        self.write(f"CH{ch}:COUPling DC")
        self.write(f"CH{ch}:SCAle {scale:.6f}")
        self.write(f"CH{ch}:POSition 0")

    def set_timebase(self, scale: float = 1e-3):
        """Escala horizontal em s/div."""
        self.write(f"HORizontal:SCAle {scale:.10f}")

    # ── Leitura de forma de onda ─────────────────────────────────────────────

    def read_waveform(self, ch: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Lê a forma de onda completa do canal `ch`.

        Retorna
        -------
        time_s    : np.ndarray — vetor de tempo em segundos
        voltage_v : np.ndarray — vetor de tensão em volts

        Comandos SCPI usados
        --------------------
        DATA:SOURCE CHx   → canal de origem
        DATA:ENC ASCII    → codificação texto (vírgulas)
        DATA:START 1      → primeiro ponto
        DATA:STOP 10000   → último ponto (até 10 000 pts)
        WFMPRE:XINCR?     → intervalo de tempo entre pontos (s)
        WFMPRE:XZERO?     → tempo do ponto inicial (s)
        WFMPRE:YMULT?     → fator de conversão ADC → Volt
        WFMPRE:YOFF?      → offset ADC (em unidades digitais)
        WFMPRE:YZERO?     → tensão de referência zero (V)
        CURVE?            → dados brutos em ADC counts
        """
        # Configura transferência
        self.write(f"DATA:SOURCE CH{ch}")
        self.write("DATA:ENC ASCII")
        self.write("DATA:START 1")
        self.write("DATA:STOP 10000")

        # Parâmetros de escala
        xincr = float(self.query("WFMPRE:XINCR?"))
        xzero = float(self.query("WFMPRE:XZERO?"))
        ymult = float(self.query("WFMPRE:YMULT?"))
        yoff  = float(self.query("WFMPRE:YOFF?"))
        yzero = float(self.query("WFMPRE:YZERO?"))

        # Dados brutos (string separada por vírgulas)
        raw_str = self.query("CURVE?")
        raw     = np.array([float(v) for v in raw_str.split(",")])

        # Converte para tempo e tensão
        time_s    = xzero + np.arange(len(raw)) * xincr
        voltage_v = (raw - yoff) * ymult + yzero

        return time_s, voltage_v

    # ── Medições automáticas (auxiliares) ────────────────────────────────────

    def measure_frequency(self, ch: int = 1) -> float:
        """Lê a frequência medida automaticamente pelo scope (Hz)."""
        self.write(f"MEASUrement:IMMed:SOUrce1 CH{ch}")
        self.write("MEASUrement:IMMed:TYPe FREQuency")
        try:
            return float(self.query("MEASUrement:IMMed:VALue?"))
        except Exception:
            return 0.0

    def measure_period(self, ch: int = 1) -> float:
        """Lê o período medido automaticamente pelo scope (s)."""
        self.write(f"MEASUrement:IMMed:SOUrce1 CH{ch}")
        self.write("MEASUrement:IMMed:TYPe PERIod")
        try:
            return float(self.query("MEASUrement:IMMed:VALue?"))
        except Exception:
            return 0.0

    def autoset(self):
        """AutoSet — centraliza e escala o sinal automaticamente."""
        self.write("AUTOSet EXECute")
        import time; time.sleep(3.0)

    # ── Configuração horizontal (timebase) ───────────────────────────────────

    def set_timebase_scale(self, scale: float):
        """Escala horizontal em s/div. DPO2012B: 1ns/div a 50s/div."""
        self.write(f"HORizontal:SCAle {scale:.10f}")

    def set_record_length(self, length: int):
        """Comprimento do registro. DPO2012B: 1000, 10000, 100000, 1000000."""
        self.write(f"HORizontal:RECOrdlength {length}")

    def get_timebase_scale(self) -> float:
        try: return float(self.query("HORizontal:SCAle?"))
        except: return 0.0

    def get_record_length(self) -> int:
        try: return int(float(self.query("HORizontal:RECOrdlength?")))
        except: return 0

    # ── Configuração vertical (canal) ────────────────────────────────────────

    def set_ch_scale(self, ch: int, scale: float):
        """Escala vertical em V/div."""
        self.write(f"CH{ch}:SCAle {scale:.6f}")

    def set_ch_coupling(self, ch: int, coupling: str):
        """Acoplamento: AC | DC | GND."""
        self.write(f"CH{ch}:COUPling {coupling}")

    def set_ch_bandwidth(self, ch: int, bw: str):
        """Largura de banda: TWEnty (20MHz) | FULl (100MHz)."""
        self.write(f"CH{ch}:BANdwidth {bw}")

    def set_ch_position(self, ch: int, pos: float):
        """Posição vertical em divisões."""
        self.write(f"CH{ch}:POSition {pos:.3f}")

    def get_ch_scale(self, ch: int) -> float:
        try: return float(self.query(f"CH{ch}:SCAle?"))
        except: return 0.0

    def get_ch_coupling(self, ch: int) -> str:
        try: return self.query(f"CH{ch}:COUPling?")
        except: return "DC"

    def get_ch_bandwidth(self, ch: int) -> str:
        try: return self.query(f"CH{ch}:BANdwidth?")
        except: return "FUL"

    # ── Trigger ──────────────────────────────────────────────────────────────

    def set_trigger_source(self, source: str):
        """Fonte do trigger: CH1 | CH2 | EXT."""
        self.write(f"TRIGger:A:EDGE:SOUrce {source}")

    def set_trigger_slope(self, slope: str):
        """Borda do trigger: RISE | FALL."""
        self.write(f"TRIGger:A:EDGE:SLOpe {slope}")

    def set_trigger_level(self, level: float):
        """Nível do trigger em Volts."""
        self.write(f"TRIGger:A:LEVel {level:.4f}")

    def set_trigger_mode(self, mode: str):
        """Modo do trigger: AUTO | NORMal."""
        self.write(f"TRIGger:A:MODe {mode}")

    def get_trigger_source(self) -> str:
        try: return self.query("TRIGger:A:EDGE:SOUrce?")
        except: return "CH1"

    def get_trigger_level(self) -> float:
        try: return float(self.query("TRIGger:A:LEVel?"))
        except: return 0.0

    def get_trigger_mode(self) -> str:
        try: return self.query("TRIGger:A:MODe?")
        except: return "AUTO"

    # ── Aquisição ────────────────────────────────────────────────────────────

    def set_acquire_mode(self, mode: str):
        """Modo: SAMple | PEAKdetect | AVErage | HIRes."""
        self.write(f"ACQuire:MODe {mode}")

    def set_acquire_numavg(self, n: int):
        """Número de médias (modo Average): 2, 4, 8, 16, 32, 64, 128, 256, 512."""
        self.write(f"ACQuire:NUMAVg {n}")

    def get_acquire_mode(self) -> str:
        try: return self.query("ACQuire:MODe?")
        except: return "SAM"

    def read_all_settings(self) -> dict:
        """Lê as configurações atuais do DPO 2012B de uma vez."""
        return {
            "tb_scale":   self.get_timebase_scale(),
            "rec_length": self.get_record_length(),
            "ch1_scale":  self.get_ch_scale(1),
            "ch1_coup":   self.get_ch_coupling(1),
            "ch1_bw":     self.get_ch_bandwidth(1),
            "ch2_scale":  self.get_ch_scale(2),
            "ch2_coup":   self.get_ch_coupling(2),
            "ch2_bw":     self.get_ch_bandwidth(2),
            "trig_src":   self.get_trigger_source(),
            "trig_lvl":   self.get_trigger_level(),
            "trig_mode":  self.get_trigger_mode(),
            "acq_mode":   self.get_acquire_mode(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Gerenciador de conexão
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """
    Varre recursos VISA e conecta o osciloscópio automaticamente.
    (AFG não é necessário — o sinal vem do circuito de diapasões.)
    """

    def __init__(self):
        self.rm  = pyvisa.ResourceManager()
        self.dpo: TektronixDPO | None = None

    def scan_and_connect(self) -> dict:
        status = {
            "dpo_connected": False,
            "dpo_name":      "Não encontrado",
            "errors":        [],
        }

        try:
            resources = self.rm.list_resources()
        except Exception as exc:
            status["errors"].append(f"Erro ao listar recursos VISA: {exc}")
            return status

        print(f"[VISA] Recursos: {resources}")

        for res in resources:
            try:
                tmp         = self.rm.open_resource(res)
                tmp.timeout = 3000
                idn         = tmp.query("*IDN?").strip()
                tmp.close()

                if any(k in idn.upper() for k in ("DPO", "MSO", "TDS")):
                    self.dpo = TektronixDPO(res, self.rm)
                    self.dpo.connect()
                    status["dpo_connected"] = True
                    status["dpo_name"]      = idn
                    break

            except Exception as exc:
                status["errors"].append(f"{res}: {exc}")

        return status

    def close_all(self):
        if self.dpo:
            try: self.dpo.disconnect()
            except: pass

    @property
    def ready(self) -> bool:
        return self.dpo is not None and self.dpo.connected
