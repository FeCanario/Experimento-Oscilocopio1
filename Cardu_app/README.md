# RLC Experimento — Análise de Ressonância

App de bancada para aquisição de dados de um circuito **RLC série** usando
gerador de funções (AFG) e osciloscópio (DPO) da Tektronix via interface VISA.

Calcula automaticamente **f₀, Q, BW, f₁, f₂** a partir dos dados medidos,
usando o modelo do oscilador harmônico forçado amortecido descrito em:

> *Batimentos e Ressonância* — Revista Brasileira de Ensino de Física (RBEF/SciELO)
> https://www.scielo.br/j/rbef/a/D7k5Pxj7HcmmbpGZJMf4wNs/

---

## Estrutura do projeto

```
RLC_App/
├── main.py            # Ponto de entrada — execute este arquivo
├── app.py             # Interface gráfica (CustomTkinter)
├── instruments.py     # Drivers VISA: TektronixAFG, TektronixDPO, ConnectionManager
├── worker.py          # Thread de varredura em frequência
├── calculations.py    # Física do RLC (f₀, Q, BW, ajuste Levenberg–Marquardt)
├── requirements.txt   # Dependências Python
└── README.md          # Este arquivo
```

---

## Instalação

```powershell
cd c:\Dalke\RLC_App
pip install -r requirements.txt
```

### Dependências

| Pacote | Uso |
|---|---|
| `customtkinter` | Interface gráfica moderna |
| `matplotlib` | Gráfico Bode embutido |
| `numpy` | Cálculos numéricos |
| `pyvisa` | Comunicação VISA com instrumentos |
| `pyvisa-py` | Backend VISA puro Python (alternativa ao NI-VISA) |
| `scipy` | Ajuste de curva Levenberg–Marquardt *(opcional, mas recomendado)* |

> **Nota:** Se você já tiver o **NI-VISA** instalado (software Tektronix),
> não precisa do `pyvisa-py`.

---

## Como executar

```powershell
cd c:\Dalke\RLC_App
python main.py
```

---

## Como usar o app

### 1. Montagem do circuito

```
AFG CH1 OUT ──┬── R ──┬── L ──┬── C ──┬── GND (AFG)
              │        │       │        │
           DPO CH1   (medição V_in)   DPO CH2
                                    (medição V_out sobre R)
```

- **CH1 do osciloscópio** → mede **V_in** (sinal de entrada do gerador)
- **CH2 do osciloscópio** → mede **V_out** (tensão sobre o resistor R)
- O ganho medido é: `G = V_out / V_in`

### 2. Conectar os instrumentos

1. Ligue o AFG e o DPO via **cabo USB** no computador
2. Abra o app (`python main.py`)
3. Clique em **CONECTAR** na topbar
4. Aguarde os badges ficarem verdes: `AFG ●` e `DPO ●`

### 3. Preencher os parâmetros

Na **barra de controles** (parte inferior):

| Campo | Descrição |
|---|---|
| `R [Ω]` | Valor do resistor (opcional — só para curva teórica) |
| `L [mH]` | Valor do indutor (opcional) |
| `C [µF]` | Valor do capacitor (opcional) |
| `f início [Hz]` | Frequência inicial da varredura |
| `f fim [Hz]` | Frequência final da varredura |
| `Pontos` | Número de pontos medidos (mínimo 5, recomendado 50–100) |
| `V_in [Vpp]` | Amplitude do sinal gerado pelo AFG |
| `Delay [ms]` | Tempo de espera entre cada ponto para o scope estabilizar |

### 4. Iniciar a medição

1. Clique **▶ INICIAR**
2. O gráfico atualiza em tempo real a cada ponto medido
3. Ao terminar, o ajuste de curva é aplicado automaticamente
4. As métricas aparecem no **painel LCD** à direita

### 5. Resultados

O painel lateral mostra:

| Métrica | Fórmula | Descrição |
|---|---|---|
| **f₀** | `1 / (2π√LC)` | Frequência de ressonância |
| **Q** | `(1/R)√(L/C)` | Fator de qualidade |
| **BW** | `f₀ / Q` | Largura de banda (−3 dB) |
| **f₁** | solução exata | Frequência de meia-potência inferior |
| **f₂** | solução exata | Frequência de meia-potência superior |

Cada métrica exibe dois valores:
- **Linha grande (ciano):** resultado do ajuste experimental
- **Linha pequena (cinza):** valor teórico calculado a partir de R, L, C

### 6. Exportar

Clique em **💾 CSV** para salvar os dados em:
```
rlc_YYYYMMDD_HHMMSS.csv
```

Colunas: `freq`, `v_in`, `v_out`, `gain`

---

## Base teórica (do artigo)

O circuito RLC série é o equivalente elétrico do
**oscilador harmônico forçado amortecido** (artigo, Eq. 4):

```
m·ẍ + b·ẋ + k·x = F₀·cos(ωt)
```

Analogia elétrica:

```
Massa  m   ↔  Indutância  L
Amort. b   ↔  Resistência R
Rigidez k  ↔  1/Capacitância (1/C)
Força  F₀  ↔  Tensão V_in
```

Amplitude de ressonância (artigo, Eq. 6):

```
x₀ = F₀ / √[(k − mω²)² + (bω)²]
```

Equivalente elétrico (ganho no resistor):

```
|H(jω)| = V_out/V_in = R / √[R² + (ωL − 1/ωC)²]
```

Frequência natural (artigo, Eq. 7):

```
ω₀ = √(k/m)   →   f₀ = 1 / (2π√(LC))
```

---

## O que ainda precisa ser feito

### 🔴 Prioridade alta

- [ ] **Testar com o hardware real**
  - Verificar se os comandos SCPI do AFG e DPO funcionam nos modelos disponíveis
  - Ajustar o `setup_vpp_measurement()` se o modelo de DPO usar sintaxe diferente
  - Confirmar que CH1 e CH2 são lidos simultaneamente sem interferência

- [ ] **Validar o cálculo de ganho**
  - Confirmar que `G = V_out_CH2 / V_in_CH1` está correto para o circuito montado
  - Se o AFG não estiver conectado ao CH1, usar `v_in` nominal como referência

- [ ] **Auto-escala vertical do DPO**
  - Atualmente a escala é fixada em `V_in / 4` no início
  - Em frequências longe da ressonância, V_out pode ser muito pequeno
  - Implementar leitura, verificação e reajuste da escala se o valor ficar < 5% do fundo de escala

### 🟡 Prioridade média

- [ ] **Seleção de canal**
  - Permitir ao usuário escolher em qual canal está V_in e em qual está V_out
  - Adicionar dropdowns `CH_Vin` e `CH_Vout` na barra de controles

- [ ] **Salvar e carregar configuração**
  - Guardar os últimos valores de R, L, C, f_start, f_stop, etc. em `config.json`
  - Recarregar automaticamente ao abrir o app

- [ ] **Comparar múltiplas varreduras**
  - Manter curvas anteriores no gráfico (com cores diferentes e transparência)
  - Botão "Adicionar ao gráfico" em vez de sempre limpar

- [ ] **Exportar gráfico**
  - Botão "📷 Salvar imagem" que exporta o plot em PNG/PDF

- [ ] **Unidades automáticas nos eixos**
  - Exibir Hz, kHz ou MHz automaticamente conforme a faixa

### 🟢 Prioridade baixa / melhorias futuras

- [ ] **Medição de fase**
  - O DPO pode medir a defasagem entre CH1 e CH2
  - Plotar fase (°) no segundo eixo Y (Bode completo: amplitude + fase)

- [ ] **Modo manual (sem instrumentos)**
  - Permitir importar um CSV já existente e calcular as métricas sem conectar hardware

- [ ] **Relatório automático**
  - Gerar um PDF com: gráfico, métricas, parâmetros usados e data/hora

- [ ] **Testes automatizados**
  - Criar `tests/` com testes unitários para `calculations.py`
  - Usar dados sintéticos para validar f₀, Q, BW

- [ ] **Suporte a outros fabricantes**
  - Adicionar drivers para Rigol DS1054Z (osciloscópio popular)
  - Adicionar driver para Siglent SDG (gerador)

---

## Erros comuns e soluções

| Erro | Causa | Solução |
|---|---|---|
| `no default root window` | CTkFont criado fora do app | Já corrigido — use tuples de fonte |
| `AFG ✗ não encontrado` | Instrumento não reconhecido pelo IDN | Verificar se `"AFG"` está no `*IDN?` do aparelho |
| `DPO ✗ não encontrado` | Mesmo acima para osciloscópio | Verificar se `"DPO"`, `"MSO"` ou `"TDS"` está no IDN |
| `Nenhum dado coletado` | Worker abortou antes do primeiro ponto | Ver log completo (clique na barra de log) |
| Valores `0.0` nas medições | DPO não configurou medição corretamente | Verificar sintaxe `MEASUrement:MEAS1` para o modelo do scope |
| Aviso `TCPIP...psutil` | Pacote opcional não instalado | `pip install psutil zeroconf` (não obrigatório) |

---

## Diagrama de fluxo

```
main.py
  └── RLCApp.__init__()
        ├── _build_layout()          ← monta a UI
        └── after(400, _scan_thread) ← conecta instrumentos ao abrir

Usuário clica ▶ INICIAR
  └── _start_sweep()
        └── MeasurementWorker.run()  ← thread separada
              ├── afg.set_waveform()
              ├── dpo.configure_channel() × 2
              └── loop por frequência:
                    ├── afg.set_frequency(f)
                    ├── dpo.configure_timebase()
                    ├── sleep(delay_ms)
                    ├── vin  = dpo.get_vpp(meas=1)
                    ├── vout = dpo.get_vpp(meas=2)
                    ├── gain = vout / vin
                    └── on_step() → GUI atualiza gráfico

Varredura concluída
  └── on_finish()
        └── fit_experimental_curve()  ← Levenberg–Marquardt
              └── atualiza LCD cards (f₀, Q, BW, f₁, f₂)
```
