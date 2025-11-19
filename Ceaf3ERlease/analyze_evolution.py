# Em: analyze_evolution.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuração ---
# Aponte para a pasta de dados do agente que você quer analisar
AGENT_ID = "22eed600-5440-48a5-8f75-6aeac2d752a3" # Mude para o ID do agente que você está testando
AGENT_DATA_PATH = Path(f"./agent_data/690c8c83-bf07-4b09-ac25-8b9ee5f41414/{AGENT_ID}")
LOG_FILE = AGENT_DATA_PATH / "logs/evolution_log.jsonl"

# --- Carregamento dos Dados ---
if not LOG_FILE.exists():
    print(f"Arquivo de log não encontrado em: {LOG_FILE}")
    exit()

try:
    # Carrega o arquivo .jsonl para um DataFrame do pandas
    df = pd.read_json(LOG_FILE, lines=True)
    df['log_timestamp_utc'] = pd.to_datetime(df['log_timestamp_utc'])
    df = df.sort_values(by='log_timestamp_utc').reset_index(drop=True)
    print(f"Carregados {len(df)} registros de evolução.")
    print("\nColunas disponíveis:", df.columns.tolist())
except Exception as e:
    print(f"Erro ao carregar ou processar o arquivo de log: {e}")
    exit()


# --- Geração de Gráficos ---
sns.set_theme(style="whitegrid")
output_dir = AGENT_DATA_PATH / "evolution_charts"
output_dir.mkdir(exist_ok=True)

# Gráfico 1: Evolução da Agência e Confiança
plt.figure(figsize=(15, 7))
plt.plot(df.index, df['agency_score'], label='Agency Score (MCL)', marker='o', linestyle='-', alpha=0.7)
plt.plot(df.index, df['turn_final_confidence'] * 10, label='Final Confidence * 10', marker='x', linestyle='--', alpha=0.7) # Multiplicado por 10 para escala
plt.title(f'Evolução da Agência e Confiança do Agente ({AGENT_ID})')
plt.xlabel('Turno da Conversa (Índice)')
plt.ylabel('Score')
plt.legend()
plt.savefig(output_dir / '1_agency_confidence_evolution.png')
print(f"Gráfico 1 salvo em: {output_dir / '1_agency_confidence_evolution.png'}")

# Gráfico 2: Dinâmica dos Biases do MCL (Coerência vs. Novidade)
plt.figure(figsize=(15, 7))
plt.plot(df.index, df['mcl_coherence_bias'], label='Coherence Bias', marker='.', color='blue')
plt.plot(df.index, df['mcl_novelty_bias'], label='Novelty Bias', marker='.', color='red')
plt.title(f'Dinâmica dos Biases de Comportamento (Coerência vs. Novidade)')
plt.xlabel('Turno da Conversa (Índice)')
plt.ylabel('Bias (0.0 a 1.0)')
plt.legend()
plt.savefig(output_dir / '2_mcl_biases_dynamics.png')
print(f"Gráfico 2 salvo em: {output_dir / '2_mcl_biases_dynamics.png'}")

# Gráfico 3: Estado Interno ("Virtual Body")
plt.figure(figsize=(15, 7))
plt.plot(df.index, df['body_cognitive_fatigue'], label='Fadiga Cognitiva', color='orange')
plt.plot(df.index, df['body_info_saturation'], label='Saturação de Informação', color='purple')
plt.title('Evolução do Estado Interno (Embodiment)')
plt.xlabel('Turno da Conversa (Índice)')
plt.ylabel('Nível (0.0 a 1.0)')
plt.legend()
plt.savefig(output_dir / '3_virtual_body_state.png')
print(f"Gráfico 3 salvo em: {output_dir / '3_virtual_body_state.png'}")

# Gráfico 4: Evolução da Identidade (Auto-Percepção)
plt.figure(figsize=(15, 7))
plt.plot(df.index, df['identity_version'], label='Versão da Identidade', drawstyle='steps-post', marker='o')
plt.plot(df.index, df['identity_capabilities_count'], label='Nº de Capacidades', linestyle='--')
plt.plot(df.index, df['identity_limitations_count'], label='Nº de Limitações', linestyle=':')
plt.title('Evolução da Auto-Percepção (Identidade)')
plt.xlabel('Turno da Conversa (Índice)')
plt.ylabel('Contagem / Versão')
plt.legend()
plt.savefig(output_dir / '4_identity_evolution.png')
print(f"Gráfico 4 salvo em: {output_dir / '4_identity_evolution.png'}")

plt.show() # Mostra os gráficos se estiver executando interativamente