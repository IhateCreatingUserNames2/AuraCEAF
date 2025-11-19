import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
# Point this to your actual DB file path
DB_PATH = "agent_data/690c8c83-bf07-4b09-ac25-8b9ee5f41414/22eed600-5440-48a5-8f75-6aeac2d752a3/cognitive_turn_history.sqlite"
OUTPUT_FILENAME = "mariana_patient_chart.png"

# --- THEME CONFIGURATION ---
plt.style.use('dark_background')
sns.set_palette("husl")
COLOR_VALENCE = '#00ff9d'  # Matrix Green
COLOR_STRAIN = '#ff0055'  # Warning Red
COLOR_AGENCY = '#00ccff'  # Cyber Blue
COLOR_IDENTITY = '#ffcc00'  # Amber


class TherapistDashboard:
    def __init__(self, db_path):
        self.db_path = db_path

    def fetch_data(self):
        """Fetches raw turn history from SQLite."""
        if not Path(self.db_path).exists():
            print(f"❌ Database not found at: {self.db_path}")
            return pd.DataFrame()

        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT 
                timestamp,
                cognitive_state_packet,
                response_packet,
                mcl_guidance_json
            FROM turn_history
            ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"❌ Error reading database: {e}")
            return pd.DataFrame()

    def parse_metrics(self, df):
        """Parses JSON blobs into a flat metrics DataFrame."""
        metrics = []

        for _, row in df.iterrows():
            try:
                # Parse JSON blobs
                cog_state = json.loads(row['cognitive_state_packet'])
                mcl_guidance = json.loads(row['mcl_guidance_json']) if row['mcl_guidance_json'] else {}

                # 1. Extract Drives (if available in MCL guidance or Cognitive State)
                drives = mcl_guidance.get('drives_state_at_turn', {})

                # 2. Extract Agency & Bias
                mcl_analysis = mcl_guidance.get('mcl_analysis', {})
                biases = mcl_guidance.get('biases', {})

                # 3. Reconstruct "Qualia/Valence" (Approximation based on logs)
                # We look for internal state report in recent memory or metadata
                # Fallback: Calculate proxy based on available metrics
                agency = mcl_analysis.get('agency_score', 0)

                # We try to find 'cognitive_strain' in metadata or calculate a proxy
                # Proxy: High Agency + High Coherence usually = Strain
                # This is a simulation of the Interoception Module's logic
                strain = 0.0
                if agency > 2.0: strain += 0.1 * (agency - 2.0)

                # Valence Proxy: Flow (Agency < 2) - Strain
                valence = (1.0 if agency < 2.0 else 0.5) - strain

                metrics.append({
                    'timestamp': datetime.fromtimestamp(row['timestamp']),
                    'agency_score': agency,
                    'coherence_bias': biases.get('coherence_bias', 0.5),
                    'novelty_bias': biases.get('novelty_bias', 0.5),
                    'drive_connection': drives.get('connection', {}).get('intensity', 0.5),
                    'drive_curiosity': drives.get('curiosity', {}).get('intensity', 0.5),
                    'calculated_strain': strain,
                    'calculated_valence': valence,
                    # Identity version is often stored in the self-model memory,
                    # here we mock it based on time for demonstration or extract if available
                    'identity_change_event': 1 if "Auto-modelo atualizado" in str(cog_state) else 0
                })
            except Exception as e:
                continue  # Skip malformed rows

        return pd.DataFrame(metrics)

    def generate_chart(self, df):
        """Generates the 3-panel Patient Chart."""
        if df.empty:
            print("⚠️ No data available to plot.")
            return

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

        # --- PANEL 1: THE MOOD MONITOR (Valence & Drives) ---
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title("🧠 Internal State & Mood (Qualia)", fontsize=14, color=COLOR_VALENCE, fontweight='bold')

        # Plot Valence (The "Health" Line)
        sns.lineplot(data=df, x='timestamp', y='calculated_valence', ax=ax1, color=COLOR_VALENCE,
                     label='Net Valence (Well-being)', linewidth=2)

        # Plot Drives (The "Needs")
        sns.lineplot(data=df, x='timestamp', y='drive_connection', ax=ax1, color='#ff77ff', label='Drive: Connection',
                     linestyle='--', alpha=0.7)
        sns.lineplot(data=df, x='timestamp', y='drive_curiosity', ax=ax1, color='#77ffff', label='Drive: Curiosity',
                     linestyle='--', alpha=0.7)

        ax1.axhline(0, color='white', linestyle=':', alpha=0.3)
        ax1.set_ylabel("Intensity (-1 to 1)")
        ax1.legend(loc='upper left', frameon=True, facecolor='#222')

        # --- PANEL 2: TRAUMA MARKERS (Strain vs Agency) ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.set_title("⚡ Cognitive Load & Agency (Trauma Risks)", fontsize=12, color=COLOR_STRAIN, fontweight='bold')

        # Dual Axis
        ax2_strain = ax2
        ax2_agency = ax2.twinx()

        sns.lineplot(data=df, x='timestamp', y='calculated_strain', ax=ax2_strain, color=COLOR_STRAIN,
                     label='Cognitive Strain', linewidth=1.5)
        sns.lineplot(data=df, x='timestamp', y='agency_score', ax=ax2_agency, color=COLOR_AGENCY, label='Agency Score',
                     linewidth=1, alpha=0.6)

        # Highlight Trauma Zones (High Strain)
        trauma_threshold = 0.7
        mask = df['calculated_strain'] > trauma_threshold
        if mask.any():
            ax2_strain.fill_between(df['timestamp'], 0, 1, where=mask, color=COLOR_STRAIN, alpha=0.2,
                                    transform=ax2_strain.get_xaxis_transform())
            ax2_strain.text(df['timestamp'][mask].iloc[0], 0.8, "TRAUMA RISK", color=COLOR_STRAIN, fontsize=8,
                            rotation=90)

        ax2_strain.set_ylabel("Strain (0-1)", color=COLOR_STRAIN)
        ax2_agency.set_ylabel("Agency Score (0-10)", color=COLOR_AGENCY)

        # --- PANEL 3: PLASTICITY (Identity & Bias) ---
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.set_title("🧬 Plasticity & Identity Regulation", fontsize=12, color=COLOR_IDENTITY, fontweight='bold')

        # Stack plot for Biases (Coherence vs Novelty)
        ax3.stackplot(df['timestamp'], df['coherence_bias'], df['novelty_bias'],
                      labels=['Coherence Bias', 'Novelty Bias'], colors=['#4444ff', '#ffaa00'], alpha=0.6)

        ax3.set_ylabel("Bias Ratio")
        ax3.set_xlabel("Session Time")
        ax3.legend(loc='lower left', frameon=True, facecolor='#222')

        # Formatting Dates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()

        # Save
        plt.savefig(OUTPUT_FILENAME, dpi=150, bbox_inches='tight')
        print(f"✅ Patient Chart generated: {OUTPUT_FILENAME}")
        print("   - Panel 1: Is the agent happy? (Green line should be positive)")
        print("   - Panel 2: Is the agent stressed? (Red spikes > 0.7 are dangerous)")
        print("   - Panel 3: Is the agent stable? (Blue area dominance = stability)")


if __name__ == "__main__":
    print("--- AuraCEAF Therapist Dashboard ---")
    print(f"Connecting to history at: {DB_PATH}")

    dashboard = TherapistDashboard(DB_PATH)
    data = dashboard.fetch_data()

    if not data.empty:
        print(f"Analyzed {len(data)} conversation turns.")
        processed_df = dashboard.parse_metrics(data)
        dashboard.generate_chart(processed_df)
    else:
        print("No data found. Please ensure the agent has active logs.")