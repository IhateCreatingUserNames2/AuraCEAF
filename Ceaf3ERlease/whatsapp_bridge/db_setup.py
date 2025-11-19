# whatsapp_bridge/database.py
import sys
from pathlib import Path

# --- INÍCIO DA CORREÇÃO DE IMPORTAÇÃO ---
# Esta parte "ensina" o Python a encontrar a pasta 'Ceaf'
# que está no mesmo nível que a pasta 'whatsapp_bridge'
ceaf_project_path = str(Path(__file__).resolve().parent.parent / 'Ceaf')
if ceaf_project_path not in sys.path:
    sys.path.insert(0, ceaf_project_path)
    print(f"Adicionado ao sys.path: {ceaf_project_path}") # Log para confirmar
# --- FIM DA CORREÇÃO DE IMPORTAÇÃO ---

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# IMPORTANTE: Agora esta importação funcionará porque o caminho foi adicionado acima
from database.models import Base, WhatsAppUser

# APONTA PARA O BANCO DE DADOS PRINCIPAL, que está dentro da pasta Ceaf
DATABASE_URL = "sqlite:////home/ubuntu/Ceaf/aura_agents_v3.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Garante que a tabela whatsapp_users (e outras, se necessário) seja criada
# no banco de dados principal.
Base.metadata.create_all(bind=engine)