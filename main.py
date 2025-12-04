import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import psycopg2
from sqlalchemy import create_engine

# -------------------------------------------------------------
# 1. CONFIGURAÇÃO DO POSTGRES (inputs no Streamlit)
# -------------------------------------------------------------
st.sidebar.header("Configuração do PostgreSQL")

pg_host = st.sidebar.text_input("Host", "localhost")
pg_port = st.sidebar.text_input("Porta", "5432")
pg_db   = st.sidebar.text_input("Database", "meubanco")
pg_user = st.sidebar.text_input("Usuário", "postgres")
pg_pass = st.sidebar.text_input("Senha", "1234", type="password")

# Criar URL do SQLAlchemy
pg_url = f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

# Criar engine
try:
    engine = create_engine(pg_url)
except:
    st.error("Erro ao conectar ao PostgreSQL. Verifique host, porta e credenciais.")


# -------------------------------------------------------------
# 2. CARREGAR CSV NO POSTGRES
# -------------------------------------------------------------
@st.cache_data
def load_csv_to_postgres():
    df = pd.read_csv("/mnt/data/demand_forecasting.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    df.to_sql("demand", engine, if_exists="replace", index=False)
    return True

if st.sidebar.button("Carregar CSV para o PostgreSQL"):
    load_csv_to_postgres()
    st.sidebar.success("CSV carregado com sucesso para o PostgreSQL!")


# -------------------------------------------------------------
# 3. CONSULTAR O BANCO
# -------------------------------------------------------------
def query_postgres(query):
    try:
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"Erro na consulta SQL: {e}")
        return pd.DataFrame()

# Obter dados
df = query_postgres("SELECT * FROM demand")

if df.empty:
    st.warning("A tabela 'demand' ainda não está carregada.")
    st.stop()

df["Daily_Demand"] = df["Sales Quantity"]


# -------------------------------------------------------------
# 4. REGRESSÃO LINEAR PARA ESTIMAR DEMANDA
# -------------------------------------------------------------
categorical_cols = ["Store ID", "Promotions", "Seasonality Factors",
                    "External Factors", "Customer Segments"]
numeric_cols = ["Price"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X = df[categorical_cols + numeric_cols]
y = df["Sales Quantity"]
model.fit(X, y)

df["Predicted_Demand"] = model.predict(X)
D_estimated = df["Predicted_Demand"].mean() * 365


# -------------------------------------------------------------
# 5. FUNÇÃO DE OTIMIZAÇÃO (ADAPTADA)
# -------------------------------------------------------------
def alkahtani_davizon_optimization(Sm, Sv, hm, hv, alpha_m, alpha_v, D):

    if hm == 0 or hv == 0:
        return {
            "QM": 0,
            "QV": 0,
            "Custo Total": np.inf,
            "Mensagem": "Erro: custos de armazenagem não podem ser zero."
        }

    QM = np.sqrt((2 * Sm * D) / (hm * (1 - alpha_m)))
    QV = np.sqrt((2 * Sv * D) / (hv * (1 - alpha_v)))

    CT = (
        (Sm * D / QM) + (hm * QM / 2) +
        (Sv * D / QV) + (hv * QV / 2)
    )

    return {
        "QM": QM,
        "QV": QV,
        "Custo Total": CT,
        "Mensagem": "Cálculo realizado com sucesso."
    }


# -------------------------------------------------------------
# 6. INTERFACE DO STREAMLIT
# -------------------------------------------------------------
st.title("Otimização Alkahtani–Davizón com PostgreSQL + Regressão Linear")

st.subheader("📦 Demanda anual estimada via Regressão Linear")
st.write(f"**Demanda anual prevista (D): {D_estimated:.2f} unidades**")

st.header("Parâmetros do Modelo")

col1, col2 = st.columns(2)

with col1:
    Sm = st.number_input("Custo de Setup (Metal)", value=200.0)
    hm = st.number_input("Custo de Armazenagem (Metal)", value=2.0)
    alpha_m = st.number_input("Taxa de Defeito Metal (0–1)", value=0.05)

with col2:
    Sv = st.number_input("Custo de Setup (Vidro)", value=180.0)
    hv = st.number_input("Custo de Armazenagem (Vidro)", value=1.8)
    alpha_v = st.number_input("Taxa de Defeito Vidro (0–1)", value=0.04)

if st.button("Calcular Otimização"):
    result = alkahtani_davizon_optimization(Sm, Sv, hm, hv, alpha_m, alpha_v, D_estimated)

    st.success(result["Mensagem"])

    st.write("### Resultados")
    st.write(f"**Lote ótimo Metal (QM): {result['QM']:.2f} unidades**")
    st.write(f"**Lote ótimo Vidro (QV): {result['QV']:.2f} unidades**")
    st.write(f"**Custo Total Anual: R$ {result['Custo Total']:.2f}**")
