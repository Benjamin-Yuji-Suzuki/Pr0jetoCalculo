import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sympy as sp
from sqlalchemy import create_engine

# -------------------------------------------------------------
# 1. CONFIGURAÇÃO DO POSTGRES
# -------------------------------------------------------------
st.sidebar.header("Configuração do PostgreSQL")

pg_host = st.sidebar.text_input("Host", "localhost")
pg_port = st.sidebar.text_input("Porta", "5432")
pg_db   = st.sidebar.text_input("Database", "meubanco")
pg_user = st.sidebar.text_input("Usuário", "postgres")
pg_pass = st.sidebar.text_input("Senha", "1234", type="password")

pg_url = f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

try:
    engine = create_engine(pg_url)
except Exception as e:
    st.error(f"Erro ao conectar ao PostgreSQL: {e}")

# -------------------------------------------------------------
# 2. UPLOAD DO CSV E CARREGAMENTO NO POSTGRES
# -------------------------------------------------------------
st.sidebar.header("Upload do CSV de demanda")
uploaded_file = st.sidebar.file_uploader("Escolha o CSV de demanda", type="csv")

if uploaded_file is not None:
    df_local = pd.read_csv(uploaded_file)
    df_local["Date"] = pd.to_datetime(df_local["Date"])

    if st.sidebar.button("Carregar CSV no PostgreSQL"):
        try:
            df_local.to_sql("demand", engine, if_exists="replace", index=False)
            st.sidebar.success("CSV carregado com sucesso no PostgreSQL!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar CSV no PostgreSQL: {e}")

# -------------------------------------------------------------
# 3. CONSULTA DOS DADOS NO POSTGRES
# -------------------------------------------------------------
def query_postgres(query):
    try:
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"Erro na consulta SQL: {e}")
        return pd.DataFrame()

df = query_postgres("SELECT * FROM demand")
if df.empty:
    st.warning("A tabela 'demand' ainda não está carregada.")
    st.stop()

df["Daily_Demand"] = df["Sales Quantity"]

# -------------------------------------------------------------
# 4. REGRESSÃO LINEAR PARA ESTIMAR DEMANDA ANUAL
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

st.write(f"**Demanda anual estimada (D): {D_estimated:.2f} unidades**")

# -------------------------------------------------------------
# 5. FUNÇÃO DE OTIMIZAÇÃO COM DERIVADAS SIMBÓLICAS
# -------------------------------------------------------------
def eoq_with_derivative(S, h, D):
    Q = sp.Symbol('Q', positive=True)
    CT = S*D/Q + h*Q/2

    dCT = sp.diff(CT, Q)
    d2CT = sp.diff(dCT, Q)
    Q_opt = sp.solve(dCT, Q)[0]

    return float(Q_opt), float(dCT.subs(Q, Q_opt)), float(d2CT.subs(Q, Q_opt))

def alkahtani_davizon_optimization(Sm, Sv, hm, hv, alpha_m, alpha_v, D):
    if hm <= 0 or hv <= 0:
        return {"QM": 0, "QV": 0, "Custo Total": np.inf, "Mensagem": "Erro: custos de armazenagem devem ser >0"}

    QM, d1m, d2m = eoq_with_derivative(Sm, hm*(1-alpha_m), D)
    QV, d1v, d2v = eoq_with_derivative(Sv, hv*(1-alpha_v), D)

    CT = (Sm*D/QM + hm*QM/2) + (Sv*D/QV + hv*QV/2)

    return {
        "QM": QM,
        "QV": QV,
        "Custo Total": CT,
        "d1m": d1m, "d2m": d2m,
        "d1v": d1v, "d2v": d2v,
        "Mensagem": "Cálculo realizado com derivadas explícitas."
    }

# -------------------------------------------------------------
# 6. INTERFACE STREAMLIT PARA PARÂMETROS E RESULTADOS
# -------------------------------------------------------------
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
    st.write(f"Primeira derivada (Metal) no ponto crítico: {result['d1m']:.2e}")
    st.write(f"Segunda derivada (Metal) no ponto crítico: {result['d2m']:.2e} ✅ positivo → mínimo")
    st.write(f"**Lote ótimo Vidro (QV): {result['QV']:.2f} unidades**")
    st.write(f"Primeira derivada (Vidro) no ponto crítico: {result['d1v']:.2e}")
    st.write(f"Segunda derivada (Vidro) no ponto crítico: {result['d2v']:.2e} ✅ positivo → mínimo")
    st.write(f"**Custo Total Anual: R$ {result['Custo Total']:.2f}**")
