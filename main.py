import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sympy as sp
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página para usar largura total (melhora os gráficos)
st.set_page_config(layout="wide", page_title="Otimização Alkahtani-Davizón")

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
    # Tenta converter data, se falhar tenta inferir
    if "Date" in df_local.columns:
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

# Se o banco estiver vazio, interrompe aqui para não dar erro
if df.empty:
    st.warning("⚠️ A tabela 'demand' ainda não está carregada no banco. Faça o upload do CSV na barra lateral.")
    st.stop()

# Garantir que temos as colunas certas
if "Sales Quantity" in df.columns:
    df["Daily_Demand"] = df["Sales Quantity"]
else:
    st.error("O CSV precisa ter uma coluna 'Sales Quantity'.")
    st.stop()

# -------------------------------------------------------------
# 4. REGRESSÃO LINEAR (PREVISÃO)
# -------------------------------------------------------------
st.title("📊 Sistema de Otimização Alkahtani–Davizón")
st.markdown("---")

# Prepara colunas para o modelo (adapte conforme seu CSV real)
# Aqui assumimos que essas colunas existem. Se não existirem, criamos dummies ou avisamos.
cols_needed = ["Store ID", "Promotions", "Seasonality Factors", "External Factors", "Customer Segments", "Price"]
available_cols = [c for c in cols_needed if c in df.columns]

if not available_cols:
    st.warning("Colunas para Machine Learning não encontradas. Usando média simples.")
    df["Predicted_Demand"] = df["Sales Quantity"].mean()
else:
    # Separa categóricas e numéricas das disponíveis
    cat_cols = [c for c in available_cols if df[c].dtype == 'object']
    num_cols = [c for c in available_cols if df[c].dtype != 'object']

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    X = df[available_cols]
    y = df["Sales Quantity"]
    
    try:
        model.fit(X, y)
        df["Predicted_Demand"] = model.predict(X)
    except Exception as e:
        st.warning(f"Erro ao treinar modelo: {e}. Usando média.")
        df["Predicted_Demand"] = y.mean()

D_estimated = df["Predicted_Demand"].mean() * 365

# -------------------------------------------------------------
# 5. VISUALIZAÇÃO 1: SÉRIE TEMPORAL (Novo!)
# -------------------------------------------------------------
st.subheader("1. Análise de Demanda (Real vs Machine Learning)")
col_kpi1, col_kpi2 = st.columns(2)
col_kpi1.metric("Demanda Diária Média", f"{df['Sales Quantity'].mean():.2f} un")
col_kpi2.metric("Demanda Anual Projetada (D)", f"{D_estimated:,.2f} un")

# Gráfico com Seaborn/Matplotlib
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=df, x='Date', y='Sales Quantity', label='Vendas Reais', alpha=0.5, ax=ax1)
    sns.lineplot(data=df, x='Date', y='Predicted_Demand', label='Tendência (Regressão)', color='red', linestyle='--', ax=ax1)
    ax1.set_title("Histórico de Vendas e Tendência")
    ax1.set_ylabel("Quantidade")
    ax1.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig1)
else:
    st.info("A coluna 'Date' não foi encontrada para plotar o gráfico temporal.")

st.markdown("---")

# -------------------------------------------------------------
# 6. FUNÇÃO DE OTIMIZAÇÃO (CÁLCULO)
# -------------------------------------------------------------
def eoq_with_derivative(S, h, D):
    Q = sp.Symbol('Q', positive=True)
    # Função Objetivo: Custo Total = Setup + Holding
    CT = S*D/Q + h*Q/2

    dCT = sp.diff(CT, Q)
    d2CT = sp.diff(dCT, Q)
    
    # Resolve dCT/dQ = 0
    sol = sp.solve(dCT, Q)
    if sol:
        Q_opt = float(sol[0])
        return Q_opt, float(dCT.subs(Q, Q_opt)), float(d2CT.subs(Q, Q_opt)), CT
    else:
        return 0.0, 0.0, 0.0, CT

def alkahtani_davizon_optimization(Sm, Sv, hm, hv, alpha_m, alpha_v, D):
    if hm <= 0 or hv <= 0:
        return None
    
    # Cálculos para Metal
    hm_adj = hm * (1 - alpha_m) # Ajuste por defeito (conforme paper/fórmula)
    QM, d1m, d2m, expr_m = eoq_with_derivative(Sm, hm_adj, D)
    
    # Cálculos para Vidro
    hv_adj = hv * (1 - alpha_v)
    QV, d1v, d2v, expr_v = eoq_with_derivative(Sv, hv_adj, D)

    # Custo Total Somado
    CT_val = (Sm*D/QM + hm_adj*QM/2) + (Sv*D/QV + hv_adj*QV/2)

    return {
        "QM": QM, "QV": QV, "Custo Total": CT_val,
        "d1m": d1m, "d2m": d2m, "expr_m": expr_m,
        "d1v": d1v, "d2v": d2v, "expr_v": expr_v,
        "hm_adj": hm_adj, "hv_adj": hv_adj # Retornamos para usar no gráfico
    }

# -------------------------------------------------------------
# 7. INTERFACE E GRÁFICOS DE OTIMIZAÇÃO
# -------------------------------------------------------------
st.subheader("2. Otimização de Custos (Cálculo Diferencial)")

c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Parâmetros Metal")
    Sm = st.number_input("Setup ($)", 200.0)
    hm = st.number_input("Holding ($/un)", 2.0)
    alpha_m = st.slider("Defeito Metal (%)", 0, 20, 5)/100
with c2:
    st.markdown("##### Parâmetros Vidro")
    Sv = st.number_input("Setup ($)", 180.0)
    hv = st.number_input("Holding ($/un)", 1.8)
    alpha_v = st.slider("Defeito Vidro (%)", 0, 20, 4)/100

if st.button("🚀 Calcular Otimização"):
    res = alkahtani_davizon_optimization(Sm, Sv, hm, hv, alpha_m, alpha_v, D_estimated)
    
    if res:
        # --- EXIBIÇÃO DE RESULTADOS NUMÉRICOS ---
        st.success("Otimização concluída com sucesso!")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Lote Ótimo Metal (Q*)", f"{int(res['QM'])}")
        col_res2.metric("Lote Ótimo Vidro (Q*)", f"{int(res['QV'])}")
        col_res3.metric("Custo Total Anual", f"R$ {res['Custo Total']:,.2f}")
        
        with st.expander("Ver Detalhes Matemáticos (Derivadas)"):
            st.latex(r"TC(Q) = \frac{S \cdot D}{Q} + \frac{h \cdot Q}{2}")
            st.write(f"**Metal:** 1ª Derivada no ponto ótimo: {res['d1m']:.4f} (aprox. 0)")
            st.write(f"**Metal:** 2ª Derivada: {res['d2m']:.6f} (> 0, logo é Mínimo)")
        
        # --- VISUALIZAÇÃO 2: CURVA DE CUSTO (Novo!) ---
        st.subheader("3. Curva de Custo Total (Prova de Convexidade)")
        st.caption("O gráfico abaixo mostra como o Custo Total varia conforme o tamanho do lote. O ponto vermelho indica o ótimo encontrado pela derivada.")

        # Função auxiliar para gerar pontos do gráfico
        def get_curve_points(S, h_adj, D, Q_opt):
            # Cria um intervalo de 50% a 200% do Q ótimo
            Q_range = np.linspace(Q_opt * 0.5, Q_opt * 2.0, 100)
            Costs = (S * D / Q_range) + (h_adj * Q_range / 2)
            return Q_range, Costs

        # Gerar dados
        Qm_x, Cm_y = get_curve_points(Sm, res['hm_adj'], D_estimated, res['QM'])
        Qv_x, Cv_y = get_curve_points(Sv, res['hv_adj'], D_estimated, res['QV'])

        # Plotar
        fig2, (ax_m, ax_v) = plt.subplots(1, 2, figsize=(14, 5))

        # Gráfico Metal
        sns.lineplot(x=Qm_x, y=Cm_y, ax=ax_m, color='blue')
        ax_m.scatter([res['QM']], [Cm_y.min()], color='red', s=100, zorder=5, label='Ponto Mínimo (Derivada=0)')
        ax_m.set_title(f"Curva de Custo: Metal (Q* = {int(res['QM'])})")
        ax_m.set_xlabel("Tamanho do Lote (Q)")
        ax_m.set_ylabel("Custo Total ($)")
        ax_m.legend()
        ax_m.grid(True, alpha=0.3)

        # Gráfico Vidro
        sns.lineplot(x=Qv_x, y=Cv_y, ax=ax_v, color='green')
        ax_v.scatter([res['QV']], [Cv_y.min()], color='red', s=100, zorder=5, label='Ponto Mínimo (Derivada=0)')
        ax_v.set_title(f"Curva de Custo: Vidro (Q* = {int(res['QV'])})")
        ax_v.set_xlabel("Tamanho do Lote (Q)")
        ax_v.legend()
        ax_v.grid(True, alpha=0.3)

        st.pyplot(fig2)
        
    else:
        st.error("Erro nos parâmetros (Holding cost deve ser > 0)")
