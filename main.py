import streamlit as st
import pandas as pd
import sympy as sp
import numpy as np
import psycopg2 # Conector do PostgreSQL
from psycopg2 import sql
from datetime import datetime
import warnings

# --- Configuração Inicial da Página ---
st.set_page_config(
    page_title="Sistema de Otimização Alkahtani-Davizón",
    layout="wide",
    page_icon="🛡️"
)

# --- CONFIGURAÇÃO DO BANCO DE DADOS (POSTGRESQL) ---
# Edite aqui com as suas credenciais do pgAdmin 4
DB_CONFIG = {
    "dbname": "estoque_opt",
    "user": "postgres",      # Usuário padrão costuma ser 'postgres'
    "password": "1234",     # <--- COLOQUE SUA SENHA DO PGADMIN AQUI
    "host": "localhost",
    "port": "5432"
}

# --- Camada de Persistência (PostgreSQL) ---
def get_db_connection():
    """Cria e retorna uma conexão com o banco PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"Erro ao conectar no PostgreSQL: {e}")
        return None

def init_db():
    """Inicializa a tabela no PostgreSQL se não existir."""
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        # Nota: Em Postgres usa-se SERIAL para auto-incremento
        cur.execute('''
            CREATE TABLE IF NOT EXISTS simulacoes (
                id SERIAL PRIMARY KEY,
                data_hora TIMESTAMP,
                demanda_anual REAL,
                lote_otimo REAL,
                custo_minimo REAL,
                setup_total REAL,
                holding_medio REAL
            );
        ''')
        conn.commit()
        cur.close()
        conn.close()

def salvar_simulacao(D, Q_star, Min_Cost, S_m, S_v, h_m, h_v):
    """Salva os resultados da simulação no banco."""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            data_hora = datetime.now()
            
            # --- CORREÇÃO DO ERRO (NumPy -> Python Float) ---
            # O PostgreSQL não aceita tipos numpy (np.float64).
            # Convertemos explicitamente para float nativo do Python.
            val_D = float(D)
            val_Q = float(Q_star)
            val_Min = float(Min_Cost)
            val_Setup = float(S_m + S_v)
            val_Holding = float((h_m + h_v) / 2)
            
            cur.execute('''
                INSERT INTO simulacoes (data_hora, demanda_anual, lote_otimo, custo_minimo, setup_total, holding_medio)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (data_hora, val_D, val_Q, val_Min, val_Setup, val_Holding))
            
            conn.commit()
            cur.close()
        except Exception as e:
            st.error(f"Erro ao salvar no banco: {e}")
        finally:
            conn.close()

def carregar_historico():
    """Carrega o histórico de simulações."""
    conn = get_db_connection()
    df = pd.DataFrame()
    if conn:
        try:
            # Suprime o aviso de que pandas prefere SQLAlchemy
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_sql_query("SELECT * FROM simulacoes ORDER BY id DESC", conn)
        except Exception as e:
            st.error(f"Erro ao ler histórico: {e}")
        finally:
            conn.close()
    return df

# Inicializa a tabela na primeira execução
# (Colocamos num try/except para não quebrar o app se o banco não estiver rodando)
try:
    init_db()
except Exception as e:
    st.error(f"⚠️ Aviso: Não foi possível inicializar o banco de dados. Verifique se o PostgreSQL está rodando e se o banco 'estoque_opt' foi criado. Erro: {e}")

# --- Função de Otimização (Lógica Matemática com SymPy) ---
def alkahtani_optimization(D, S_m, S_v, h_m, h_v, alpha_m, alpha_v):
    """
    Calcula o Q* (Lote Econômico) usando SymPy para derivar a função de custo total.
    Retorna também a fórmula simbólica para exibição.
    """
    # Definindo a variável simbólica
    Q = sp.symbols('Q', real=True, positive=True)
    
    # Parâmetros simbólicos para exibição didática
    D_sym, Sm_sym, Sv_sym, hm_sym, hv_sym = sp.symbols('D S_m S_v h_m h_v')
    
    # Prevenção de erro de divisão por zero
    if Q == 0:
        return 0, 0, False, ""

    # Custos de Manutenção (Holding Cost) ajustados por defeitos
    H_total = (Q / 2) * (h_m * (1 + alpha_m) + h_v * (1 + alpha_v))
    
    # Custos de Setup (Fabricante + Fornecedor)
    S_total = (S_m + S_v) * D / Q
    
    # Custo Total (Função Objetivo)
    TC = S_total + H_total
    
    # 1. Primeira Derivada (dTC/dQ)
    dTC = sp.diff(TC, Q)
    
    # 2. Segunda Derivada (para provar convexidade)
    d2TC = sp.diff(TC, Q, 2)
    
    # 3. Resolução da equação dTC/dQ = 0
    sol = sp.solve(dTC, Q)
    
    if not sol:
        return 0, 0, False, ""
        
    q_opt = float(sol[0]) # Pega a primeira raiz positiva
    
    # Calcula valor do custo mínimo e verifica convexidade
    min_cost = float(TC.subs(Q, q_opt))
    is_convex = d2TC.subs(Q, q_opt) > 0
    
    # Gera a fórmula LaTeX da derivada para explicação
    latex_derivative = sp.latex(dTC)
    
    return q_opt, min_cost, is_convex, latex_derivative

# --- Interface do Usuário ---

st.title("🛡️ Sistema de Apoio à Decisão: Otimização Logística")
st.caption("Projeto Bimestral - Resolução Diferencial de Problemas")

# Abas para organizar o sistema conforme rubrica (Simulação vs Histórico)
tab1, tab2, tab3 = st.tabs(["📊 Painel de Otimização", "📚 Justificativa Matemática", "🗄️ Histórico de Decisões"])

with tab1:
    st.markdown("### Parâmetros da Operação")
    
    col_config, col_main = st.columns([1, 2])

    with col_config:
        st.subheader("Custos & Restrições")
        S_m = st.number_input("Custo Setup Fabricante ($)", value=500.0, step=50.0, help="Custo fixo por lote produzido")
        S_v = st.number_input("Custo Setup Fornecedor ($)", value=200.0, step=50.0, help="Custo fixo de pedido")
        h_m = st.number_input("Holding Cost Fabricante ($/unid)", value=2.0, step=0.1)
        h_v = st.number_input("Holding Cost Fornecedor ($/unid)", value=1.5, step=0.1)
        
        st.write("---")
        alpha_m = st.slider("Taxa de Defeito Interna (%)", 0.0, 10.0, 2.0) / 100
        alpha_v = st.slider("Taxa de Defeito Fornecedor (%)", 0.0, 10.0, 5.0) / 100

    with col_main:
        st.subheader("Demanda (Input)")
        
        # Opção para usar dados de exemplo se o usuário não tiver CSV
        use_sample = st.checkbox("Usar dados simulados (Demo)", value=True)
        
        df = None
        D_annual = 0
        sigma_demand = 0
        
        if use_sample:
            dates = pd.date_range(start='2023-01-01', periods=365)
            values = np.random.normal(loc=100, scale=15, size=365)
            values = [max(0, x) for x in values]
            df = pd.DataFrame({'Data': dates, 'Demanda': values})
            
            # Gráfico pequeno para não poluir
            st.area_chart(df.set_index('Data'), height=150)
            
            daily_avg = df['Demanda'].mean()
            D_annual = daily_avg * 365
            sigma_demand = df['Demanda'].std()
            
        else:
            uploaded_file = st.file_uploader("Carregar CSV Real", type="csv")
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'Demanda' in df.columns:
                        daily_avg = df['Demanda'].mean()
                        D_annual = daily_avg * 365
                        sigma_demand = df['Demanda'].std()
                    else:
                        st.error("CSV deve ter coluna 'Demanda'")
                except Exception as e:
                    st.error(f"Erro: {e}")

        if D_annual > 0:
            c1, c2 = st.columns(2)
            c1.metric("Demanda Anual Estimada", f"{int(D_annual):,}")
            c2.metric("Volatilidade (Risco)", f"{sigma_demand:.2f}")

            st.write("---")
            if st.button("🚀 Calcular Solução Ótima", type="primary"):
                
                with st.spinner("O SymPy está derivando a função de custo..."):
                    q_star, min_cost, convex, diff_eq = alkahtani_optimization(
                        D_annual, S_m, S_v, h_m, h_v, alpha_m, alpha_v
                    )
                
                # Salvar no Banco de Dados (Persistência)
                salvar_simulacao(D_annual, q_star, min_cost, S_m, S_v, h_m, h_v)
                st.toast("Simulação salva no histórico PostgreSQL!", icon="🐘")
                
                # Exibição dos Resultados
                st.success("##### Recomendação do Sistema")
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Lote Econômico (Q*)", f"{int(q_star)} un.")
                col_res2.metric("Custo Total Mínimo", f"${min_cost:,.2f}")
                col_res3.metric("Status da Solução", "Ótima Global" if convex else "Inconclusiva")
                
                st.info(f"""
                **Interpretação:** Para minimizar os custos totais da cadeia, o pedido de produção deve ser de aproximadamente **{int(q_star)} unidades**. 
                Valores menores aumentam excessivamente os custos de setup, e valores maiores aumentam os custos de estoque e risco de defeitos.
                """)
                
                # Guardar a equação para a aba de explicação
                st.session_state['last_diff_eq'] = diff_eq

with tab2:
    st.header("Como a solução foi escolhida?")
    st.markdown("""
    O modelo utiliza o cálculo diferencial para encontrar o ponto exato onde a curva de custo total muda de direção (ponto de mínimo).
    
    **1. Função Objetivo (Custo Total):**
    $$ TC(Q) = \\frac{D}{Q}(S_m + S_v) + \\frac{Q}{2}[h_m(1 + \\alpha_m) + h_v(1 + \\alpha_v)] $$
    
    **2. Derivada Primeira (Custo Marginal):**
    Para encontrar o mínimo, igualamos a derivada a zero:
    """)
    
    if 'last_diff_eq' in st.session_state:
        st.latex(f"\\frac{{dTC}}{{dQ}} = {st.session_state['last_diff_eq']} = 0")
    else:
        st.info("Rode uma simulação para ver a derivada calculada pelo SymPy.")
        
    st.markdown("""
    **3. Validação:**
    O sistema verifica automaticamente a **segunda derivada**. Se $TC''(Q) > 0$, confirmamos que a solução é um mínimo global (convexidade).
    """)

with tab3:
    st.header("🗄️ Histórico de Simulações (PostgreSQL)")
    st.markdown("Registro persistente vindo do banco de dados `estoque_opt`.")
    
    df_hist = carregar_historico()
    if not df_hist.empty:
        st.dataframe(df_hist, use_container_width=True)
        
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar Relatório (CSV)",
            data=csv,
            file_name='historico_otimizacao.csv',
            mime='text/csv',
        )
    else:
        st.info("Nenhuma simulação realizada ainda ou não foi possível conectar ao banco.")