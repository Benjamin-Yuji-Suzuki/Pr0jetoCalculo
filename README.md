
# 🛡️ Sistema de Apoio à Decisão: Otimização Logística

> **Projeto Bimestral - Resolução Diferencial de Problemas**  
> *Ciência da Computação - CESUPA*

---

## 📌 Sobre o Projeto

Este sistema é uma aplicação **Full Stack** desenvolvida para solucionar problemas reais de otimização de estoque em cadeias de suprimentos.

Baseado no modelo matemático de **Alkahtani-Davizón**, o sistema calcula o **Lote Econômico de Produção (EPQ)** ideal considerando:

- 🏭 Custos de setup (Fabricante e Fornecedor)  
- 📦 Custos de manutenção de estoque (Holding Cost)  
- 📉 Taxas de defeito e imperfeições no processo  
- 📊 Demanda estocástica  

A ferramenta utiliza **Cálculo Diferencial** (via biblioteca `SymPy`) para derivar a função de custo em tempo real e encontrar o ponto de mínimo global, garantindo a decisão mais econômica para o gestor.

---

## 🚀 Funcionalidades

- **Cálculo Simbólico Automático**: Derivação da função de custo total \(TC(Q)\) em tempo real.  
- **Validação de Convexidade**: Teste automático da segunda derivada \(\frac{d^2TC}{dQ^2} > 0\).  
- **Persistência de Dados**: Histórico completo de simulações salvo em banco de dados **PostgreSQL**.  
- **Upload de Dados Reais**: Suporte a arquivos CSV para análise de demanda histórica.  
- **Dashboard Interativo**: Interface amigável para ajuste de parâmetros de sensibilidade.  

---

## 🛠️ Tecnologias Utilizadas

| Componente   | Tecnologia   | Função |
|--------------|--------------|--------|
| **Linguagem** | Python 3.11 | Lógica principal e orquestração |
| **Matemática** | SymPy | Cálculo diferencial simbólico e resolução de equações |
| **Frontend** | Streamlit | Interface do usuário e visualização de dados |
| **Backend/DB** | PostgreSQL | Armazenamento persistente do histórico de decisões |
| **Análise** | Pandas/NumPy | Manipulação de datasets e estatística descritiva |

---

## ⚙️ Pré-requisitos e Instalação

### 1. Banco de Dados (PostgreSQL)

Certifique-se de ter o **PostgreSQL** e o **pgAdmin 4** instalados.

1. Abra o pgAdmin 4.  
2. Crie um novo banco de dados chamado `estoque_opt`.  
3. A senha configurada no projeto é `1234` (se a sua for diferente, altere no arquivo `main.py`).  

---

### 2. Instalação do Python e Dependências

```bash
# Clone este repositório
git clone https://github.com/seu-usuario/seu-projeto.git

# Entre na pasta
cd seu-projeto

# Instale as bibliotecas necessárias
pip install streamlit pandas sympy numpy psycopg2-binary
```

---

### 3. Execução

Para rodar o sistema, utilize o comando:

```bash
python -m streamlit run main.py
```

O navegador abrirá automaticamente em:  
👉 [http://localhost:8501](http://localhost:8501)

---

## 📚 Como Reproduzir os Experimentos (SymPy)

A seção de **Justificativa Matemática** no aplicativo demonstra como o SymPy é utilizado. Para reproduzir manualmente:

1. **Importação**: O sistema define símbolos (`Q`, `D`, `S`, etc.) usando `sp.symbols`.  
2. **Modelagem**: A função de custo total `TC` é construída como uma expressão Python.  
3. **Derivação**: Utilizamos `sp.diff(TC, Q)` para obter a derivada primeira.  
4. **Resolução**: `sp.solve(dTC, Q)` encontra as raízes da equação (pontos críticos).  

Essa abordagem garante **precisão absoluta**, evitando erros de arredondamento comuns em métodos numéricos tradicionais.

---

## 📂 Importação de Dados

O sistema aceita arquivos `.csv` para cálculo da demanda média.  
O arquivo deve conter uma coluna chamada **Demanda** ou **Sales Quantity**.

### Exemplo de CSV compatível:

```csv
Date,Sales Quantity,Price
2024-01-01,330,24.38
2024-01-02,334,74.98
...
```

---

## 👥 Autores

Projeto desenvolvido para a disciplina de **Resolução Diferencial de Problemas**.

- Benjamin Yuji Suzuki 👍  
- Felipe de Freitas 😶‍🌫️  
- Alberto Costa ✍️  

> *"A otimização não é apenas sobre cortar custos, é sobre encontrar o equilíbrio perfeito."* ⚖️
