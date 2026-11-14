"""
Calculadora de Engajamento de TikToker - Streamlit + TensorFlow (arquivo único)

Como rodar localmente:
    pip install -r requirements.txt
    streamlit run app.py

Como funciona (resumo):
    1. Ao abrir, o app treina um modelo com uma base SIMULADA de vídeos.
    2. Na barra lateral, a equipe pode enviar arquivos CSV com dados reais
       (colunas: categoria, hashtags, alcance) e clicar em "Treinar modelo"
       para retreinar a rede neural (TensorFlow/Keras) com esses dados.
    3. O usuário escolhe categoria e hashtags em caixas de seleção e o app
       mostra o alcance previsto com um alerta: Flopado, Regular ou Viral.
"""

import io
import os

# Reduz mensagens de log do TensorFlow (precisa vir ANTES do import do tensorflow)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# 1) CONFIGURAÇÕES GERAIS
# ---------------------------------------------------------------------------
SEED = 42                # Semente fixa: resultados reprodutíveis
N_VIDEOS = 2000          # Tamanho da base simulada
EPOCAS = 60              # Quantas vezes o modelo "revê" os dados no treino
MAX_HASHTAGS = 15        # Máximo de hashtags da base simulada
LIMITE_HASHTAGS = 30     # Linhas com mais hashtags que isso são ignoradas no upload
MIN_LINHAS = 30          # Mínimo de linhas válidas para treinar com arquivos
COLUNAS = ["categoria", "hashtags", "alcance"]

LIMITE_FLOP = 10_000     # Abaixo disso: Flopado
LIMITE_VIRAL = 100_000   # Acima disso: Viral

# Alcance "típico" de cada categoria (valores fictícios, só para a simulação)
ALCANCE_BASE = {
    "Humor": 60_000,
    "Dança": 90_000,
    "Beleza": 30_000,
    "Games": 20_000,
    "Educação": 8_000,
    "Culinária": 40_000,
    "Fitness": 25_000,
}
CATEGORIAS_SIMULADAS = list(ALCANCE_BASE.keys())


# ---------------------------------------------------------------------------
# 2) DADOS: BASE SIMULADA E LEITURA DOS ARQUIVOS ENVIADOS
# ---------------------------------------------------------------------------
def gerar_base_simulada() -> pd.DataFrame:
    """
    Cria vídeos fictícios seguindo regras simples:
      - cada categoria tem um alcance base diferente;
      - o número ideal de hashtags fica entre 3 e 6 (pico em ~4,5);
      - acima de 10 hashtags o alcance cai (parece "spam" para o público);
      - um ruído aleatório deixa os dados mais realistas.
    """
    rng = np.random.default_rng(SEED)
    categorias = rng.choice(CATEGORIAS_SIMULADAS, size=N_VIDEOS)
    hashtags = rng.integers(0, MAX_HASHTAGS + 1, size=N_VIDEOS)

    base = np.array([ALCANCE_BASE[c] for c in categorias], dtype="float64")
    bonus_ideal = 0.7 * np.exp(-((hashtags - 4.5) ** 2) / (2 * 2.0**2))
    penalidade = 0.15 * np.maximum(0, hashtags - 10)
    ruido = rng.normal(0, 0.35, size=N_VIDEOS)

    # Trabalhamos em escala logarítmica: efeitos viram somas simples
    log_alcance = np.log(base) + bonus_ideal - penalidade + ruido

    return pd.DataFrame(
        {
            "categoria": categorias,
            "hashtags": hashtags,
            "alcance": np.exp(log_alcance).round().astype(int),
        }
    )


def ler_arquivos(arquivos) -> tuple:
    """
    Lê um ou mais CSVs enviados pelo usuário e devolve (tabela_limpa, linhas_removidas).

    Cada arquivo precisa ter as colunas: categoria, hashtags, alcance.
    Aceita separador vírgula ou ponto e vírgula, e codificação UTF-8 ou Latin-1.
    """
    partes = []
    for arq in arquivos:
        bruto = arq.getvalue()
        df = None
        for codificacao in ("utf-8-sig", "latin-1"):
            try:
                # sep=None: o pandas descobre sozinho se o separador é "," ou ";"
                df = pd.read_csv(
                    io.BytesIO(bruto), sep=None, engine="python", encoding=codificacao
                )
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError(f"Não consegui ler o arquivo '{arq.name}'.")

        df.columns = [str(c).strip().lower() for c in df.columns]
        faltando = [c for c in COLUNAS if c not in df.columns]
        if faltando:
            raise ValueError(
                f"O arquivo '{arq.name}' não tem a(s) coluna(s): {', '.join(faltando)}."
            )
        partes.append(df[COLUNAS])

    df = pd.concat(partes, ignore_index=True)
    total = len(df)

    # Limpeza: tira linhas vazias, textos em branco e números inválidos
    df = df.dropna(subset=["categoria"])
    df["categoria"] = df["categoria"].astype(str).str.strip()
    df["hashtags"] = pd.to_numeric(df["hashtags"], errors="coerce")
    df["alcance"] = pd.to_numeric(df["alcance"], errors="coerce")
    df = df.dropna()
    df = df[
        (df["categoria"] != "")
        & (df["alcance"] > 0)
        & (df["hashtags"] >= 0)
        & (df["hashtags"] <= LIMITE_HASHTAGS)
    ]
    df["hashtags"] = df["hashtags"].round().astype(int)
    df["alcance"] = df["alcance"].round().astype(int)
    df = df.reset_index(drop=True)

    if len(df) < MIN_LINHAS:
        raise ValueError(
            f"Foram encontradas só {len(df)} linhas válidas; "
            f"são necessárias pelo menos {MIN_LINHAS}."
        )
    if df["categoria"].nunique() < 2:
        raise ValueError("Os dados precisam ter pelo menos 2 categorias diferentes.")

    return df, total - len(df)


# ---------------------------------------------------------------------------
# 3) TREINO DO MODELO (TENSORFLOW)
# ---------------------------------------------------------------------------
def treinar_modelo(df: pd.DataFrame, origem: str) -> dict:
    """
    Treina a rede neural com a tabela recebida e devolve um "pacote" com o
    modelo e as informações necessárias para usá-lo.
    """
    keras.utils.set_random_seed(SEED)

    # As categorias vêm dos próprios dados (assim categorias novas funcionam)
    categorias = sorted(df["categoria"].unique().tolist())

    # --- Alvo em escala log e padronizado ---------------------------------
    # Alcances variam de milhares a milhões. Com o logaritmo e a
    # padronização (média 0, desvio 1) o modelo aprende de forma bem mais estável.
    y_log = np.log(df["alcance"].to_numpy(dtype="float64"))
    media_log = float(y_log.mean())
    desvio_log = float(y_log.std()) or 1.0
    y = ((y_log - media_log) / desvio_log).astype("float32")

    # --- Divisão 80% treino / 20% teste -----------------------------------
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(df))
    corte = int(len(df) * 0.8)
    treino, teste = indices[:corte], indices[corte:]

    # O Keras 3 não aceita arrays de texto do NumPy: convertemos a categoria
    # para tensor tf.string, no formato (n, 1). Hashtags viram float32 (n, 1).
    def para_tensores(idx):
        cats = tf.constant([str(c) for c in df["categoria"].to_numpy()[idx]])
        tags = df["hashtags"].to_numpy(dtype="float32")[idx]
        return {
            "categoria": tf.reshape(cats, (-1, 1)),
            "hashtags": tf.constant(tags.reshape(-1, 1)),
        }

    x_treino, x_teste = para_tensores(treino), para_tensores(teste)
    ds_treino = (
        tf.data.Dataset.from_tensor_slices((x_treino, y[treino]))
        .shuffle(len(treino), seed=SEED)
        .batch(32)
    )

    # --- Pré-processamento dentro do modelo -------------------------------
    # Categoria (texto) -> vetor one-hot. StringLookup faz o papel de
    # "StringLookup + CategoryEncoding" em uma única camada.
    entrada_cat = keras.Input(shape=(1,), dtype=tf.string, name="categoria")
    cat_codificada = keras.layers.StringLookup(
        vocabulary=categorias, output_mode="one_hot"
    )(entrada_cat)

    # Hashtags -> normalizadas (média 0, desvio 1) com a camada Normalization.
    # O adapt() aprende média e desvio usando SÓ os dados de treino.
    entrada_tags = keras.Input(shape=(1,), dtype=tf.float32, name="hashtags")
    normalizador = keras.layers.Normalization(axis=-1)
    normalizador.adapt(df["hashtags"].to_numpy(dtype="float32")[treino].reshape(-1, 1))
    tags_normalizadas = normalizador(entrada_tags)

    # --- Rede neural de regressão -----------------------------------------
    # As camadas ocultas permitem aprender que existe um número "ideal" de
    # hashtags (e não apenas "mais é melhor" ou "menos é melhor").
    x = keras.layers.Concatenate()([cat_codificada, tags_normalizadas])
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    saida = keras.layers.Dense(1)(x)  # sem ativação: prevê um número livre
    modelo = keras.Model([entrada_cat, entrada_tags], saida)

    modelo.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    modelo.fit(ds_treino, epochs=EPOCAS, verbose=0)

    # --- Erro no conjunto de teste ----------------------------------------
    # O erro (MAE) está em escala log padronizada; convertemos para "% típico".
    previsto = modelo(x_teste, training=False).numpy().flatten()
    erro_log = float(np.mean(np.abs(previsto - y[teste])) * desvio_log)

    return {
        "modelo": modelo,
        "media_log": media_log,
        "desvio_log": desvio_log,
        "erro_pct": float(np.exp(erro_log) - 1),
        "categorias": categorias,
        "max_hashtags": int(df["hashtags"].max()),
        "n_videos": len(df),
        "origem": origem,
    }


@st.cache_resource(show_spinner="Preparando o modelo inicial, aguarde alguns segundos...")
def treinar_base_simulada() -> dict:
    """Treina uma única vez o modelo inicial (base simulada) e guarda em cache."""
    return treinar_modelo(gerar_base_simulada(), origem="simulada")


def prever_alcance(pacote: dict, categorias, hashtags) -> np.ndarray:
    """Recebe listas de categorias/hashtags e devolve os alcances previstos."""
    entrada = {
        "categoria": tf.constant([[c] for c in categorias]),
        "hashtags": tf.constant([[float(h)] for h in hashtags]),
    }
    saida = pacote["modelo"](entrada, training=False).numpy().flatten()
    return np.exp(saida * pacote["desvio_log"] + pacote["media_log"])  # volta da escala log


# ---------------------------------------------------------------------------
# 4) FUNÇÕES AUXILIARES DA INTERFACE
# ---------------------------------------------------------------------------
def formatar_numero(valor: float) -> str:
    """Formata no padrão brasileiro: 12345 -> '12.345'."""
    return f"{valor:,.0f}".replace(",", ".")


def classificar(alcance: float) -> str:
    """Transforma o número previsto em uma decisão: Flopado, Regular ou Viral."""
    if alcance < LIMITE_FLOP:
        return "Flopado"
    if alcance > LIMITE_VIRAL:
        return "Viral"
    return "Regular"


# ---------------------------------------------------------------------------
# 5) INTERFACE (STREAMLIT)
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Calculadora de Engajamento",
        page_icon="📈",
        layout="centered",
    )

    # ----- Barra lateral, parte 1: treinar com arquivos -----
    with st.sidebar:
        st.header("Treinar a IA")
        st.caption(
            "Envie arquivos CSV com as colunas **categoria**, **hashtags** e "
            f"**alcance** (mínimo de {MIN_LINHAS} linhas) e clique em treinar."
        )
        arquivos = st.file_uploader(
            "Arquivos de treino (CSV)", type="csv", accept_multiple_files=True
        )
        st.download_button(
            "Baixar exemplo de CSV",
            data=gerar_base_simulada().head(5).to_csv(index=False),
            file_name="exemplo_treino.csv",
            mime="text/csv",
        )

        if st.button("Treinar modelo", type="primary"):
            if not arquivos:
                st.warning("Envie ao menos um arquivo CSV antes de treinar.")
            else:
                try:
                    dados, removidas = ler_arquivos(arquivos)
                    with st.spinner("Treinando a IA com os seus dados..."):
                        st.session_state["pacote"] = treinar_modelo(dados, origem="arquivos")
                    msg = f"Modelo treinado com {len(dados)} vídeos."
                    if removidas:
                        msg += f" {removidas} linha(s) inválida(s) foram ignoradas."
                    st.success(msg)
                except Exception as erro:  # mostra o motivo em vez de quebrar o app
                    st.error(f"Não foi possível treinar: {erro}")

        if "pacote" in st.session_state and st.button("Voltar para a base simulada"):
            del st.session_state["pacote"]
            st.rerun()

    # Usa o modelo treinado com arquivos; se não houver, o modelo inicial
    pacote = st.session_state.get("pacote") or treinar_base_simulada()
    simulado = pacote["origem"] == "simulada"

    # ----- Barra lateral, parte 2: informações do modelo -----
    with st.sidebar:
        st.divider()
        st.header("Sobre o modelo")
        st.metric("Erro típico no teste", f"± {pacote['erro_pct']:.0%}")
        origem_txt = "base simulada" if simulado else "arquivos enviados"
        st.caption(
            f"Treinado com {pacote['n_videos']} vídeos ({origem_txt}), "
            "80% para treino e 20% para teste. O erro indica a diferença "
            "típica entre o alcance previsto e o real na base de teste."
        )
        st.caption(
            f"🔴 Flopado: abaixo de {formatar_numero(LIMITE_FLOP)}  \n"
            f"🟡 Regular: de {formatar_numero(LIMITE_FLOP)} a "
            f"{formatar_numero(LIMITE_VIRAL)}  \n"
            f"🟢 Viral: acima de {formatar_numero(LIMITE_VIRAL)}"
        )

    # ----- Área principal -----
    st.title("📈 Calculadora de Engajamento de TikToker")
    st.write(
        "Selecione a categoria do vídeo e a quantidade de hashtags para "
        "estimar o alcance antes de publicar."
    )
    if simulado:
        st.info(
            "Atenção: esta previsão utiliza dados simulados e serve como "
            "demonstração. Para usar dados reais, envie arquivos na barra "
            "lateral e clique em **Treinar modelo**."
        )
    else:
        st.success("Modelo treinado com os arquivos enviados pela equipe.")

    col1, col2 = st.columns(2)
    categoria = col1.selectbox("Categoria do vídeo", pacote["categorias"])
    opcoes_hashtags = list(range(pacote["max_hashtags"] + 1))
    hashtags = col2.selectbox(
        "Quantidade de hashtags", opcoes_hashtags, index=min(4, len(opcoes_hashtags) - 1)
    )

    if st.button("Calcular alcance"):
        alcance = float(prever_alcance(pacote, [categoria], [hashtags])[0])
        status = classificar(alcance)

        st.metric("Alcance previsto (visualizações)", formatar_numero(alcance))

        if status == "Viral":
            st.success("🚀 **VIRAL**: grande potencial de alcance para este vídeo.")
        elif status == "Regular":
            st.warning("📊 **REGULAR**: desempenho dentro do esperado.")
        else:
            st.error("📉 **FLOPADO**: baixo potencial de alcance neste formato.")

        # Gráfico: como o alcance muda conforme o número de hashtags
        curva = prever_alcance(pacote, [categoria] * len(opcoes_hashtags), opcoes_hashtags)
        melhor = opcoes_hashtags[int(np.argmax(curva))]
        st.subheader(f"Alcance previsto por nº de hashtags ({categoria})")
        st.line_chart(
            pd.DataFrame(
                {"Alcance previsto": curva},
                index=pd.Index(opcoes_hashtags, name="Hashtags"),
            )
        )
        st.caption(
            f"Para esta categoria, o modelo indica melhor resultado com cerca de {melhor} hashtags."
        )

    st.divider()
    st.caption(
        "A previsão é uma estimativa de um modelo de IA e deve apoiar, "
        "não substituir, a análise da equipe de marketing."
    )


if __name__ == "__main__":
    main()
