import pandas as pd
from rank_bm25 import BM25Okapi
from .utils import tokenizar


def buscar_bm25(
    df: pd.DataFrame,
    columna: str,
    query: str,
    top_n: int = 10,
    score_minimo: float = 0.0,
) -> pd.DataFrame:
    """
    Busqueda BM25 sobre un DataFrame.

    Parametros
    ----------
    df           : DataFrame con los contratos
    columna      : nombre de la columna de texto (ej: 'descripcion_del_proceso')
    query        : texto de busqueda
    top_n        : numero de resultados a retornar
    score_minimo : filtrar resultados por debajo de este score

    Retorna
    -------
    DataFrame filtrado con columna 'score_bm25' aniadida, ordenado de mayor a menor.
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    serie = df[columna].fillna("").reset_index(drop=True)
    corpus = [tokenizar(t) for t in serie]

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenizar(query))

    resultado = df.copy().reset_index(drop=True)
    resultado["score_bm25"] = scores.round(3)
    resultado = resultado[resultado["score_bm25"] > score_minimo]
    resultado = resultado.sort_values("score_bm25", ascending=False).head(top_n)

    return resultado.reset_index(drop=True)
