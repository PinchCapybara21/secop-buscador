import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from .utils import limpiar

MODELO_DEFAULT = "paraphrase-multilingual-MiniLM-L12-v2"


def buscar_embeddings(
    df: pd.DataFrame,
    columna: str,
    query: str,
    top_n: int = 10,
    score_minimo: float = 0.5,
    modelo: str = MODELO_DEFAULT,
    batch_size: int = 512,
) -> pd.DataFrame:
    """
    Busqueda semantica por embeddings sobre un DataFrame.

    Parametros
    ----------
    df           : DataFrame con los contratos
    columna      : nombre de la columna de texto (ej: 'descripcion_del_proceso')
    query        : texto de busqueda
    top_n        : numero de resultados a retornar
    score_minimo : similitud coseno minima (0 a 1). Default 0.5
    modelo       : modelo sentence-transformers a usar
    batch_size   : tamano de lote para encoding

    Retorna
    -------
    DataFrame filtrado con columna 'score_embedding' aniadida, ordenado de mayor a menor.
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    model = SentenceTransformer(modelo)

    textos = [limpiar(t) for t in df[columna].fillna("").tolist()]
    corpus_vecs = model.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_vec = model.encode(
        [limpiar(query)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores = (corpus_vecs @ query_vec.T).flatten()

    resultado = df.copy().reset_index(drop=True)
    resultado["score_embedding"] = scores.round(3)
    resultado = resultado[resultado["score_embedding"] >= score_minimo]
    resultado = resultado.sort_values("score_embedding", ascending=False).head(top_n)

    return resultado.reset_index(drop=True)
