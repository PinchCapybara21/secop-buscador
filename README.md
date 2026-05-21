# secop-buscador

Funciones de búsqueda sobre contratos del SECOP. Soporta búsqueda por palabras clave (BM25) y búsqueda semántica (embeddings). Recibe un DataFrame, el nombre de la columna de texto y una query, y devuelve el DataFrame filtrado y ordenado por relevancia.

---

## Instalación

```bash
git clone https://github.com/PinchCapybara21/secop-buscador.git
cd secop-buscador
pip install -r requirements.txt
```

---

## Uso rápido

```python
import pandas as pd
from buscador import buscar_bm25, buscar_embeddings

df = pd.read_parquet("contratos.parquet")

# Búsqueda por palabras clave (rápida, sin GPU)
resultados = buscar_bm25(
    df=df,
    columna="descripcion_del_proceso",
    query="ciencia tecnologia innovacion",
    top_n=10,
)

# Búsqueda semántica (más precisa, tarda más en datasets grandes)
resultados = buscar_embeddings(
    df=df,
    columna="descripcion_del_proceso",
    query="convenio universidad investigacion",
    top_n=10,
    score_minimo=0.6,
)

print(resultados[["descripcion_del_proceso", "score_embedding"]])
```

Ambas funciones devuelven el DataFrame original filtrado y con una columna de score añadida (`score_bm25` o `score_embedding`), ordenado de mayor a menor relevancia.

---

## Parámetros

### `buscar_bm25`

| Parámetro     | Tipo    | Default | Descripción                              |
|---------------|---------|---------|------------------------------------------|
| df            | DataFrame | —     | DataFrame con los contratos              |
| columna       | str     | —       | Nombre de la columna de texto            |
| query         | str     | —       | Texto de búsqueda                        |
| top_n         | int     | 10      | Número de resultados a retornar          |
| score_minimo  | float   | 0.0     | Score mínimo para incluir un resultado   |

### `buscar_embeddings`

| Parámetro     | Tipo    | Default                                      | Descripción                              |
|---------------|---------|----------------------------------------------|------------------------------------------|
| df            | DataFrame | —                                          | DataFrame con los contratos              |
| columna       | str     | —                                            | Nombre de la columna de texto            |
| query         | str     | —                                            | Texto de búsqueda                        |
| top_n         | int     | 10                                           | Número de resultados a retornar          |
| score_minimo  | float   | 0.5                                          | Similitud coseno mínima (0 a 1)          |
| modelo        | str     | `paraphrase-multilingual-MiniLM-L12-v2`      | Modelo sentence-transformers             |
| batch_size    | int     | 512                                          | Tamaño de lote para encoding             |

---

## Estructura del repositorio

```
secop-buscador/
├── buscador/
│   ├── __init__.py       # exporta buscar_bm25 y buscar_embeddings
│   ├── bm25.py           # búsqueda por palabras clave
│   ├── embeddings.py     # búsqueda semántica
│   └── utils.py          # limpieza y tokenización de texto
├── tests/
│   └── test_buscador.py  # ejemplos de uso y pruebas
├── requirements.txt
└── README.md
```

---

## Cuándo usar cada método

| Situación | Recomendación |
|---|---|
| Dataset grande (>500k filas), necesitas rapidez | `buscar_bm25` |
| Dataset mediano (<200k filas), necesitas precisión semántica | `buscar_embeddings` |
| Quieres lo mejor de los dos | Usa BM25 para filtrar candidatos y embeddings para reordenar |

---

## Contexto

Este módulo fue desarrollado para el proyecto de búsqueda semántica de contratos universitarios en SECOP 1, usando como base contratos históricos descargados del SECOP 2.
