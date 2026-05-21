import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buscador import buscar_bm25, buscar_embeddings

# ── datos de prueba ────────────────────────────────────────────
data = {
    "descripcion_del_proceso": [
        "Convenio interadministrativo para aunar esfuerzos en ciencia tecnologia e innovacion",
        "Prestacion de servicios profesionales en seguridad y salud en el trabajo",
        "Formacion y capacitacion de docentes en el plan territorial de educacion",
        "Adquisicion de equipos de computo y tecnologias de la informacion",
        "Mantenimiento de instalaciones locativas y obras civiles",
        "Apoyo a la gestion administrativa de la secretaria de salud",
    ],
    "entidad": [
        "Minciencias", "Secretaria Salud", "Secretaria Educacion",
        "MinTIC", "Universidad Nacional", "Hospital Universitario"
    ]
}
df = pd.DataFrame(data)

COLUMNA = "descripcion_del_proceso"
QUERIES = [
    "ciencia tecnologia innovacion",
    "seguridad salud trabajo",
    "formacion capacitacion docentes",
    "tecnologias informacion comunicaciones",
]

# ── prueba BM25 ────────────────────────────────────────────────
print("=" * 55)
print("BM25")
print("=" * 55)
for q in QUERIES:
    res = buscar_bm25(df, COLUMNA, q, top_n=3)
    print(f"\nQuery: '{q}'")
    for _, row in res.iterrows():
        print(f"  [{row['score_bm25']}] {row[COLUMNA][:80]}")

# ── prueba embeddings ──────────────────────────────────────────
print("\n" + "=" * 55)
print("EMBEDDINGS")
print("=" * 55)
for q in QUERIES:
    res = buscar_embeddings(df, COLUMNA, q, top_n=3, score_minimo=0.3)
    print(f"\nQuery: '{q}'")
    for _, row in res.iterrows():
        print(f"  [{row['score_embedding']}] {row[COLUMNA][:80]}")
