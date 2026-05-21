import unicodedata
import re

STOPWORDS = {
    "de","del","la","el","los","las","un","una","unos","unas",
    "y","o","en","con","para","por","a","al","se","que","es",
    "su","sus","como","entre","sobre","le","lo","nos","sin",
    "ha","han","ser","sido","este","esta","estos","estas","no",
}

def limpiar(texto: str, max_chars: int = 512) -> str:
    """Normaliza texto: minusculas, sin tildes, sin caracteres especiales."""
    texto = unicodedata.normalize("NFD", str(texto).lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto[:max_chars]

def tokenizar(texto: str) -> list:
    """Limpia y elimina stopwords. Devuelve lista de tokens."""
    return [t for t in limpiar(texto).split() if t not in STOPWORDS and len(t) > 2]
