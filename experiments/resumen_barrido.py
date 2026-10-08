"""Resume los CSV de un barrido de hiperparámetros (P5).

Uso: python experiments/resumen_barrido.py results/sweep
Para cada run_*.csv calcula la media de los últimos 100 episodios, la mejor
media móvil de 100 episodios y el primer episodio en que esa media llega a 200.
"""
import sys
from pathlib import Path

import pandas as pd


def resumir(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    mov = df["reward"].rolling(100).mean()
    resuelto = mov[mov >= 200]
    return {
        "experimento": csv_path.stem.replace("_run", ""),
        "episodios": len(df),
        "pasos_totales": int(df["steps"].sum()),
        "media_ult_100": round(df["reward"].tail(100).mean(), 1),
        "desv_ult_100": round(df["reward"].tail(100).std(), 1),
        "mejor_media_100": round(mov.max(), 1),
        "ep_resuelto": int(resuelto.index[0]) + 1 if len(resuelto) else None,
    }


if __name__ == "__main__":
    carpeta = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    filas = [resumir(p) for p in sorted(carpeta.glob("*run_*.csv"))]
    tabla = pd.DataFrame(filas).sort_values("mejor_media_100", ascending=False)
    print(tabla.to_string(index=False))
    tabla.to_csv(carpeta / "resumen.csv", index=False)
