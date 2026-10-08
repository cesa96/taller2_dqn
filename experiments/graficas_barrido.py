"""Gráficas del barrido de hiperparámetros (P5).

Uso: python experiments/graficas_barrido.py <carpeta_fase2> <carpeta_final>
Genera experiments/figuras/barrido_fase2.png y experiments/figuras/semillas_final.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent / "figuras"
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TEXTO, TEXTO2, REJILLA = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": REJILLA, "axes.labelcolor": TEXTO2,
                     "xtick.color": TEXTO2, "ytick.color": TEXTO2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def media_movil(csv: Path) -> pd.Series:
    return pd.read_csv(csv)["reward"].rolling(100).mean()


def estilo(ax, titulo):
    ax.axhline(200, color=TEXTO2, lw=1, ls="--")
    ax.text(0.01, 205, "resuelto (200)", color=TEXTO2, fontsize=9, va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa (media móvil de 100 episodios)")
    ax.set_title(titulo, loc="left", color=TEXTO, fontsize=12)
    ax.grid(axis="y", color=REJILLA, lw=0.8)


def barrido(carpeta: Path) -> None:
    etiquetas = {"target250": "target cada 250", "lr5e-4": "lr 0.0005", "batch128": "batch 128",
                 "base": "base (ε en 50k pasos)", "epsdec25k": "ε en 25k pasos", "gamma095": "γ = 0.95"}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for color, (clave, nombre) in zip(COLORES, etiquetas.items()):
        ax.plot(media_movil(carpeta / f"f2_{clave}_run_42.csv"), color=color, lw=2, label=nombre)
    estilo(ax, "Barrido fase 2: una variación a la vez sobre la base (600 ep., semilla 42)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "barrido_fase2.png", dpi=150); plt.close(fig)


def semillas(carpeta: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    series = {}
    for color, s in zip(COLORES, [42, 7, 123]):
        series[s] = media_movil(carpeta / f"run_{s}.csv")
        ax.plot(series[s], color=color, lw=2, label=f"semilla {s}")
    ax.plot(pd.concat(series, axis=1).mean(axis=1), color=TEXTO, lw=2, ls=":", label="promedio 3 semillas")
    estilo(ax, "Configuración final: 1000 episodios, 3 semillas")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "semillas_final.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    barrido(Path(sys.argv[1]))
    semillas(Path(sys.argv[2]))
