# Verificación de la entrega del punto 6

La evaluación se ejecutó con los tres checkpoints publicados y sin entrenar de nuevo. Cada modelo se probó en 100 episodios (semillas 20000–20099), con ε = 0 y límite de 500 pasos. La línea base usa las mismas semillas y acciones uniformes.

## Comprobaciones realizadas

- **Cuatro pruebas automatizadas aprobadas:** prioridad de terminación frente a truncamiento simultáneo, independencia entre retorno ≥ 200 y aterrizaje, acciones greedy sin gradientes ni actualización de pesos, y repetibilidad del generador aleatorio con los mismos reinicios.
- **Estadísticas revisadas desde los CSV:** 100 filas por política y archivo, media, desviación (`ddof=0`), mínimo, máximo, umbral de 200, tipos de final y límites de pasos coinciden con los JSON y el README.
- **Repetición parcial independiente:** los tres primeros episodios DQN y los tres primeros aleatorios, repetidos sin renderizar, coinciden exactamente en retorno, duración y tipo de final con la evaluación oficial. Esto incluye el episodio que se grabó en el GIF.
- **Modelos y datos originales intactos:** las huellas SHA-256 de los checkpoints, CSV de entrenamiento, `config.yaml` y `train.py` coinciden con los del commit fuente.
- **Curvas revisadas:** media móvil retrospectiva con ventana completa; los primeros 99 valores no se rellenan. Los valores ausentes de loss no se convierten en cero. Las tres figuras se revisaron visualmente y se generaron con las versiones del entorno de evaluación.
- **GIF revisado:** 600 × 400 píxeles; se verificaron fotogramas de inicio, descenso y final. La codificación fusiona fotogramas idénticos, por lo que los 160 fotogramas capturados se guardan como 154 fotogramas con duración acumulada de 6400 ms.
- **Línea base repetida:** los tres CSV aleatorios son idénticos. Son los mismos 100 escenarios; no se suman como 300 observaciones independientes.

Comando para ejecutar las cuatro pruebas desde la raíz:

```bash
python -m unittest discover -s tests -v
```

Las verificaciones de repetición se hicieron en esta máquina y con estas dependencias; no demuestran igualdad exacta entre otras plataformas. La evidencia estructurada está en [verificacion.json](verificacion.json).

El archivo [requirements-evaluation.lock.txt](../requirements-evaluation.lock.txt) registra todas las versiones instaladas para esta ejecución. Para repetirla en un entorno nuevo se puede usar `pip install -r requirements-evaluation.lock.txt`. El archivo original `requirements.txt` se conserva.
