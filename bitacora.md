Bitácora de dificultades – P4 (Agente DQN y flujo de entrenamiento)
Formato: qué problema apareció, por qué pasó y cómo se resolvió (o por qué no), y si fue conceptual o técnico.
1. Dependencias: LunarLander no se crea con `requirements.txt`
Problema: `gymnasium[classic-control]` no incluye Box2D. `gym.make("LunarLander-v3")` falla por falta de Box2D y pygame.
Por qué: `requirements.txt` (P1) no tiene el extra `box2d`.
Resolución: instalar `gymnasium==1.1.1`, `Box2D>=2.3.3` y `pygame>=2.1.3`. `box2d-py` no compila en el entorno (falla el build del paquete), así que se usó el wheel de `Box2D`.
Tipo: técnico.
Pendiente para el grupo: corregir `requirements.txt` (P1).
2. El logging de llamadas escribe en cada paso
Problema: los métodos del esqueleto llaman `log_method_call`, que abre y escribe `results/method_calls.csv` en cada llamada. `act()` y `update()` se llaman cientos de miles de veces por corrida.
Por qué: el esqueleto registra llamadas para verificar que se ejecutan los métodos, no pensado para bucles de entrenamiento.
Resolución: quitar `log_method_call` de `act()`, `update()` y `train.main()`. Se conserva solo en `DQNAgent.__init__`, que se llama una vez.
Tipo: técnico.
Pendiente para el grupo: confirmar con el profesor si el registro de llamadas es un requisito.
3. Dos versiones de `ReplayBuffer`
Problema: existe `Replay_buffer.py` en la raíz con otra firma (`push(state, action, reward, next_state, done)`, `sample(batch_size)` sin device). El README usa `src/replay_buffer.py`.
Por qué: P3 dejó la versión antigua al crear la nueva en `src/`.
Resolución: `train.py` y `agent.py` usan solo `src/replay_buffer.py`. El archivo de la raíz no se usa.
Tipo: conceptual/organizativo (cuál versión es la válida).
4. Ruta de los logs
Problema: el taller dice `logs/run_<seed>.csv` en un punto y `results/` en la estructura.
Resolución: se usó `results/run_<seed>.csv`, con `--out-dir` para cambiarla.
Tipo: organizativo.
5. Terminación vs. truncamiento en el objetivo de Bellman
Problema: al principio existía el riesgo de usar `truncated` para el factor `(1 − done)`. Si un episodio se corta por el límite de 500 pasos, el módulo sigue volando y el valor futuro no es cero.
Resolución: el buffer guarda `terminated`. `train.py` pasa `terminated` (no `truncated`) a `push`. Así, un corte por tiempo sigue haciendo bootstrap.
Tipo: conceptual.
6. Límite de pasos del ambiente
Problema: el ambiente trae 1000 pasos por defecto, pero el config pide 500.
Resolución: `gym.make(..., max_episode_steps=500)` en `train.py`.
Tipo: técnico.
7. Error en la tabla de acciones del README
Problema: la tabla de acciones de la sección 2 (P2) está al revés en 1 y 2. En Gymnasium, 1 es motor lateral izquierdo, 2 es motor principal y 3 es lateral derecho.
Resolución: el código no depende de los nombres (solo de `n_actions = 4`). Pendiente avisar al grupo para que P2 corrija la tabla.
Tipo: conceptual (nombres de acciones).

Bitácora de dificultades – P5 (Experimentación e hiperparámetros)
1. El config original no dejaba de explorar
Problema: en la primera ronda del barrido (500 episodios) todas las configuraciones quedaron entre −50 y −20, salvo la que cambiaba ε.
Por qué: `epsilon_decay_steps = 100000` se cuenta en pasos, pero al principio los episodios duran ~100 pasos porque el módulo choca rápido. En 500 episodios solo hubo ~51 000 pasos y ε seguía en 0.51.
Resolución: se repitió el barrido (fase 2) con ε en 50 000 pasos como base. Los resultados de la fase 1 se dejaron documentados como configuración fallida.
Tipo: conceptual (relación entre el calendario de ε, los pasos y el largo de los episodios).
2. train.py no guardaba el modelo ni permitía variar hiperparámetros
Problema: para el barrido había que editar `config.yaml` en cada corrida, y no quedaba ningún `.pt`.
Resolución: se añadieron `--set clave=valor` y `--tag` a `train.py`, y el guardado del mejor modelo (media móvil de 100 episodios) y del modelo final.
Tipo: técnico.
3. Tiempo de cómputo
Problema: cada corrida de 1000 episodios tarda ~20 minutos en CPU, y solo había 2 núcleos.
Por qué: se hace una actualización de la red en cada paso (~280 000 por corrida).
Resolución: `torch.set_num_threads(1)` para correr 2–3 semillas en paralelo, y barrido con una sola semilla y menos episodios (500–600). La desventaja es que las diferencias pequeñas del barrido no son concluyentes (ver reflexión).
Tipo: técnico.
4. Varianza entre semillas
Problema: con la misma configuración, la semilla 7 resolvió el ambiente (255.6) y la 123 terminó en 91.2. La combinación lr 0.0005 + target 250 dio con la semilla 42 menos que cada cambio por separado.
Por qué: inestabilidad propia de DQN (sobreestimación, olvido de experiencias, objetivo que se mueve).
Resolución: se reporta la media de 3 semillas y se eligió como mejor modelo el de la semilla 7. No se resolvió del todo: haría falta Double DQN o más semillas por configuración.
Tipo: conceptual.
5. Métrica de entrenamiento vs. evaluación
Problema: la media de entrenamiento de la semilla 42 fue 139.5, pero su modelo obtuvo 201.0 en evaluación.
Por qué: durante el entrenamiento sigue habiendo un 5 % de acciones aleatorias (ε = 0.05).
Resolución: se evaluó cada modelo con ε = 0 en 100 episodios con `experiments/analisis_politica.py`.
Tipo: conceptual.
6. Episodios que terminan por tiempo con recompensa alta
Problema: el 36 % de los episodios del mejor modelo terminaban por tiempo, lo que parecía un fallo.
Por qué: el módulo aterriza pero sigue usando los motores laterales y nunca queda en reposo, que es lo que termina el episodio.
Resolución: se clasificó el final de cada episodio (aterrizaje, choque o tiempo) y se revisó su recompensa (todas ≥ 116). Se explica en la reflexión.
Tipo: conceptual.
