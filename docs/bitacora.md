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
