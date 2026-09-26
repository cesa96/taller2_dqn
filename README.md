# Taller 2: agente DQN para LunarLander-v3

## Estado actual del proyecto

El ambiente seleccionado para la actividad es `LunarLander-v3` de Gymnasium.
Por ahora, los métodos Python son esqueletos: **solo registran el nombre del
método y la fecha/hora de llamada** en `results/method_calls.csv`. Todavía no
crean el ambiente, entrenan ni evalúan un agente, ni generan resultados de
aprendizaje. Las secciones sobre red y flujo describen el diseño previsto; los
resultados y reflexiones basados en datos quedan pendientes de ejecutar el
entrenamiento.

## 1. Descripción del ambiente

LunarLander simula el control de un módulo lunar que debe aterrizar entre las
banderas de una plataforma. El agente debe reducir su velocidad y orientar el
módulo para conseguir un aterrizaje seguro. Es adecuado para DQN porque tiene
observaciones numéricas y un conjunto discreto de acciones.

## 2. Espacio de observaciones y acciones

### Observaciones

El espacio de observación es un vector continuo de 8 valores:

| Índice | Valor observado |
| --- | --- |
| 0 | Posición horizontal del módulo |
| 1 | Posición vertical del módulo |
| 2 | Velocidad horizontal |
| 3 | Velocidad vertical |
| 4 | Ángulo del módulo |
| 5 | Velocidad angular |
| 6 | Indicador de contacto de la pata izquierda con el suelo |
| 7 | Indicador de contacto de la pata derecha con el suelo |

### Acciones

El espacio es `Discrete(4)`. Cada acción selecciona una de estas opciones:

| Acción | Efecto |
| --- | --- |
| 0 | No encender motores |
| 1 | Encender el motor principal |
| 2 | Encender el motor lateral izquierdo |
| 3 | Encender el motor lateral derecho |

### Recompensa

La recompensa combina el progreso hacia la zona de aterrizaje, la reducción
de velocidad, la orientación y el contacto de las patas. También aplica
penalizaciones por el consumo de combustible. El aterrizaje seguro recibe una
recompensa final positiva y un choque una penalización final. Por ello, no basta
con maximizar una recompensa puntual: se busca aterrizar de forma controlada y
con poco gasto de combustible.

## 3. Flujo lógico previsto para el entrenamiento


## 4. Particularidades del entorno

- LunarLander combina control de posición, velocidad y orientación; una acción
  útil depende de la fase del aterrizaje.
- Los motores consumen combustible, así que mantenerlos encendidos puede
  facilitar el control inmediato, pero reduce la recompensa.
- Es importante distinguir entre aterrizaje seguro, choque y truncamiento por
  límite de tiempo: terminar por tiempo no necesariamente significa que el
  módulo haya aterrizado.
- Gymnasium requiere la dependencia de Box2D para este entorno. Antes de
  ejecutar una futura implementación del entrenamiento, se debe instalar
  `gymnasium[box2d]` además de las dependencias de PyTorch y las herramientas
  de registro/gráficas que se utilicen.
## 5. Explicación de la red neuronal


## 6. Resultados del entrenamiento


## 7. Reflexión sobre los resultados obtenidos


## 8. Reflexión sobre los principales retos o dificultades



