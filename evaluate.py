import gymnasium as gym
import numpy as np
env = gym.make("LunarLander-v3")

print("=" * 60)
print("INFORMACIÓN DEL ENTORNO")
print("=" * 60)

# Espacio de observaciones
print("\nESPACIO DE OBSERVACIONES")
print("-" * 30)
print(env.observation_space)

print("\nDimensión del estado:")
print(env.observation_space.shape)

print("\nLímites inferiores:")
print(env.observation_space.low)

print("\nLímites superiores:")
print(env.observation_space.high)

# Espacio de acciones
print("\nESPACIO DE ACCIONES")
print("-" * 30)
print(env.action_space)

print("\nNúmero de acciones disponibles:")
print(env.action_space.n)

acciones = {
    0: "No hacer nada",
    1: "Activar motor lateral izquierdo",
    2: "Activar motor principal",
    3: "Activar motor lateral derecho"
}

for k, v in acciones.items():
    print(f"{k} -> {v}")

# Estado inicial
obs, info = env.reset(seed=42)

print("\nESTADO INICIAL DE EJEMPLO")
print("-" * 30)
print(obs)

variables_estado = [
    "Posición X",
    "Posición Y",
    "Velocidad X",
    "Velocidad Y",
    "Ángulo",
    "Velocidad angular",
    "Contacto pata izquierda",
    "Contacto pata derecha"
]

print("\nInterpretación del estado:")
for nombre, valor in zip(variables_estado, obs):
    print(f"{nombre}: {valor:.4f}")

env.close()


"""



## Tabla de observaciones

| Variable | Significado | Rango aproximado | Tipo |
|-----------|-------------|------------------|------|
| Posición X | Distancia horizontal respecto al centro | [-2.5, 2.5] | float |
| Posición Y | Altura respecto al punto de aterrizaje | [-2.5, 2.5] | float |
| Velocidad X | Velocidad horizontal | [-10, 10] | float |
| Velocidad Y | Velocidad vertical | [-10, 10] | float |
| Ángulo | Inclinación del módulo lunar (lander) | [-6.28, 6.28] | float |
| Velocidad angular | Velocidad de rotación | [-10, 10] | float |
| Contacto pata izquierda | 1 si toca el suelo, 0 si no | [0, 1] | float |
| Contacto pata derecha | 1 si toca el suelo, 0 si no | [0, 1] | float |


## Tabla de acciones
0	No hacer nada

1	Motor lateral izquierdo

2	Motor principal (empuje hacia arriba)

3	Motor lateral derecho


## Sistema de Recompensas

El entorno LunarLander utiliza una función de recompensa diseñada para incentivar aterrizajes seguros, estables y eficientes en el consumo de combustible. 
La recompensa total de un episodio corresponde a la suma de las recompensas obtenidas en cada paso de tiempo.

### Componentes de la recompensa

- El módulo recibe puntos cuando se mueve hacia el centro de la zona de aterrizaje.
- Pierde puntos si se aleja de ese objetivo.
- Se premia descender con una velocidad baja y controlada.
- Se castigan las velocidades altas.
- Inclinaciones fuertes generan penalizaciones.
- Encender los motores gasta combustible y siempre genera una penalización.



### Recompensas y penalizaciones principales

| Evento | Recompensa |
|----------|------------|
| Cada pata en contacto con el suelo | +10 |
| Uso del motor lateral | -0.03 por paso |
| Uso del motor principal | -0.3 por paso |
| Aterrizaje exitoso | +100 |
| Colisión o destrucción | -100 |

### Objetivo del agente

La política óptima debe:

1. Mantenerse cerca de la plataforma.
2. Reducir la velocidad de caída de manera progresiva.
3. Mantener una orientación estable.
4. Usar el combustible con moderación.
5. Aterrizar suavemente sin chocar.

### Interpretación en Reinforcement Learning

La recompensa funciona como una guía que le dice al agente qué acciones lo acercan a un aterrizaje correcto y cuáles lo alejan de él.
Las recompensas positivas refuerzan comportamientos seguros y controlados (rutas que puede repetir), mientras que las penalizaciones evitan movimientos bruscos, inclinaciones peligrosas y el uso excesivo de motores.

## ¿Por qué no se apilan frames (el vector ya incluye velocidades).?
El estado actual tiene toda l ainformación necesaria para la toma de decisiones, el estado tiene toda la información.



## Particularidades del ambiente
* Espacio de estados continuo

Existen prácticamente infinitos estados posibles. Por esta razón, algoritmos tabulares como Q-Learning puro no escalan bien sin discretización, el estado está compuesto por variables continuas:

Posición (x, y)
Velocidad (vx, vy)
Ángulo
Velocidad angular
Contacto de las patas
 

* Problema de control dinámico

El agente no sólo debe decidir qué hacer, sino cuándo hacerlo, por ejemplo cuando al encender demasiado el motor principal genera inestabilidad.
Las acciones tienen efectos que se propagan durante varios pasos del tiempo.

* Recompensa densa  

A diferencia de otros entornos donde la recompensa sólo aparece al final, LunarLander proporciona retroalimentación continua:

    -    acercarse al objetivo
    -    reducir velocidad
    -    mantenerse vertical
    -    tocar el suelo con las patas
    -  Optimización multiobjetivo

El agente debe optimizar varios aspectos simultáneamente:

    *    Llegar a la plataforma.
    *   Reducir velocidad.
    *   Mantener estabilidad.
    *   Ahorrar combustible.
    *   Evitar colisiones.

Muchas veces estos objetivos entran en conflicto al estabilizarse o ahorrar combustible.
El agente debe encontrar un equilibrio.

* * Alta dependencia temporal
Una acción puede afectar el resultado muchos pasos después.
* *  Física realista (Box2D)
El entorno está construido sobre Box2D.

El agente debe tener en cuenta todas las observaciones:

- gravedad
- aceleración
- momento angular
- velocidad lineal
- colisiones
- contacto de las patas

Por ello es mucho más complejo que entornos como CartPole
"""
