# Prediccion de la resistencia a compresion del concreto con Machine learning

Aplicacion de modelo ML a um proyecto de ingenieria estructural.
## Problema.
La resistencia a compresion f'c del concreto a los 28 dias. Un modelo que la prediga a partir de la dosificación permite optimizar mezclas y reducir ensayos, dando resultados que ayuden a la toma de decisiones,para el planeamiento y ejecución de obras obteniendo una mayor eficiencia. 
**pregunta de investigacion** ¿Con qué precisión se puede predecir la resistencia a compresión del concreto a partir de su dosificación y edad, y qué variables influyen más?

## Dataset
Concrete Compressive Strength (Yeh, 1998), UCI Machine Learning Repository.
1,030 observaciones, 8 variables predictoras (cemento, escoria, ceniza volante, agua, superplastificante, agregado grueso, agregado fino, edad) y 1 variable objetivo (resistencia a compresión, MPa).

## Marco VDS (Yu & Barter, 2024)
Se aplica el marco PCS: predictibilidad, computabilidad y estabilidad.

## Integrantes
- Sergio Cornejo Espinoza
- Gerson Sanabria
- Raul Ayvar
- Cristhofer Apaza 

## Estructura del repositorio
- `data/raw/`: datos originales sin modificar
- `data/processed/`: datos limpios
- `docs/`: informes y presentaciones
- `notebooks/`: análisis exploratorio
- `src/`: código fuente en Python

## Cómo reproducir
    pip install -r requirements.txt

## Entregas
- T1: definición del problema, datos y repositorio
