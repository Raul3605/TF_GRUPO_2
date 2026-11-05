# Predicción de la resistencia a compresión del concreto con Machine Learning

Aplicación de modelos de ML a un problema de ingeniería estructural.

## Problema

La resistencia a compresión (f'c) del concreto se determina a los 28 días mediante ensayos de probetas. Un modelo que la prediga a partir de la dosificación permite optimizar mezclas y reducir ensayos.

**Pregunta de investigación:** ¿Con qué precisión se puede predecir la resistencia a compresión del concreto a partir de su dosificación y edad, y qué variables influyen más?

## Dataset

Concrete Compressive Strength (Yeh, 1998), UCI Machine Learning Repository. 1 030 observaciones, 8 variables predictoras y 1 variable objetivo (resistencia a compresión, MPa). Tras eliminar 25 filas duplicadas quedan 1 005 observaciones, que corresponden a 428 mezclas distintas.

## Marco VDS (Yu y Barter, 2024)

Se aplica el marco PCS: predictibilidad, computabilidad y estabilidad. La validación se hace por mezcla (GroupKFold) para evitar fuga de datos entre edades de una misma dosificación.

## Integrantes

- Sergio Cornejo Espinoza
- Gerson Sanabria
- Raul Ayvar
- Cristhofer Apaza


## Estructura del repositorio

- `data/raw/`: datos originales sin modificar
- `data/processed/`: datos limpios y resultados del T2
- `docs/`: informes, presentaciones y figuras
- `notebooks/`: análisis exploratorio
- `src/`: código fuente en Python

## Cómo reproducir

    pip install -r requirements.txt
    python src/eda_t2.py

## Entregas

- T1: definición del problema, datos y repositorio
- T2: análisis exploratorio y plan de algoritmos
- T3: implementación, resultados y presentación final

## Bitácora de uso de IA

Se usó el asistente de IA Claude (Anthropic) como apoyo. Cada fila debe revisarse y completarse con los prompts reales.

| Etapa | Uso de la IA | Validación |
|---|---|---|
| Elección del tema | Ideas de temas con datos reales | Se eligió el dataset del concreto |
| T1 | Estructura del repositorio, informe y diapositivas | Revisado por el estudiante |
| T2 | Script de EDA, variables derivadas, validación agrupada e informe | Código ejecutado sobre el dataset del repo |
