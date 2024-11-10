Machine Learning Handbook

1. Introducción al Aprendizaje Supervisado

El aprendizaje supervisado es una técnica donde un modelo es entrenado con un conjunto de datos que contiene tanto las entradas (características) como las salidas deseadas (objetivos). El objetivo del aprendizaje supervisado es que el modelo sea capaz de hacer predicciones precisas en datos nuevos.

Ejemplo

En este curso, comenzamos con un *modelo de regresión lineal*, un modelo simple pero poderoso en el que ajustamos una línea recta a los datos.

---

2. Regresión Lineal

La *regresión lineal* es un modelo de aprendizaje supervisado que se utiliza para predecir valores numéricos. En este contexto, ajustamos una línea recta que mejor representa la relación entre dos variables: una característica de entrada y una salida.

Ejemplo Práctico

Imagina que quieres predecir el *precio de una casa* en función de su *tamaño* en pies cuadrados. Tenemos datos históricos de precios de casas en Portland, EE. UU., con tamaños en el eje horizontal (X) y precios en el eje vertical (Y).

Si conocemos el tamaño de una casa (por ejemplo, 1250 pies cuadrados), nuestro modelo de regresión lineal nos permitirá estimar el precio al encontrar la intersección de ese tamaño con la línea ajustada en el gráfico. En este caso, la predicción para 1250 pies cuadrados podría ser $$220,000.

---

3. Tipos de Problemas en Aprendizaje Supervisado

- *Regresión*: Cuando el modelo predice valores numéricos continuos. Ejemplo: Predecir precios de casas.
- *Clasificación*: Cuando el modelo predice categorías discretas. Ejemplo: Identificar si una imagen es de un perro o un gato.

Diferencia entre Clasificación y Regresión
En la *clasificación*, los resultados posibles son un conjunto discreto y finito (por ejemplo, "gato" o "perro"). En la *regresión*, el modelo puede producir infinitos valores continuos (por ejemplo, cualquier precio de una casa).

---

4. Conjunto de Datos y Notación

En Machine Learning, usamos un conjunto de datos llamado *conjunto de entrenamiento* para enseñar al modelo.

- *Entrada (X)*: Características o entidades. En este caso, el tamaño de la casa.
- *Salida (Y)*: El objetivo o variable de salida. En este caso, el precio de la casa.
- *Ejemplo de entrenamiento*: Un par de entrada y salida \((x, y)\), por ejemplo, \((2104, 400)\).

Usamos:
- *\(m\)* para denotar el número total de ejemplos en el conjunto de entrenamiento.
- *\(x^{(i)}\)* y *\(y^{(i)}\)* para representar las características y objetivos del i-ésimo ejemplo.

---

5. Modelo de Regresión Lineal

La función de predicción de nuestro modelo de regresión lineal es:
\[
f(x) = w \cdot x + b
\]

Donde:
- *\(w\)*: Es la pendiente de la línea.
- *\(b\)*: Es la intersección con el eje Y.
- *\(y\)*: Es el valor objetivo real.
- *\(\u{005E}{y}\)*: Es el valor predicho por el modelo, también conocido como estimación de \(y\).

Para ajustar el modelo, calculamos los valores de \(w\) y \(b\) que minimicen la diferencia entre los valores reales y las predicciones.

---

6. Predicciones con el Modelo

Para una nueva entrada \(x\) (por ejemplo, el tamaño de una casa), el modelo predice un valor estimado \(\u{005E}{y}\), el cual puede ser aproximado con la función lineal que generamos:
\[
\u{005E}{y} = f(x) = w \cdot x + b
\]

Este valor \(\u{005E}{y}\) representa una predicción del precio de la casa basada en el tamaño proporcionado.

---

7. Visualización y Ajuste de Datos

El ajuste del modelo se representa gráficamente como una línea recta que minimiza la diferencia entre los puntos de datos reales y los valores predichos por la función lineal.

En este caso:
- *Eje X*: Tamaño de la casa (entrada).
- *Eje Y*: Precio de la casa (salida).

Cada punto representa una casa con su tamaño y precio en el conjunto de entrenamiento. La línea recta representa nuestro modelo de regresión lineal ajustado.

---

8. Notación y Convenciones en Machine Learning

1. *Hipótesis o Función*: \(f(x)\) o \(f{w,b}(x)\), representa la función que el modelo utiliza para hacer predicciones.
2. *Predicción*: \(\u{005E}{y}\), el valor predicho o estimado de \(y\).
3. *Variables*:
   - *\(x\)*: Entrada, como el tamaño de la casa.
   - *\(y\)*: Valor real de salida, como el precio de la casa.
4. *Parámetros del Modelo*:
   - *\(w\)*: Pendiente de la línea.
   - *\(b\)*: Intersección con el eje Y.

Con esta notación, es más fácil comunicar los conceptos de aprendizaje automático de manera uniforme y precisa.

---

9. Implementación Matemática de la Regresión Lineal

La fórmula de regresión lineal que se utiliza para predecir es:
\[
f(x) = w \cdot x + b
\]

El objetivo del algoritmo de aprendizaje es encontrar los valores óptimos para \(w\) y \(b\), de modo que la línea recta que definan minimice la diferencia entre los valores reales y las predicciones.

---

Resumen

- La *regresión lineal* es una técnica para predecir valores numéricos continuos.
- Utilizamos un *conjunto de entrenamiento* para entrenar al modelo, compuesto por características de entrada \(x\) y objetivos de salida \(y\).
- La fórmula del modelo de regresión lineal es \(f(x) = w \cdot x + b\).
- La diferencia entre *regresión* y *clasificación* es que la regresión predice valores continuos, mientras que la clasificación predice categorías discretas.
