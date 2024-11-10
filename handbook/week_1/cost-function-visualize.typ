Handbook de Machine Learning: Función de Costo en Regresión Lineal

1. Introducción a la Función de Costo

La *función de costo* en machine learning es una métrica que indica qué tan bien un modelo predice el valor de salida deseado. En el caso de *regresión lineal*, el objetivo es encontrar los parámetros óptimos de la función lineal para minimizar la diferencia entre las predicciones y los valores reales en el conjunto de entrenamiento.

- Dada la función lineal:
\[
f_{w, b}(x) = w \cdot x + b
\]
Donde:
- $w$ y $b$ son los *parámetros* o *coeficientes* del modelo que se ajustan durante el entrenamiento.
- $x$ representa las *características de entrada*.
- $f_{w, b}(x)$ es la *predicción* de la salida.

Ejemplos de $f(x)$ con diferentes valores de $w$ y $b$

1. *Caso 1*: $w = 0$, $b = 1.5$
- La función $f(x) = 0 \cdot x + 1.5 = 1.5$ produce una *línea horizontal* a $y = 1.5$.
   
2. *Caso 2*: $w = 0.5$, $b = 0$
- La función $f(x) = 0.5 \cdot x$ tiene una *pendiente* de $0.5$ y *pasa por el origen*.
   
3. *Caso 3*: $w = 0.5$, $b = 1$
- La función $f(x) = 0.5 \cdot x + 1$ cruza el eje $y$ en $1$ y tiene una pendiente de $0.5$.

2. Definición de la Función de Costo

Para elegir los valores óptimos de $w$ y $b$, se necesita una métrica que evalúe el error de predicción en todos los ejemplos de entrenamiento. Esta métrica es la *función de costo*.

Error Cuadrático

Para un ejemplo de entrenamiento $i$, el error se define como la diferencia entre la predicción y el valor objetivo:

\[
\text{error} = \hat{y} - y
\]

Donde:
- $\hat{y}$ es la predicción de la salida para una entrada $x$ dada.
- $y$ es el valor objetivo real.

Error Cuadrático Medio (MSE)

La función de costo total calcula el error cuadrático medio en todo el conjunto de entrenamiento:

\[
J(w, b) = \frac{1}{2m} \sum_{i=1}^m \left( f_{w, b}(x^{(i)}) - y^{(i)} \right)^2
\]

Donde:
- $m$ es el número total de ejemplos en el conjunto de entrenamiento.
- La división por $2$ es una convención que simplifica cálculos posteriores.

3. Interpretación de la Función de Costo

- Si $J(w, b)$ es *grande*, la predicción promedio está lejos de los valores reales; el modelo no se ajusta bien a los datos.
- Si $J(w, b)$ es *pequeño*, la predicción promedio está cerca de los valores reales; el modelo se ajusta bien a los datos.

4. Ejemplo de Cálculo de la Función de Costo

Supongamos un conjunto de entrenamiento con $m = 47$ ejemplos. La función de costo calcularía el error cuadrático para cada ejemplo, sumaría estos errores y los dividiría por $2m$.

1. *Cálculo del Error Cuadrático*: Para un ejemplo específico $i$, se calcula $(\hat{y}^{(i)} - y^{(i)})^2$.
2. *Suma de Errores*: Se suman los errores cuadráticos de todos los ejemplos.
3. *Promedio del Error Cuadrático*: Se divide el total de errores entre $2m$ para obtener el error cuadrático medio.

Esta función de costo ayuda a determinar qué tan bien se ajusta la línea del modelo a los datos de entrenamiento.

5. Aplicación en Machine Learning

La función de costo es una herramienta central en el entrenamiento de modelos de regresión lineal y otros modelos de machine learning. Reducir el valor de $J(w, b)$ mediante el ajuste de $w$ y $b$ es el objetivo principal en el entrenamiento del modelo.

Resumen

- La *función de costo* mide la precisión del modelo.
- En *regresión lineal*, $J(w, b)$ mide el error cuadrático medio de las predicciones.
- Optimizar los valores de $w$ y $b$ para *minimizar $J(w, b)$* ayuda a mejorar el rendimiento del modelo en la tarea de predicción.

Para comprender la función de costo más a fondo, revisar cómo cambiar $w$ y $b$ afecta la forma de la línea y el valor de $J(w, b)$ en tus datos de entrenamiento.
