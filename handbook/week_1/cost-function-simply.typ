= Machine Learning Handbook: La Función de Costo en Regresión Lineal



La Función de Costo en la Regresión Lineal
Una parte fundamental de la implementación de la regresión lineal es definir la función de costo, la cual permite evaluar la calidad del modelo. En este tema, explicaremos qué es la función de costo y cómo se utiliza para mejorar el modelo.

1. Introducción
Para entrenar un modelo de regresión lineal, trabajamos con un conjunto de datos de entrenamiento que contiene pares de entrada `(x)` y salida `(y)`. El modelo utilizado para ajustar estos datos es la función lineal \( f{w, b}(x) = w \cdot x + b \), donde:

- *w* y *b* son los parámetros del modelo.
- *w* es conocido como el peso o pendiente.
- *b* es conocido como el sesgo o intersección con el eje y.

La meta es ajustar los valores de *w* y *b* para que la línea generada por \( f(x) \) se ajuste bien a los datos de entrenamiento.

2. Ejemplos de Variación de Parámetros

Ejemplo 1
Si \( w = 0 \) y \( b = 1.5 \):
\[
f(x) = 0 \cdot x + 1.5 = 1.5
\]
Esta función produce una línea horizontal en *y = 1.5*. Aquí, *b* representa la intersección con el eje y.

Ejemplo 2
Si \( w = 0.5 \) y \( b = 0 \):
\[
f(x) = 0.5 \cdot x
\]
Con esta configuración, la línea pasa por el origen (0,0) y tiene una pendiente de *0.5*, ya que el valor de *w* determina la pendiente.

Ejemplo 3
Si \( w = 0.5 \) y \( b = 1 \):
\[
f(x) = 0.5 \cdot x + 1
\]
Esta línea cruza el eje y en *y = 1* y tiene una pendiente de *0.5*. Cambiar *b* desplaza la línea verticalmente.

3. Definición de la Función de Costo
La función de costo mide qué tan cerca están las predicciones del modelo de los valores reales del conjunto de entrenamiento. La función de costo \( J(w, b) \) se define para evaluar la diferencia entre la predicción del modelo \( \hat{y} \) y el objetivo real \( y \).

Para una entrada dada \( x^{(i)} \), la predicción del modelo es:
\[
\hat{y}^{(i)} = f{w, b}(x^{(i)}) = w \cdot x^{(i)} + b
\]

La función de costo, conocida como *error cuadrático medio (MSE)*, se define como:
\[
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w, b}(x^{(i)}) - y^{(i)} \right)^2
\]
donde:
- *m* es el número total de ejemplos de entrenamiento.
- El término adicional *1/2* se agrega por convención, para simplificar cálculos posteriores en la optimización del modelo.

4. Interpretación Intuitiva de la Función de Costo
La función de costo nos ayuda a encontrar los valores de *w* y *b* que minimizan el error en el conjunto de entrenamiento. Esto significa que buscamos valores para *w* y *b* que hagan que \( J(w, b) \) sea lo más pequeño posible.

Cuando *J(w, b)* es pequeño, significa que la línea generada por \( f(x) \) se ajusta bien a los puntos de datos. Si *J(w, b)* es grande, indica que el modelo necesita ajustes en *w* y *b* para mejorar el ajuste.

5. Ejemplo Visual con Función Simplificada

Caso Simplificado
Consideremos un modelo simplificado donde \( f{w}(x) = w \cdot x \) (es decir, *b = 0*).

Para un conjunto de entrenamiento con puntos en las posiciones *(1,1), (2,2), (3,3)*, se puede observar cómo varía el error cuadrático medio al cambiar el valor de *w*:

1. Para \( w = 1 \):
   - La línea de predicción coincide con todos los puntos de datos, lo cual genera un error *J(w) = 0*.
   
2. Para \( w = 0.5 \):
   - La línea de predicción se aleja de los puntos, aumentando el error cuadrático.

Al visualizar *J(w)* en función de *w*, se observa que el valor mínimo de *J* ocurre cuando \( w = 1 \).

6. Minimización de la Función de Costo
El objetivo en regresión lineal es encontrar los valores de *w* y *b* que minimicen \( J(w, b) \), es decir, que minimicen el error entre las predicciones y los valores reales. Matemáticamente, esto se expresa como:
\[
\min{w, b} J(w, b)
\]

Para encontrar los mejores parámetros, se utilizan algoritmos de optimización, como el descenso de gradiente, que permite ajustar iterativamente *w* y *b* hasta que \( J(w, b) \) alcance un valor mínimo.

Resumen
- La función de costo es esencial para medir el rendimiento de un modelo de regresión lineal.
- *J(w, b)* se minimiza ajustando *w* y *b* para que la línea generada se aproxime a los datos de entrenamiento.
- La función de costo por error cuadrático es ampliamente utilizada en problemas de regresión debido a su simplicidad y efectividad en la optimización de parámetros del modelo.
