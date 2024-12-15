#import "lib/template.typ": main
#import "lib/simpleTable.typ": simpleTable
#import "lib/codeBlock.typ": codeBlock
#show: doc => main(
  title: [
    Machine Learning
  ],
  version: "v0.1.",
  authors: (
    (name: "Rolando Lora", email: "rolando.lora@fundacion-jala.org"),
  ),
  abstract: [
    This is a collection of notes and thoughts that I've been taking while learning about machine learning.
    It is based on the *"Machine Learning"* specialization from Coursera by _Andrew Ng_ as well as the lessons and labs from our course at *Fundación Jala*.
  ],
  doc,
)

= Supervised Learning

== Linear Regression Notes

To recap, simple linear regression with one variable is given by:

#simpleTable(
  columns: (1fr, 1fr),
  [*Attribute*], [*Formula*],
  [*Model*], [
    $ f_(w,b)(x) = w x + b $
  ],
  [*Parameters*], [
    $ w, b $
  ],
  [*Cost Function*], [
    $ J(w, b) = 1/(2m) sum_(i=1)^m ( f_(w,b)( x^(\(i\)) ) - y^(\(i\) ) )^2 $
  ],
  [*Objective*], [
    $ min_(w,b) J(w, b) $
  ],
)

This is the simplest form of linear regression, and we are given a dataset:

$ (X, Y) $

Where $X$ is a vector of features and $Y$ is a vector of labels.

#figure(
  image("./images/2024-10-30-simple-regression.png"),
  caption: [
    I made this diagram using _excalidraw_, that, in my head, represents what we are trying to do. $x^(\(i\))$ is the $i$-th training example, and $y^(\(i\))$ is the $i$-th training label.
  ]
)

=== Cost Function

One interesting observation is how the cost function $J(w)$ changes as we change the value of $w$. For example, the code below plots the cost for a simple target:

$ f_(w, b = 0) = w x $

Notice that for convenience, we are using $b = 0$, so our target is simply $f_(w) = w x$.

#codeBlock(
  ```python
  def plot_simple_error(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    w_range: NDArray[np.float64],
    x_marker_position: float,
  ) -> Tuple[Figure, Axes]:

    fig, ax = plt.subplots(figsize=(10, 6))
    errors = np.array([cost_function(y, simple_hypothesis(x, w)) for w in w_range])
    ax.plot(w_range, errors, color="blue", label="J(w)")
    ax.axvline(
        x=x_marker_position,
        color="red",
        linestyle="--",
        label=f"w = {x_marker_position}",
    )
    ax.set_xlabel("w")
    ax.set_ylabel("J(w)")
    ax.set_title("Cost as a function of w - J(w)")
    ax.legend()

    return fig, ax
  ```
)

This allow us to visualize the behavior of the cost function by using a *known model* and a range of sampling values for $w$. In the example below, we are using:

$ f_(w) =  (4 x) / 3 $

#codeBlock(
  ```python
  w: float = 4 / 3
  x_train = np.linspace(-5, 5, 100)
  y_train = simple_hypothesis(x_train, w)
  w_sample_range = np.linspace(-5, 8, 100)

  fig, ax = plot_simple_error(
      x=x_train, y=y_train, w_range=w_sample_range, x_marker_position=w
  )
  ```
)

@simple-cost shows the resulting plot. We can observe how the cost approaches a minimum as we change the value of $w$ from both sides, converging to a value close to $1.33$.

#figure(
  image("./images/cost-linear-reg-line.png"),
  caption: [
    Plot of the cost function $J(w)$ as a function of $w$ for the target $f_(w) =  (4 x) / 3$.
  ]
)<simple-cost>

A similar approach can be used to now introduce $b$ as a second target parameter. For example, using a target of the form:

$ f_(w, b) = (2 x) / 5 - 3 / 2  $

#codeBlock(
  ```python
  w = 2.5
  b = -1.5
  x_train = np.linspace(-5, 5, 100)
  y_train = complex_hypothesis(x_train, w, b)
  w_sample_range = np.linspace(-5, 5, 100)
  b_sample_range = np.linspace(-5, 5, 100)

  fig, ax = plot_complex_error_with_contour(
      x=x_train, y=y_train, w_range=w_sample_range, b_range=b_sample_range
  )
  ```
)

@complex-cost shows the resulting plot. We can observe how the cost function has a minimum at $(w, b) = (2.5, -1.5)$ but it is a bit more difficult to observe. As we increase the number of dimensions in the feature space, it becomes even more difficult to visualize the cost function.

But the main idea is that for linear regression, the cost function is *convex and will always have a global minimum.*

#figure(
  image("./images/cost-linear-reg-contour.png"),
  caption: [
    Plot of the cost function $J(w, b)$ as a function of $w$ and $b$ for the target $f_(w, b) = (2 x) / 5 - 3 / 2$.
  ]
)<complex-cost>

== Machine Learning Handbook: Implementación del Algoritmo de Descenso de Gradiente

Introducción

El descenso de gradiente es un algoritmo fundamental para el ajuste de parámetros en modelos de machine learning. Este handbook explica cómo implementar correctamente el algoritmo y evitar errores comunes.

Estructura del Algoritmo de Descenso de Gradiente

El descenso de gradiente busca minimizar la función de costo \(J(w, b)\) ajustando los parámetros \(w\) y \(b\). La fórmula básica de actualización es la siguiente:

\[
w := w - \alpha \cdot \frac{\partial J(w, b)}{\partial w}
\]

Donde:
- \(w\): Parámetro a ajustar.
- \(\alpha\): Tasa de aprendizaje, un número pequeño (e.g., 0.01).
- \(\frac{\partial J(w, b)}{\partial w}\): Derivada parcial de la función de costo respecto a \(w\).

Notas sobre el Operador de Asignación

En la implementación del algoritmo, utilizamos el signo igual \(=\) como operador de asignación, que significa tomar un valor y guardarlo en una variable. Esto es diferente de su uso en matemáticas, donde \(=\) representa una afirmación de verdad. En el código, es importante distinguir entre la asignación y la comprobación de igualdad (en lenguajes como Python, la comprobación de igualdad es \(==\)).

Desempaquetando los Componentes

Tasa de Aprendizaje \(\alpha\)

La tasa de aprendizaje \(\alpha\) controla el tamaño de los pasos hacia abajo en la función de costo:
- **Valores grandes** de \(\alpha\) provocan pasos grandes, lo que puede acelerar la convergencia, pero también aumentar el riesgo de sobrepasar el mínimo.
- **Valores pequeños** permiten pasos más controlados pero más lentos.

La elección de \(\alpha\) es crucial para el buen rendimiento del algoritmo.

Derivada de la Función de Costo

El término \(\frac{\partial J(w, b)}{\partial w}\) indica la dirección de ajuste de \(w\). Aunque este término se deriva del cálculo, puedes entenderlo como la dirección en la que debe moverse \(w\) para reducir \(J(w, b)\).

Actualización Simultánea de Parámetros

El algoritmo de descenso de gradiente requiere que los parámetros \(w\) y \(b\) se actualicen **simultáneamente**. La implementación correcta del descenso de gradiente se realiza de esta forma:

1. Calculamos las expresiones para los nuevos valores de \(w\) y \(b\), almacenándolos temporalmente.
2. Actualizamos ambos parámetros al mismo tiempo utilizando los valores temporales.

En este caso, el valor de \(w\) utilizado para actualizar \(b\) es diferente al valor inicial, lo cual altera el resultado y la precisión del algoritmo. Esta variante no es un verdadero descenso de gradiente, sino un método con propiedades diferentes.

Convergencia del Algoritmo

El proceso de actualización se repite hasta que los cambios en \(w\) y \(b\) son mínimos. En este punto, el algoritmo ha alcanzado un **mínimo local** de la función de costo.

Resumen

Implementar el descenso de gradiente de forma correcta implica:
- Usar una tasa de aprendizaje adecuada.
- Calcular y almacenar temporalmente los valores de actualización de \(w\) y \(b\).
- Actualizar ambos parámetros simultáneamente.

= Machine Learning Handbook: La Función de Costo en Regresión Lineal

La Función de Costo en la Regresión Lineal

Una parte fundamental de la implementación de la regresión lineal es definir la función de costo, la cual permite evaluar la calidad del modelo. En este tema, explicaremos qué es la función de costo y cómo se utiliza para mejorar el modelo.

1. Introducción
Para entrenar un modelo de regresión lineal, trabajamos con un conjunto de datos de entrenamiento que contiene pares de entrada `(x)` y salida `(y)`. El modelo utilizado para ajustar estos datos es la función lineal $f_{w, b}(x) = w \cdot x + b$, donde:

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
$$\hat{y}^{(i)} = f_{w, b}(x^{(i)}) = w \cdot x^{(i)} + b$$
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
Consideremos un modelo simplificado donde \( f_{w}(x) = w \cdot x \) (es decir, *b = 0*).

Para un conjunto de entrenamiento con puntos en las posiciones *(1,1), (2,2), (3,3)*, se puede observar cómo varía el error cuadrático medio al cambiar el valor de *w*:

1. Para \( w = 1 \):
- La línea de predicción coincide con todos los puntos de datos, lo cual genera un error *J(w) = 0*.

2. Para \( w = 0.5 \):
- La línea de predicción se aleja de los puntos, aumentando el error cuadrático.

Al visualizar *J(w)* en función de *w*, se observa que el valor mínimo de *J* ocurre cuando \( w = 1 \).

6. Minimización de la Función de Costo
El objetivo en regresión lineal es encontrar los valores de *w* y *b* que minimicen \( J(w, b) \), es decir, que minimicen el error entre las predicciones y los valores reales. Matemáticamente, esto se expresa como:
\[
$$\min_{w, b} J(w, b)$$
\]

Para encontrar los mejores parámetros, se utilizan algoritmos de optimización, como el descenso de gradiente, que permite ajustar iterativamente *w* y *b* hasta que \( J(w, b) \) alcance un valor mínimo.

Resumen
- La función de costo es esencial para medir el rendimiento de un modelo de regresión lineal.
- *J(w, b)* se minimiza ajustando *w* y *b* para que la línea generada se aproxime a los datos de entrenamiento.
- La función de costo por error cuadrático es ampliamente utilizada en problemas de regresión debido a su simplicidad y efectividad en la optimización de parámetros del modelo.

Machine Learning Handbook: Visualización de la Función de Coste

Introducción a la Visualización de la Función de Coste

El objetivo de la regresión lineal es encontrar los valores óptimos de los parámetros del modelo, \( w \) y \( b \), que minimicen la función de costo, \( J(w, b) \). Este proceso de optimización nos permite ajustar la función \( f(x) = w \cdot x + b \) para representar la relación entre las variables de entrada y salida de nuestro conjunto de entrenamiento.

La función de costo \( J(w, b) \) mide el error cuadrático promedio entre las predicciones del modelo y los valores reales, proporcionando una forma cuantitativa de evaluar el rendimiento del modelo.

Visualización de la Función de Coste en 2D y 3D

1. *Visualización en 2D (solo \( w \))*
   En una primera aproximación, al fijar \( b = 0 \), la función de costo se visualiza en 2D. En este caso, la gráfica de \( J(w) \) tiene la forma de una parábola o un "cuenco" orientado hacia arriba, indicando que el mínimo de la función está en el fondo de la curva.

   - *Ejemplo:* Si \( w = 0.06 \) y \( b = 50 \), la función \( f(x) = 0.06x + 50 \) se muestra en una gráfica como una línea que no ajusta bien el conjunto de datos sobre precios de viviendas. En este caso, \( J(w, b) \) será alto debido a la alta diferencia entre las predicciones y los valores reales.

2. *Visualización en 3D (con \( w \) y \( b \))*
   Cuando consideramos \( w \) y \( b \) como parámetros, la función de costo \( J(w, b) \) toma una forma en 3D, generando un gráfico de superficie similar a un "cuenco" extendido en tres dimensiones. En este gráfico, el eje vertical representa el valor de \( J(w, b) \), mientras que los ejes horizontales representan los valores de \( w \) y \( b \).

   - *Ejemplo Visual 3D:* Si \( w = -0.15 \) y \( b = 800 \), \( f(x) = -0.15 \cdot x + 800 \) produce una línea con pendiente negativa, lo cual no ajusta bien el conjunto de datos de precios de viviendas. Este punto se representa en la superficie 3D lejos del fondo, indicando un alto costo.

   - *Gráfico 3D:* La gráfica de superficie muestra que los valores altos de \( J(w, b) \) están alejados del mínimo en el centro del cuenco.

3. *Gráfica de Contornos (Mapa Topográfico de la Función de Coste)*
   La gráfica de contornos es una proyección 2D del gráfico de superficie 3D. En esta gráfica, los óvalos o elipses representan cortes horizontales del cuenco, donde cada curva conecta puntos de igual valor de \( J(w, b) \). El centro de las elipses indica el mínimo de la función de coste.

   - *Ejemplo de Interpretación de Contornos:*
     - Puntos en contornos exteriores representan valores altos de \( J(w, b) \) y, por ende, modelos con alta ineficacia en las predicciones.
     - El centro de las elipses corresponde al valor mínimo de \( J(w, b) \), que representa el mejor ajuste de los datos.

Ejemplos de Parámetros y Ajuste de Modelo

Se exploran diferentes valores de \( w \) y \( b \) para observar cómo afectan el ajuste de la línea \( f(x) \) al conjunto de datos de precios de viviendas:

1. *\( w = -0.15, b = 800 \)*:
   - *Función:* \( f(x) = -0.15 \cdot x + 800 \)
   - *Visualización:* La línea cruza el eje vertical en 800 y tiene una pendiente negativa.
   - *Costo:* Alto, debido a que la línea no se ajusta bien a los datos.

2. *\( w = 0, b = 360 \)*:
   - *Función:* \( f(x) = 0 \cdot x + 360 \) (línea plana).
   - *Visualización:* La línea es horizontal y corta el eje vertical en 360.
   - *Costo:* Aún alto, aunque menor en comparación con el ejemplo anterior.

3. *Valores cercanos al mínimo de \( J(w, b) \)*:
   - *Función:* Se obtiene una línea de mejor ajuste para el conjunto de datos, con el costo mínimo posible.
   - *Visualización:* La línea pasa cerca de los puntos de datos, minimizando la distancia vertical entre las predicciones y los valores reales.

Algoritmo de Optimización: Descenso de Gradiente

Para encontrar el valor óptimo de los parámetros \( w \) y \( b \), se emplea el *Descenso de Gradiente*, un algoritmo que permite reducir iterativamente el costo moviéndose hacia el mínimo en la gráfica de \( J(w, b) \).

1. *Proceso:* El descenso de gradiente ajusta \( w \) y \( b \) paso a paso en la dirección que reduce más rápidamente el valor de \( J(w, b) \).
2. *Aplicación:* Este método es crucial no solo para la regresión lineal, sino también para modelos avanzados de IA y aprendizaje profundo.

El descenso de gradiente representa una herramienta esencial para la optimización de modelos, y en el laboratorio opcional puedes experimentar visualmente cómo se modifica \( J(w, b) \) con cada iteración del algoritmo.

Handbook de Machine Learning: Función de Costo en Regresión Lineal

1. Introducción a la Función de Costo

La *función de costo* en machine learning es una métrica que indica qué tan bien un modelo predice el valor de salida deseado. En el caso de *regresión lineal*, el objetivo es encontrar los parámetros óptimos de la función lineal para minimizar la diferencia entre las predicciones y los valores reales en el conjunto de entrenamiento.

- Dada la función lineal:
  $$f_{w, b}(x) = w \cdot x + b$$
  Donde:
  - $w$ y $b$ son los *parámetros* o *coeficientes* del modelo que se ajustan durante el entrenamiento.
  - $x$ representa las *características de entrada*.
  - $f_{w, b}(x)$ es la *predicción* de la salida.

Ejemplos de $f(x)$ con diferentes valores de $w$ y $b$:

1. *Caso 1*: $w = 0$, $b = 1.5$
   - La función $f(x) = 0 \cdot x + 1.5 = 1.5$ produce una *línea horizontal* a $y = 1.5$.

2. *Caso 2*: $w = 0.5$, $b = 0$
   - La función $f(x) = 0.5 \cdot x$ tiene una *pendiente* de $0.5$ y *pasa por el origen*.

3. *Caso 3*: $w = 0.5$, $b = 1$
   - La función $f(x) = 0.5 \cdot x + 1$ cruza el eje y en $1$ y tiene una pendiente de $0.5$.

2. Definición de la Función de Costo

Para elegir los valores óptimos de $w$ y $b$, se necesita una métrica que evalúe el error de predicción en todos los ejemplos de entrenamiento. Esta métrica es la *función de costo*.

Error Cuadrático

Para un ejemplo de entrenamiento $i$, el error se define como la diferencia entre la predicción y el valor objetivo:

$$
\text{error} = \hat{y} - y
$$

Donde:
- $\hat{y}$ es la predicción de la salida para una entrada $x$ dada.
- $y$ es el valor objetivo real.

Error Cuadrático Medio (MSE)

La función de costo total calcula el error cuadrático medio en todo el conjunto de entrenamiento:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^m \left( f_{w, b}(x^{(i)}) - y^{(i)} \right)^2
$$

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

---

Resumen

- La *función de costo* mide la precisión del modelo.
- En *regresión lineal*, $J(w, b)$ mide el error cuadrático medio de las predicciones.
- Optimizar los valores de $w$ y $b$ para *minimizar $J(w, b)$* ayuda a mejorar el rendimiento del modelo en la tarea de predicción.

Para comprender la función de costo más a fondo, revisar cómo cambiar $w$ y $b$ afecta la forma de la línea y el valor de $J(w, b)$ en tus datos de entrenamiento.

= Handbook: Machine Learning - Gradiente Descendente para Regresión Lineal

# Introducción
En este capítulo, exploraremos el proceso de entrenamiento de un modelo de regresión lineal usando la **función de coste por error cuadrático** y el **gradiente descendente**. Revisaremos las ecuaciones clave, el proceso de cálculo de las derivadas, y el funcionamiento del algoritmo de descenso en gradiente.

# Modelo de Regresión Lineal
La regresión lineal es un modelo de predicción que intenta ajustar una línea recta a los datos para predecir valores. Para ajustar el modelo a los datos, usaremos el gradiente descendente para minimizar el error.

- **Función de coste (J)**: Mide el error promedio entre las predicciones del modelo y los valores reales. Se define como:
  J(w, b) = (1 / 2m) * Σ(i=1 to m) (yᵢ - (wxᵢ + b))²

- **Modelo de predicción (f)**: La predicción para cada dato es:
  f(xᵢ) = wxᵢ + b

# Algoritmo de Descenso de Gradiente
El objetivo del descenso de gradiente es minimizar la función de coste actualizando los parámetros w y b en cada iteración, calculando las derivadas parciales de J con respecto a estos parámetros.

1. **Derivada de J con respecto a w**:
   ∂J/∂w = (1/m) * Σ(i=1 to m) ((wxᵢ + b) - yᵢ) * xᵢ

2. **Derivada de J con respecto a b**:
   ∂J/∂b = (1/m) * Σ(i=1 to m) ((wxᵢ + b) - yᵢ)

Usando estas derivadas, el algoritmo actualiza los valores de w y b en cada paso para minimizar J.

- **Convergencia**: Repetimos las actualizaciones hasta que el cambio en J sea mínimo o hasta alcanzar un número máximo de iteraciones.

# Propiedades del Gradiente Descendente en Regresión Lineal
1. **Convexidad**: La función de coste por error cuadrático es convexa, lo cual significa que tiene un único mínimo global.
2. **Convergencia**: Si α es adecuada, el algoritmo siempre convergerá al mínimo global.

# Visualización del Proceso de Gradiente Descendente
Para observar el gradiente descendente en acción:
- **Gráfica del modelo**: La línea recta de predicción se ajusta progresivamente a los datos.
- **Función de coste**: Visualizamos el descenso en J a medida que se reduce el error.

# Tipos de Gradiente Descendente
Este proceso es conocido como **descenso de gradiente por lotes**, ya que usa todo el conjunto de entrenamiento en cada paso. Existen variaciones como el **descenso de gradiente estocástico** y **mini-lote**, que operan en subconjuntos del conjunto de datos en cada actualización.

# Conclusión
¡Felicidades! Has completado la implementación del descenso de gradiente para regresión lineal. Ahora tienes los conocimientos básicos para construir un modelo de regresión que puede ser aplicado en predicciones de precios de viviendas, entre otras aplicaciones.

= Ejercicio Opcional
Repasa el algoritmo de descenso de gradiente y experimenta con diferentes tasas de aprendizaje en el código proporcionado. Observa cómo la función de coste disminuye y cómo la línea de ajuste mejora en cada iteración.

= Resumen
- **Función de coste**: Cuantifica el error.
- **Gradiente Descendente**: Algoritmo de optimización para minimizar la función de coste.
- **Aplicación Práctica**: Predicción de valores como precios de viviendas.

== Handbook: Machine Learning - Elección del Learning Rate y Funcionamiento del Gradient Descent

Introducción
En este documento, exploraremos cómo la elección de la tasa de aprendizaje (alpha) impacta la eficiencia del algoritmo de descenso de gradiente. Discutiremos los efectos de elegir una tasa de aprendizaje demasiado pequeña o demasiado grande y cómo estos casos pueden afectar la convergencia al mínimo de la función de coste.

Sección 1: Descenso de Gradiente
El algoritmo de descenso de gradiente se utiliza para minimizar una función de coste \( J(w) \), ajustando iterativamente los parámetros \( w \) de la siguiente forma:
\[
w = w - \alpha \cdot \nabla J(w)
\]
donde \( \alpha \) es la tasa de aprendizaje.

Impacto de la Tasa de Aprendizaje
La tasa de aprendizaje, o \( \alpha \), determina el tamaño del paso en cada iteración del algoritmo. Dependiendo de su valor, los pasos pueden ser:
- Demasiado pequeños, lo que resulta en un proceso de convergencia muy lento.
- Demasiado grandes, lo que podría hacer que el algoritmo no converja y diverja.

Sección 2: Efecto de una Tasa de Aprendizaje Muy Pequeña
Si \( \alpha \) es demasiado pequeño, el algoritmo dará pasos muy pequeños, como se ilustra en la siguiente gráfica:

// Aquí podrías agregar una ilustración de los pasos pequeños en Typst o insertar una imagen.

\[
\text{Paso 1}: w = w - 0.000001 \cdot \nabla J(w)
\]

Esto provoca una reducción lenta en \( J(w) \), requiriendo muchas iteraciones para alcanzar un mínimo. En resumen, una tasa de aprendizaje baja:
- Reduce el coste \( J \), pero de manera muy lenta.
- Aumenta el tiempo de convergencia al mínimo local o global.

Sección 3: Efecto de una Tasa de Aprendizaje Muy Grande
Cuando \( \alpha \) es demasiado grande, el descenso de gradiente puede "sobrepasar" el mínimo y oscilar entre valores sin llegar a converger.

Ejemplo:
\[
w{\text{nueva}} = w - 10 \cdot \nabla J(w)
\]

Esto puede resultar en un aumento del coste \( J(w) \) en lugar de una disminución, como se muestra en la siguiente gráfica:

// Inserta una ilustración de oscilación o una imagen que represente la divergencia.

Resumen de una tasa de aprendizaje alta:
- Produce oscilaciones y aumenta el coste en lugar de reducirlo.
- Puede llevar a una divergencia en lugar de una convergencia.

Sección 4: Convergencia y Mínimos Locales
Si los parámetros \( w \) llegan a un mínimo local, la derivada se vuelve cero:
\[
\frac{\partial J}{\partial w} = 0
\]
En este caso, \( w \) permanece inalterado, y el algoritmo reconoce que ha alcanzado un mínimo, ya sea local o global.

Al acercarse a un mínimo, el descenso de gradiente automáticamente ajusta el tamaño de los pasos, haciéndolos más pequeños debido a la reducción de la pendiente de la función de coste.

Sección 5: Conclusión
La tasa de aprendizaje \( \alpha \) es fundamental para el rendimiento del descenso de gradiente. Un buen valor de \( \alpha \) permite un equilibrio entre velocidad y precisión en la convergencia.

== Ejemplo Simplificado del Algoritmo de Descenso por Gradiente

Este handbook proporciona una introducción detallada al descenso por gradiente, un método esencial en el aprendizaje automático para optimizar modelos. En este ejemplo, analizaremos cómo se utiliza el descenso por gradiente para ajustar un parámetro (`w`) en función de una tasa de aprendizaje (`α`) y una derivada parcial de la función de coste.

1. Fundamentos del Descenso por Gradiente

El descenso por gradiente es un algoritmo iterativo que busca minimizar una función de coste `J(w)`. En este proceso:
- `w` es el parámetro que se actualiza en cada iteración.
- `α` es la *tasa de aprendizaje*, que determina el tamaño del paso en cada actualización.
- `∂J/∂w` es la *derivada parcial de la función de coste* con respecto a `w`, que indica la dirección del cambio para minimizar `J`.

2. Definición de Tasa de Aprendizaje (α)

La tasa de aprendizaje `α` controla la velocidad del ajuste de `w`. Si `α` es demasiado grande, el algoritmo puede oscilar y no converger; si es demasiado pequeña, la convergencia será lenta.

- *Tasa de aprendizaje alta*: Resulta en cambios significativos de `w`, pero podría omitir el mínimo.
- *Tasa de aprendizaje baja*: Permite acercarse al mínimo de manera gradual.

3. Actualización del Parámetro w

La actualización de `w` se calcula como:
\[
w = w - α \cdot \frac{∂J}{∂w}
\]
Donde:
- `w` se ajusta en la dirección que reduce `J`.
- La dirección se determina en función de la pendiente en el punto actual de `w`.

4. Ejemplo Visual de Descenso por Gradiente

Caso 1: Punto Inicial a la Derecha del Mínimo
1. Se comienza con un valor inicial de `w` ubicado en la parte derecha de la gráfica de `J(w)`.
2. La *pendiente* en este punto es positiva (`∂J/∂w > 0`).
3. *Resultado*: `w` se actualiza hacia la izquierda, acercándose al mínimo.

Caso 2: Punto Inicial a la Izquierda del Mínimo
1. El valor inicial de `w` se encuentra a la izquierda del mínimo.
2. La *pendiente* es negativa (`∂J/∂w < 0`).
3. *Resultado*: `w` aumenta, moviéndose hacia la derecha y acercándose al mínimo.

5. ¿Por Qué es Útil la Derivada?

La derivada `∂J/∂w` nos indica la dirección en la que debemos mover `w` para minimizar `J`. Este movimiento depende de la pendiente:
- *Pendiente positiva*: `w` disminuye.
- *Pendiente negativa*: `w` aumenta.

Este ajuste iterativo permite al algoritmo de aprendizaje alcanzar un valor de `w` que minimiza `J`, optimizando así el modelo.

6. Elección de la Tasa de Aprendizaje (α)

Seleccionar un valor adecuado para `α` es crítico:
- Un `α` pequeño puede llevar a una convergencia lenta.
- Un `α` grande puede resultar en inestabilidad.

En la siguiente sección exploraremos cómo elegir `α` para maximizar la eficiencia del descenso por gradiente.

---

*Conclusión:* El descenso por gradiente es un proceso iterativo y dirigido que permite ajustar parámetros de un modelo en función de su error. Este handbook ha cubierto los aspectos básicos y la lógica detrás de la actualización de `w` y el papel de `α`.

= Machine Learning Handbook
== Introducción al Descenso de Gradiente

Este handbook cubre los conceptos básicos de descenso de gradiente, un algoritmo fundamental para minimizar funciones de coste en machine learning, como la de regresión lineal, y en modelos más complejos como redes neuronales.

=== Función de Coste y Parámetros

En machine learning, queremos minimizar una función de coste \( J(w, b) \) que mide qué tan bien nuestros parámetros \( w \) y \( b \) ajustan los datos. En este contexto:

- *\( w \)*: Representa los coeficientes (pesos) de las características.
- *\( b \)*: Representa el sesgo (bias) o término independiente.

Para lograr el menor coste posible, necesitamos encontrar valores óptimos para \( w \) y \( b \).

=== Introducción al Algoritmo de Descenso de Gradiente

El algoritmo de descenso de gradiente proporciona una forma sistemática de encontrar los valores de \( w \) y \( b \) que minimizan la función de coste \( J(w, b) \). Este algoritmo es ampliamente utilizado en machine learning y deep learning.

Ejemplo de Regresión Lineal

En regresión lineal, buscamos minimizar una función de coste que toma la forma de un parábola en 3D, donde los ejes representan \( w \), \( b \) y \( J(w, b) \). La forma de "hamaca" o "arco" garantiza que hay un único mínimo global, el punto más bajo de esta superficie.

Idea Intuitiva: Analogía de la Colina

Imaginemos la superficie de \( J(w, b) \) como un campo de golf con colinas y valles. Si te colocas en un punto en la colina, puedes imaginarte mirando alrededor para decidir el camino de descenso más rápido.

- *Descenso de Gradiente*: Te ayuda a determinar la dirección en la que debes moverte en cada paso para bajar lo más rápido posible.
- *Iteración*: Desde cada nuevo punto alcanzado, repites el proceso para seguir descendiendo.

Pasos de Ejemplo

1. Comienza en un punto aleatorio en la colina.
2. Observa el gradiente (pendiente) en cada dirección.
3. Da un pequeño paso en la dirección con pendiente más pronunciada hacia abajo.
4. Repite hasta llegar a un valle, o mínimo local.

=== Mínimos Locales y el Problema de los Múltiples Valles

Para funciones más complejas (por ejemplo, funciones de coste en redes neuronales), la superficie de la función de coste puede tener varios "valles". Estos puntos bajos en la superficie se llaman *mínimos locales*. El descenso de gradiente puede detenerse en cualquier mínimo local dependiendo de los valores iniciales.

Ejemplo de Mínimos Locales

Supongamos que:

- Iniciamos en un punto de la colina y llegamos al primer valle tras varios pasos.
- Cambiamos nuestro punto de inicio a un punto cercano en la superficie.

Si comenzamos en el nuevo punto, es posible que descendamos a un valle diferente, llegando a un segundo mínimo local.

Consecuencias

Esto significa que el punto de inicio afecta el mínimo alcanzado y que no siempre se garantiza encontrar el mínimo global en funciones de coste complejas.

=== Descripción Matemática Básica

Para implementar el descenso de gradiente, necesitamos calcular las derivadas parciales de \( J(w, b) \) con respecto a \( w \) y \( b \). Estas derivadas nos indican la dirección del gradiente. Con esto:

1. *Derivada de \( J \) respecto a \( w \) y \( b \)*: Nos da la pendiente en cada dirección.
2. *Actualizar \( w \) y \( b \)*: Cada parámetro se ajusta en la dirección opuesta al gradiente para reducir \( J(w, b) \).

Al continuar este proceso, podemos acercarnos al valor de \( w \) y \( b \) que minimiza la función de coste.

== Fórmulas

- $ w := w - \u{03B1} \cdot math.frac(∂J(w, b), ∂w) $
- $ b := b - \u{03B1} \cdot math.frac(∂J(w, b), ∂b) $


Donde \( \u{03B1} \) es la tasa de aprendizaje, controlando el tamaño del paso en cada iteración.

== Resumen

El descenso de gradiente es una herramienta esencial en machine learning para encontrar mínimos de funciones de coste, optimizando modelos de regresión y redes neuronales, entre otros. Al ajustar los parámetros en la dirección del gradiente negativo, podemos minimizar el coste y mejorar el rendimiento del modelo.

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

= Feature Scaling

== Feature Engineering and its Importance

When designing a machine learning model, the choice of features can significantly impact the model's performance. Selecting or designing the right features is often a critical step in improving the accuracy of predictions. Below are the key takeaways from the example discussed about predicting house prices:

=== Example: Predicting House Prices

Let's assume we have two initial features for a house:

#simpleTable(
  columns: (1fr, 2fr),
  [*Feature*], [*Description*],
  [$x_1$], [The width of the lot (frontage of the property)],
  [$x_2$], [The depth of the lot (rectangular assumption)],
)

A simple model can be expressed as:

$ f_(w,b)(x) = w_1 x_1 + w_2 x_2 + b $

While this model may work, there is an alternative approach that leverages domain knowledge.

=== Creating a New Feature: Lot Area

Recognizing that the *area* of the lot (width × depth) may better predict the house price, we define a new feature:

$ x_3 = x_1 times x_2 $

The updated model now incorporates this additional feature:

$ f_(w,b)(x) = w_1 x_1 + w_2 x_2 + w_3 x_3 + b $

Where:

- $w_1$, $w_2$, and $w_3$ are weights the model can optimize.
- The new feature, $x_3$, allows the model to consider whether frontage, depth, or lot area is most predictive.

This process of creating new features is called *feature engineering*. By transforming or combining existing features, we can enable the model to make more accurate predictions.

=== Benefits of Feature Engineering

1. *Leverages domain knowledge*:
   By incorporating intuition about the problem (e.g., lot area is a stronger predictor), we can design more effective features.

2. *Improves model performance*:
   New features can capture relationships that may not be apparent with the original set of features.

3. *Enables flexibility in modeling*:
   Allows fitting non-linear relationships, not just straight lines, by introducing new combinations or transformations of features.

=== Engineering Features for Non-Linear Data

Feature engineering is not limited to linear models. By creating new features or applying non-linear transformations, we can allow models to fit curves or other complex relationships in the data.

*Note for the next lesson*: The subsequent video will explore how to incorporate non-linear functions into models effectively.

= Feature Scaling

== Notes on Feature Scaling and Gradient Descent

=== Overview

Feature scaling is a technique to make gradient descent converge faster by transforming features to take on comparable ranges of values. This is particularly useful when features have very different scales.

=== Example: Predicting House Prices

We have a dataset with two features:
- $x_1$: Size of the house (ranges from 300 to 2000 square feet)
- $x_2$: Number of bedrooms (ranges from 0 to 5)

For an example house:
- Size: 2000 sq. ft.
- Bedrooms: 5
- Price: $500,000$

==== Parameter Selection
1. Initial guess:
    - $w_1 = 50$, $w_2 = 0.1$, $b = 50$
    - Predicted price: $50 \cdot 2000 + 0.1 \cdot 5 + 50 = 100,050,000$, which is far off.
2. Improved guess:
    - $w_1 = 0.1$, $w_2 = 50$, $b = 50$
    - Predicted price: $0.1 \cdot 2000 + 50 \cdot 5 + 50 = 500,000$, a much better estimate.

==== Observation
- Large feature ranges (e.g., $x_1$) often correspond to smaller parameter values (e.g., $w_1 = 0.1$).
- Small feature ranges (e.g., $x_2$) often correspond to larger parameter values (e.g., $w_2 = 50$).

=== Gradient Descent and Contours
When feature scales differ significantly:
1. The cost function contours become elongated ellipses.
2. Gradient descent takes inefficient "zig-zag" paths to reach the global minimum.

By scaling features to similar ranges, the contours become more circular, allowing gradient descent to converge faster.

=== Scaling Techniques

==== Min-Max Scaling
Transform each feature by dividing by its range:
- $x_1 = frac{x_1}{2000}$ (for range $[300, 2000]$)
- $x_2 = frac{x_2}{5}$ (for range $[0, 5]$)

Result:
- Scaled $x_1$ ranges from $0.15$ to $1$
- Scaled $x_2$ ranges from $0$ to $1$

==== Standardization (Z-score Scaling)
Transform each feature to have a mean of 0 and a standard deviation of 1:
\[
z_i = \frac{x_i - \mu_i}{\sigma_i}
\]
Where:
- $\mu_i$: Mean of the feature
- $sigma_i$: Standard deviation of the feature

==== Implementation
#codeBlock(
  ```python
  import numpy as np

  def min_max_scaling(feature, feature_range):
      return feature / feature_range

  def z_score_scaling(feature, mean, std_dev):
      return (feature - mean) / std_dev

  # Example for x1 and x2
  x1 = np.array([300, 2000])
  x2 = np.array([0, 5])

  x1_scaled = min_max_scaling(x1, 2000)
  x2_scaled = min_max_scaling(x2, 5)

  print(f"Scaled x1: {x1_scaled}")
  print(f"Scaled x2: {x2_scaled}")
  ```
)

=== Key Takeaway
Feature scaling significantly improves the efficiency of gradient descent, especially when feature ranges differ greatly. Use either min-max scaling or standardization to normalize features before applying machine learning algorithms.

= Gradient Descent for Multiple Linear Regression

== Multiple Linear Regression Overview

In multiple linear regression, we generalize the simple linear regression to work with multiple features. Using vectorized notation, the model can be expressed as:

#simpleTable(
  columns: (1fr, 1fr),
  [*Attribute*], [*Formula*],
  [*Model*], [
    $ f_(w,b)(x) = w^T x + b $
  ],
  [*Parameters*], [
    $ w \in bb{R}^n, \ b \in bb{R} $
  ],
  [*Cost Function*], [
    $ J(w, b) = frac{1}{2m} sum_{i=1}^m (f_(w,b)(x^(i)) - y^(i))^2 $
  ],
  [*Objective*], [
    $ \min_{w,b} J(w, b) $
  ],
)

In this model:
- $w$ is a vector of weights for the features.
- $b$ is the bias term (a scalar).
- $x$ is a feature vector for each training example.

== Gradient Descent for Multiple Features

Gradient descent minimizes the cost function $J(w, b)$ iteratively by updating the parameters $w$ and $b$. The update rules are:

#simpleTable(
  columns: (1fr, 1fr),
  [*Parameter*], [*Update Rule*],
  [$w_j$], [
    $ w_j := w_j - alpha frac{partial J(w, b)}{partial w_j} $
  ],
  [$b$], [
    $ b := b - alpha frac{partial J(w, b)}{partial b} $
  ],
)

The partial derivatives for the parameters are computed as:

#simpleTable(
  columns: (1fr, 1fr),
  [*Derivative*], [*Formula*],
  [$frac{partial J(w, b)}{partial w_j}$], [
    $ frac{1}{m} sum_{i=1}^m lr( f_(w,b)(x^(i)) - y^(i) lr) x_j^(i) $
  ],
  [$frac{partial J(w, b)}{partial b}$], [
    $ frac{1}{m} sum_{i=1}^m lr( f_(w,b)(x^(i)) - y^(i) lr) $
  ],
)

=== Vectorized Gradient Descent

Using vectorized computation, we can compute updates efficiently:

#codeBlock(
  ```python
    import numpy as np

    def gradient_descent(X, y, w, b, alpha, num_iters):
      m = len(y)
      for i in range(num_iters):
      predictions = np.dot(X, w) + b
      errors = predictions - y
      grad_w = (1/m) * np.dot(X.T, errors)
      grad_b = (1/m) * np.sum(errors)
      w -= alpha * grad_w
      b -= alpha * grad_b
    return w, b
    ```
)

=== Example

For a dataset with features $X \in bb{R}^{m times n}$ and target values $y \in bb{R}^m$, initialize $w$ and $b$:

#codeBlock(
  ```python
# Example data
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([4, 6, 8])
w = np.zeros(X.shape[1])
b = 0
alpha = 0.01
num_iters = 1000

# Perform gradient descent
w, b = gradient_descent(X, y, w, b, alpha, num_iters)
print(f"Weights: {w}, Bias: {b}")
  ```
)

=== Notes on Convergence

1. Learning rate ($alpha$):
   - If $alpha$ is too small, convergence will be slow.
   - If $alpha$ is too large, the algorithm might diverge.
2. Vectorization significantly speeds up computation by eliminating loops.

== Alternative: Normal Equation

For multiple linear regression, an alternative to gradient descent is the *normal equation*:

$ w = (X^T X)^{-1} X^T y $

Advantages:
- No iterative process is required.
- Provides a direct solution for $w$ and $b$.

Disadvantages:
- Computationally expensive for large feature sets.
- Not applicable to most machine learning algorithms (e.g., logistic regression, neural networks).

= Linear Regression with Multiple Features

== Introduction

- *Objective*: Extend linear regression to handle multiple features (not just one).
- *Example*: In predicting housing prices, consider features like size, number of bedrooms, floors, and age.
- This approach provides more information, potentially improving prediction accuracy.

== Notation & Terminology

- Let the features be $X_1, X_2, X_3, \u{00B7}, X_n$, where each $X_j$ represents a different characteristic.
  - *Example*: For a house, $X_1 =$ size, $X_2 =$ bedrooms, $X_3 =$ floors, $X_4 =$ age.
- Define $X^{(i)}$ as the vector of features for the $i$-th training example. For instance, if $X^{(2)} = (1416, 3, 2, 40)$, it represents the feature values for the second example.
- *Dimensions*:
  - $X_j$: a single feature across all training examples.
  - $X^{(i)}$: all features for a single training example (a vector).

== Model Definition

With multiple features, the linear regression model is expressed as:

#simpleTable(
  columns: (1fr, 1fr),
  [*Attribute*], [*Formula*],
  [*Model*], [
    $ f(w, b)(X) = w_1 X_1 + w_2 X_2 + \u{00B7} + w_n X_n + b $
  ],
  [*Parameters*], [
    $ w = (w_1, w_2, \u{00B7}, w_n) $, $ b $
  ],
)

== Example: Housing Price Prediction Model

Consider a model to predict housing prices:

- Model formula:
  $ f(w, b)(X) = 0.1 X_1 + 4 X_2 + 10 X_3 - 2 X_4 + 80 $

Interpretations of parameters:
- $ w_1 = 0.1 $: For each additional square foot, the price increases by $0.1 \u{00D7} 1000 = 100$ dollars.
- $ w_2 = 4 $: Each additional bedroom adds $4000$ dollars.
- $ w_3 = 10 $: Each additional floor adds $10000$ dollars.
- $ w_4 = -2 $: Each additional year of age decreases the price by $2000$ dollars.
- $ b = 80 $: Base price (in thousands), assuming zero for all features.

== Matrix Notation

To simplify, we can represent the model using vectors:

- Define:
  - $ W = [w_1, w_2, \u{00B7}, w_n] $ (parameter vector).
  - $ X^{(i)} = [X_1^{(i)}, X_2^{(i)}, \u{00B7}, X_n^{(i)}] $ (feature vector for example $i$).
- Model becomes:
  $ f(W, b)(X^{(i)}) = W \cdot X^{(i)} + b $

This compact form allows easier manipulation and generalization when working with multiple features.

== Cost Function for Multiple Features

The cost function measures the model's accuracy by computing the error between predictions and actual values.

- For $m$ training examples, cost $J(W, b)$ is:

  $ J(W, b) = frac(1,2m) \\sum_{i=1}^{m} (f(W, b)(X^{(i)}) - y^{(i)})^2 $

This function remains convex, ensuring a global minimum, making it suitable for gradient descent optimization.

#codeBlock(
  ```python
  # Python example: Cost function implementation for multiple features
  def compute_cost(X, y, W, b):
      m = len(y)
      cost = np.sum((X @ W + b - y) * 2) / (2 * m)
      return cost
  ```
)

== Summary

- *Multiple features* enable a richer model by incorporating more information, improving predictions.
- *Matrix/vector notation* simplifies representation and calculation, making it efficient to handle large feature sets.

= Polynomial Regression

== Introduction

Polynomial regression extends linear regression by allowing the model to fit non-linear relationships in the data. Instead of fitting a straight line, we use polynomial functions to create curves that can better capture the patterns in the dataset. This is particularly useful when the data exhibits non-linear trends.

=== Why Polynomial Regression?

Suppose we have a dataset of housing prices where the feature $x$ represents the size of the house in square feet. A straight line might not capture the pattern well, as housing prices often increase non-linearly with size. Instead, we can try fitting a polynomial function like:

1. **Quadratic**: $f(x) = w_1 x + w_2 x^2 + b$
2. **Cubic**: $f(x) = w_1 x + w_2 x^2 + w_3 x^3 + b$

Each additional term allows the model to capture more complexity in the data.

== Polynomial Features

To create polynomial regression models, we generate polynomial features by raising the input feature $x$ to higher powers. For example:
- $x$
- $x^2$
- $x^3$

These features allow us to fit curves to the data.

#simpleTable(
  columns: (1fr, 1fr),
  [*Feature*], [*Range*],
  [$x$], [$1$ to $1000$],
  [$x^2$], [$1$ to $1,000,000$],
  [$x^3$], [$1$ to $1,000,000,000$],
)

*Note*: The range of values increases dramatically with higher powers, which makes feature scaling crucial for gradient descent.

=== Feature Scaling

If the range of feature values varies significantly, gradient descent may converge very slowly or fail entirely. Therefore, it is essential to scale the features to comparable ranges.

== Alternative Polynomial Functions

Besides $x^2$ and $x^3$, other transformations like the square root of $x$ can also be used:
- Example: $f(x) = w_1 x + w_2 sqrt{x} + b$
- The square root function grows slower than higher powers, making it another useful transformation for certain datasets.

== Choosing the Right Function

Choosing which polynomial or alternative functions to include in your model depends on:
1. The dataset and its characteristics.
2. Experimenting with different functions and evaluating performance using cross-validation.

In the next courses, you'll learn systematic ways to decide on the best functions and models.

== Code Example: Implementing Polynomial Regression

Using Python and Scikit-learn, we can implement polynomial regression easily. Below is an example of how to generate polynomial features and fit a regression model:

#codeBlock(
```python
  import numpy as np
  from sklearn.preprocessing import PolynomialFeatures
  from sklearn.linear_model import LinearRegression
  from sklearn.pipeline import Pipeline

  # Generate dataset
  x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
  y = np.array([1.5, 3.2, 6.0, 10.1, 15.5])

  # Create a pipeline for polynomial regression
  degree = 3
  model = Pipeline([
      ('poly_features', PolynomialFeatures(degree=degree)),
      ('linear_regression', LinearRegression())
  ])

  # Fit the model
  model.fit(x, y)

  # Predict values
  x_pred = np.linspace(1, 5, 100).reshape(-1, 1)
  y_pred = model.predict(x_pred)
  print(f"Predicted values: {y_pred}")
```
)

This code demonstrates how polynomial regression can be implemented in practice using Scikit-learn's pipeline for feature generation and model fitting.

== Summary

- Polynomial regression generalizes linear regression to fit non-linear data.
- Feature scaling is critical when using higher-order polynomial terms.
- Tools like Scikit-learn make it easy to implement polynomial regression with just a few lines of code.
- Experimentation and evaluation are key to selecting the best model and transformations for your data.

= Multiple Functions in Linear Regression

== Vectorization in Machine Learning

=== Introduction

Vectorization is a key concept in efficient algorithm implementation. It reduces code length and leverages modern hardware like GPUs (Graphics Processing Units) for significant speedups. Here are the main ideas discussed:

- *Vectorization Benefits*:
  - Shortens code for easier readability and maintenance.
  - Accelerates execution by utilizing parallel hardware capabilities.

=== Example: Prediction Without Vectorization

Consider a linear regression model where:
- $ w $ is a vector of parameters.
- $ x $ is a vector of features.
- $ b $ is the bias term.

For $ n = 3 $, the prediction can be written mathematically as:
$ f(x) = \\sum_(j=1)^n w_j \cdot x_j + b $

==== Non-Vectorized Implementation

In Python, a straightforward implementation uses a loop:
#codeBlock(

```python
  f = 0
  for j in range(n):  = Loop from 0 to n-1
      f += w[j] ** x[j]
  f += b
  ```
)

This approach, while functional, becomes inefficient for large $ n $ (e.g., $ n = 100,000 $).

==== Issues:
- Repeated iterations increase computational cost.
- Code readability decreases for larger feature vectors.

=== Efficient Implementation with Vectorization

Using vectorized operations, the prediction formula becomes:
$ f(x) = w \cdot x + b $

==== Vectorized Python Implementation

With NumPy, we can replace the loop with a single line of code:
#codeBlock(

```python
  f = np.dot(w, x) + b  = Vectorized implementation
  ```
)

==== Explanation:
- `np.dot(w, x)` computes the dot product between $ w $ and $ x $.
- Adding $ b $ completes the prediction.

==== Advantages:
1. *Code Compactness*: Reduced to one line.
2. *Execution Speed*: Utilizes hardware parallelism, especially for large $ n $.

=== Behind the Scenes of Vectorization

Vectorized operations leverage optimized libraries (e.g., NumPy) to:
1. Utilize CPU or GPU parallelism for mathematical operations.
2. Avoid redundant computations compared to sequential loops.

==== Practical Comparison
When $ n $ is large, vectorized code is not just shorter but significantly faster. Experimentation shows dramatic performance improvements in execution time.

=== Summary of Key Points

#simpleTable(
  columns: (1fr, 1fr),
  [*Aspect*], [*Details*],
  [*Mathematical Formula*], [
    $ f(x) = sum(j=1)^n w_j \cdot x_j + b $
  ],
  [*Non-Vectorized*], [
    Python loop with `range(n)` iterating through $ w $ and $ x $.
  ],
  [*Vectorized*], [
    `np.dot(w, x) + b` utilizing NumPy for dot product computation.
  ],
  [*Benefits*], [
    Compact code, faster execution, and easier maintenance.
  ],
)

=== Observations and Reflections

- Vectorization feels like a "magic trick" due to its dramatic speedup.
- Understanding how hardware executes vectorized code adds to its appreciation.
- Practical takeaways include consistently seeking opportunities to write vectorized code for efficiency and scalability.

= Abordar el Sobreadaptación

=== Sobreadaptación en un modelo
Un modelo sobreajustado tiene un rendimiento muy bueno en los datos de entrenamiento pero no generaliza bien. Esto se puede observar cuando el modelo está ajustado demasiado a los datos, como en el caso de un modelo de predicción de precios de casas.

Existen varias estrategias para mitigar el sobreajuste:

=== 1. Obtener más datos de entrenamiento

Una de las herramientas más efectivas contra el sobreajuste es conseguir más datos de entrenamiento. Con más ejemplos, el modelo puede aprender patrones más representativos y generalizar mejor.

*Ejemplo: Si se tienen más datos sobre el tamaño y precio de las casas, el modelo podrá aprender una función menos variable.*

=== 2. Reducir la cantidad de funciones utilizadas

Otra estrategia es reducir la cantidad de características que se utilizan en el modelo. Si se usan demasiadas características, el modelo puede sobreajustarse a ellas.

*Ejemplo: Si el modelo tiene muchas características polinomiales, como \(x^2\), \(x^3\), etc., una forma de reducir el sobreajuste es eliminar algunas de estas funciones.*

En algunos casos, es útil elegir solo un subconjunto de características que consideres más relevantes, por ejemplo, el tamaño de la casa, el número de dormitorios, y la antigüedad de la casa.

*Selección de características:* La idea es elegir solo las características más útiles para predecir el resultado, lo que se llama *selección de características*. Sin embargo, esto puede ser problemático si algunas características realmente útiles se descartan.

=== 3. Regularización

La regularización es otra técnica importante para reducir el sobreajuste. En lugar de eliminar completamente algunas características, la regularización hace que los parámetros de esas características tengan un impacto menor.

*Ejemplo: Si tienes un modelo con muchas funciones polinomiales, la regularización reduce el valor de los parámetros asociados a estas funciones, evitando que tengan un impacto demasiado grande.*

La regularización fomenta que el modelo ajuste parámetros más pequeños, sin llegar a forzarlos a ser cero. Esto ayuda a que el modelo no dependa excesivamente de ninguna característica en particular, haciendo que generalice mejor.

*Nota: Por lo general, se realiza regularización sobre los parámetros \(w_1, w_2, w_3, \dots, w_n\). El parámetro \(b\) puede regularizarse si es necesario, pero no es obligatorio.*

=== Resumen

Las tres formas de abordar el sobreajuste son:

1. Obtener más datos de entrenamiento.
2. Reducir la cantidad de características utilizadas.
3. Usar regularización para reducir el impacto de los parámetros grandes.

Estas técnicas son fundamentales en la práctica, especialmente al trabajar con redes neuronales y otros algoritmos de aprendizaje automático.

=== Ejemplo con Regularización

Aquí mostramos un ejemplo de cómo la regularización afecta a un modelo polinomial.

#codeBlock(
```
python
  import numpy as np
  import matplotlib.pyplot as plt

  # Función de ejemplo: y = x^2 + 2x + 1
  def polynomial(x):
      return x**2 + 2*x + 1

  x = np.linspace(-10, 10, 100)
  y = polynomial(x)

  # Aplicar regularización (en este caso, hacer los parámetros más pequeños)
  def regularized_model(x, w1, w2, b):
      return w1 * x**2 + w2 * x + b

  w1, w2, b = 1, 2, 1
  regularized_y = regularized_model(x, w1, w2, b)

  # Graficar los resultados
  plt.plot(x, y, label="Modelo sin regularización")
  plt.plot(x, regularized_y, label="Modelo con regularización")
  plt.legend()
  plt.show()
```
)

= Logistic Regression Notes

== Cost Function for Logistic Regression

The cost function is a critical part of logistic regression as it helps to evaluate how well the chosen parameters ($w$ and $b$) fit the training data. Unlike linear regression, the quadratic cost function is not suitable for logistic regression because it leads to a non-convex function, which can result in convergence to local minima during gradient descent.

=== Why Quadratic Cost Function Fails

In linear regression, the cost function is defined as:

#simpleTable(
  columns: (1fr, 1fr),
  [*Attribute*], [*Formula*],
  [*Cost Function*], [
    $ J(w, b) = frac{1}{2m} sum_{i=1}^m lr( f_{w,b}(x^{(i)}) - y^{(i)} lr)^2 $
  ],
  [*Convexity*], [Convex (Bowl-shaped)],
)

However, for logistic regression, where the hypothesis function is:

$ f_{w,b}(x) = frac{1}{1 + e^{-w \cdot x - b}} $

Using the same quadratic cost function would result in a non-convex function with multiple local minima, making gradient descent unreliable.

#figure(
  image("./images/non-convex-cost.png"),
  caption: [
    Example of a non-convex cost function for logistic regression using quadratic cost.
  ]
)

=== Convex Cost Function for Logistic Regression

To address this, a new cost function is defined:

$ J(w, b) = frac{1}{m} sum_{i=1}^m L(f_{w,b}(x^{(i)}), y^{(i)}) $

Where $L$ is the loss function for a single training example, defined as:

- If $y = 1$:
  $ L(f_{w,b}(x), y) = -log(f_{w,b}(x)) $

- If $y = 0$:
  $ L(f_{w,b}(x), y) = -log(1 - f_{w,b}(x)) $

This formulation ensures that the cost function is convex, guaranteeing convergence to a global minimum using gradient descent.

=== Loss Function Visualization

==== Case 1: When $y = 1$
The loss function is $-log(f_{w,b}(x))$. The graph of this function shows:
- Loss approaches $0$ as $f_{w,b}(x) \to 1$.
- Loss increases sharply as $f_{w,b}(x) \to 0$.

==== Case 2: When $y = 0$
The loss function is $-log(1 - f_{w,b}(x))$. The graph of this function shows:
- Loss approaches $0$ as $f_{w,b}(x) \to 0$.
- Loss increases sharply as $f_{w,b}(x) \to 1$.


=== Combined Cost Function

The overall cost function combines the two cases:

$ J(w, b) = -frac{1}{m} sum_{i=1}^m lr[ y^{(i)} log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) log(1 - f_{w,b}(x^{(i)})) lr] $

This formulation penalizes incorrect predictions heavily, ensuring the model learns to predict probabilities closer to the actual labels.

=== Gradient Descent for Logistic Regression

To minimize $J(w, b)$, we use gradient descent. The partial derivatives of $J(w, b)$ with respect to $w$ and $b$ are derived from the above cost function:

$ frac{partial J(w, b)}{partial w} = frac{1}{m} sum_{i=1}^m lr( f_{w,b}(x^{(i)}) - y^{(i)} lr) x^{(i)} $

$ frac{partial J(w, b)}{partial b} = frac{1}{m} sum_{i=1}^m lr( f_{w,b}(x^{(i)}) - y^{(i)} lr) $

#codeBlock(
```
python
  def gradient_descent(X, y, w, b, alpha, num_iters):
      for i in range(num_iters):
          predictions = 1 / (1 + np.exp(-(X.dot(w) + b)))
          errors = predictions - y
          w -= alpha * (X.T.dot(errors) / len(y))
          b -= alpha * np.sum(errors) / len(y)
      return w, b
```
)

This iterative process ensures that $w$ and $b$ converge to values that minimize the cost function.

#figure(
  image("./images/logistic-gradient-descent.jpg"),
  caption: [
    Gradient descent converging to the global minimum of the convex cost function.
  ]
)

= Logistic Regression and Decision Boundaries

== Logistic Regression: Overview

Logistic regression is one of the most commonly used classification algorithms. Unlike linear regression, which predicts continuous values, logistic regression is designed to predict probabilities between 0 and 1. For example, it can be used to classify whether a tumor is malignant (1) or benign (0).

=== Sigmoid Function

The foundation of logistic regression is the sigmoid function, also known as the logistic function. Its formula is:

$ g(z) = frac{1}{1 + e^{-z}} $

Key properties of the sigmoid function:
- $ g(z) in (0, 1) $ for all real values of $z$.
- When $ z $ to $ = 0 $, $ g(z) = 0.5 $.
- As $ z $ to $-$ As $ z $ to $ -infinity $, $ g(z) $ to $ 0 $.

#figure(
  image("./images/sigmoid-curve.png"),
  caption: [
    The sigmoid function starts near 0, increases rapidly around $ z = 0 $, and approaches 1 asymptotically.
  ]
)

=== Logistic Regression Model

Logistic regression combines linear regression with the sigmoid function. The model is defined as:

$ f(x) = g(w*x + b) $

Where:
- $ w $: Weight (slope of the linear function).
- $ b $: Bias (intercept of the linear function).
- $ z = w*x + b $: Linear combination of the input features.

Interpretation:
- $ f(x) $ gives the probability that the output is 1 (positive class) for a given input $ x $.
- $ P(y=1|x) = f(x) $
- $ P(y=0|x) = 1 - f(x) $

=== Example: Tumor Classification

Imagine a dataset where:
- The input $ x $ is the tumor size.
- The output $ y $ is 1 for malignant tumors and 0 for benign ones.

The decision boundary is determined by the point where $ f(x) = 0.5 $:
- $ w*x + b = 0 $
- $ x = -frac{b}{w} $

#figure(
  image("./images/decision-boundary.png"),
  caption: [
    The decision boundary separates the classes based on the tumor size. Data points to the right of the boundary are classified as malignant.
  ]
)

=== Decision Boundary and Interpretation

The decision boundary is a threshold where the model predicts a 50% probability of belonging to either class. For example:
- If $ f(x) = 0.7 $, the model predicts a 70% chance that the tumor is malignant.
- The complement, $ 1 - f(x) $, represents the probability of the benign class (30% in this case).

=== Advantages of Logistic Regression

- **Probabilistic Output**: Unlike hard classifications, logistic regression provides probabilities, allowing better interpretation.
- **Simplicity**: Easy to implement and computationally efficient for binary classification problems.

=== Implementation in Python

The following Python code demonstrates how to compute the sigmoid function and use it for logistic regression:

#codeBlock(
```
python
  import numpy as np

  def sigmoid(z):
      return 1 / (1 + np.exp(-z))

  # Example: Compute probability
  weights = np.array([0.5])
  bias = -1.0
  tumor_size = np.array([2.0])  # Input feature
  z = np.dot(weights, tumor_size) + bias
  probability = sigmoid(z)

  print(f"Probability of malignancy: {probability:.2f}")
```
)

In this example:
- Weight $ w = 0.5 $.
- Bias $ b = -1.0 $.
- Input tumor size $$x = 2.0$$.

The computed probability indicates the likelihood that the tumor is malignant.

=== Key Takeaways

- Logistic regression maps linear combinations of inputs to probabilities using the sigmoid function.
- The decision boundary is where the model predicts equal probabilities for both classes.
- Logistic regression is a foundational algorithm for classification tasks and is interpretable and effective for many problems.

= Overfitting Problem

== Notes on Overfitting and Underfitting

=== Understanding the Problem

Overfitting occurs when a model is too closely aligned with the training data, capturing noise rather than the underlying pattern. This leads to poor generalization on new data.

Conversely, underfitting happens when the model fails to capture the patterns in the training data, resulting in poor performance both on the training set and unseen data.

=== Example: Predicting House Prices

Consider a dataset where the input feature $x$ represents the size of a house, and $y$ is the house price. Let's explore three cases:

#simpleTable(
  columns: (1fr, 1fr),
  [*Case*], [*Model Description*],
  [Linear Fit], [
    A linear regression model predicts $y = w_1 x + b$. It poorly fits the data due to its simplicity, resulting in *high bias*.
  ],
  [Quadratic Fit], [
    Adding $x^2$ as a feature, the model becomes $y = w_1 x + w_2 x^2 + b$. It better captures the pattern, balancing bias and variance.
  ],
  [Fourth-Order Polynomial], [
    A high-degree polynomial model, $y = w_1 x + w_2 x^2 + w_3 x^3 + w_4 x^4 + b$, fits the training data perfectly. However, it generalizes poorly due to *high variance*.
  ],
)

=== Visualization

#figure(
  image("./images/overfitting-examples.png"),
  caption: [
    Comparison of models: linear (underfit), quadratic (optimal fit), and fourth-order polynomial (overfit).
  ]
)

---

=== Definitions and Insights

==== Underfitting
- High bias: The model has a rigid assumption (e.g., linearity) and cannot adapt to the data.
- Example formula:
  $ y = w_1 x + b $

==== Overfitting
- High variance: The model memorizes the training data, resulting in erratic predictions for new inputs.
- Example formula:
  $ y = w_1 x + w_2 x^2 + w_3 x^3 + w_4 x^4 + b $
- Even a small change in the training set could lead to drastically different models.

==== Proper Fit
- The goal is to achieve a balance where the model captures the pattern but ignores noise.
- Intuition: A good model generalizes well to unseen data.

---

=== Key Concept: Generalization
Generalization refers to the model's ability to perform well on new, unseen data. It is essential for ensuring practical utility in machine learning applications.

---

=== Metrics for Evaluation

The following metrics are used to identify overfitting or underfitting:

#simpleTable(
  columns: (1fr, 1fr),
  [*Metric*], [*Description*],
  [Training Error], [
    The error on the training set. Overfitting occurs when this is very low, but test error is high.
  ],
  [Test Error], [
    The error on unseen data. High test error compared to training error indicates overfitting.
  ],
)

---

=== Balancing Bias and Variance
- High bias: Underfitting. The model is too simplistic.
- High variance: Overfitting. The model is too complex.

==== Regularization
A technique to prevent overfitting by penalizing overly complex models. Examples include:
- $ L_1 $ regularization (Lasso)
- $ L_2 $ regularization (Ridge)

Cost function with $ L_2 $ regularization:
$ J(w) = 1/(2m) sum_(i=1)^m ( f_(w)(x^(i)) - y^(i) )^2 + lambda sum_{j=1}^n w_j^2 $

---

=== Summary

To build effective models:
- Avoid underfitting by increasing model complexity or adding features.
- Prevent overfitting using regularization or reducing model complexity.
- Always validate on unseen data to ensure generalization.

= Cost Function with Regularization

In this video, we discuss the cost function in linear regression and how regularization can help prevent overfitting by reducing the magnitude of parameters.

== Regularization Intuition

When training a model, if we fit a high-degree polynomial, the model may overfit the data, as it may curve excessively to match every point. Regularization aims to penalize large weights, forcing them to shrink, which simplifies the model and helps prevent overfitting.

For instance, if we have a cost function for polynomial regression:

$ J(w) = sum_{i=1}^{m} lr( f(w) - y_i lr)^2 $

To incorporate regularization, we add a term that penalizes large parameters:

$ J(w) = sum_{i=1}^{m} lr( f(w) - y_i lr)^2 + lambda sum_{j=1}^{n} w_j^2 $

Here, $lambda$ is the regularization parameter, and the term $sum_{j=1}^{n} w_j^2$ encourages smaller weights. If $lambda$ is large, the model becomes simpler, as it minimizes the weights' magnitude.

#simpleTable(
  columns: (1fr, 1fr),
  [*Attribute*], [*Formula*],
  [*Regularized Cost Function*], [
    $ J(w) = sum_{i=1}^{m} (f(w) - y_i)^2 + lambda sum_{j=1}^{n} w_j^2 $
  ],
  [*Regularization Term*], [
    $ lambda sum_{j=1}^{n} w_j^2 $
  ],
  [*Objective*], [
    $ \min_w J(w) $
  ],
)

== Regularization Effect

The effect of regularization is to adjust the model complexity. Let's visualize the behavior of the cost function with regularization for a housing price prediction example.

For small $lambda$ (near 0), the model behaves as if there's no regularization, potentially overfitting. For a very large $lambda$, the model becomes overly simplistic, as the weights are heavily penalized.

#codeBlock(
```
python
  import numpy as np
  import matplotlib.pyplot as plt

  def plot_regularized_cost(lambda_value: float):
      # Plot cost function with regularization effect
      w_range = np.linspace(-5, 5, 100)
      cost_values = (w_range**2) + lambda_value * np.sum(w_range**2)  # Simplified cost function for demonstration
      plt.plot(w_range, cost_values, label=f"λ = {lambda_value}")
      plt.xlabel("w")
      plt.ylabel("Cost Function J(w)")
      plt.title(f"Effect of Regularization (λ = {lambda_value})")
      plt.legend()
      plt.show()

  plot_regularized_cost(0)  # No regularization
  plot_regularized_cost(10) # Strong regularization
```
)

=== Cost Function Behavior

When $lambda = 0$, the model has no regularization, and we may observe overfitting.

When $lambda$ is large (e.g., $10^{10}$), the model becomes overly simplistic, with small weights, and the curve becomes smoother.

== Choosing Lambda

Choosing $lambda$ is crucial. A small value of $lambda$ results in overfitting, while a large value leads to underfitting. Cross-validation is often used to find an optimal $lambda$.

For practical implementation, $lambda$ is typically tuned based on training data size and the complexity of the model.

=== Final Notes

- Regularization helps prevent overfitting by penalizing large weights.
- The regularized cost function combines both the original cost function and a penalty term.
- $lambda$ controls the balance between fitting the data and simplifying the model.
- In practice, choose $lambda$ using cross-validation to achieve the best generalization.

= Regularized Logistic Regression

== Key Concepts

Regularized logistic regression helps prevent overfitting by adding a regularization term to the cost function. This is particularly useful when working with polynomial features or models with many parameters.

=== Cost Function

The cost function for logistic regression is modified by adding a regularization term, which penalizes large values of the parameters to avoid overfitting. The cost function with regularization is given by:

$
J(w, b) = frac{1}{m} sum_{i=1}^{m} lr( -y^{(i)} log(f(w^T x^{(i)})) - (1 - y^{(i)}) log(1 - f(w^T x^{(i)})) lr) + frac{lambda}{2m} sum_{j=1}^{n} w_j^2
$

Where:
- $m$ is the number of training examples.
- $w_j$ are the parameters of the model.
- $lambda$ is the regularization parameter.

The term $frac{lambda}{2m} sum_{j=1}^{n} w_j^2$ penalizes large values of $w$, keeping the model more general and reducing the risk of overfitting.

=== Gradient Descent Update

To minimize the cost function, we use gradient descent. The update rules for the parameters are as follows:

1. For $w_j$:
   $
   w_j := w_j - alpha lr( frac{1}{m} sum_{i=1}^{m} (f(w^T x^{(i)}) - y^{(i)}) x_j^{(i)} + frac{lambda}{m} w_j lr)
   $

2. For $b$:
   $
   b := b - alpha lr( frac{1}{m} sum_{i=1}^{m} (f(w^T x^{(i)}) - y^{(i)}) lr)
   $

Here, $alpha$ is the learning rate, and $b$ is not regularized, so no regularization term is added to the update rule for $b$.

== Example

Consider a situation where we have a polynomial feature of a high order, leading to overfitting. The regularization term helps simplify the decision boundary, as shown in the following graph:

=== Implementing Regularized Logistic Regression

In practice, you would implement the gradient descent update for the parameters as follows:

#codeBlock(
  ```
  python
  def regularized_logistic_regression(X, y, lambda_val, alpha, num_iters):
      m, n = X.shape
      w = np.zeros(n)
      b = 0

      for _ in range(num_iters):
          z = np.dot(X, w) + b
          f = 1 / (1 + np.exp(-z))  # Sigmoid function
          dw = (1 / m) * np.dot(X.T, (f - y)) + (lambda_val / m) * w
          db = (1 / m) * np.sum(f - y)

          w -= alpha * dw
          b -= alpha * db

      return w, b
  ```
)

In this function:
- `X` is the matrix of input features.
- `y` is the vector of labels.
- `lambda_val` is the regularization parameter.
- `alpha` is the learning rate.
- `num_iters` is the number of iterations for gradient descent.

This code allows us to fit a regularized logistic regression model to a dataset, minimizing the cost function with respect to the parameters $w$ and $b$.

= Logistic Regression Notes

== Simplified Cost Function

In this class, we simplified the logistic regression cost function for easier implementation, especially when using gradient descent. Here's a breakdown:

=== Loss Function Simplification

The simplified loss function for binary classification (where $y \in {0, 1}$) is written as:

#simpleTable(
  columns: (1fr, 1fr),
  [*Case*], [*Simplified Loss*],
  [$y = 1$], [
    $ -log(f(x)) $
  ],
  [$y = 0$], [
    $ -log(1 - f(x)) $
  ],
)

However, we can combine these cases into a single expression:

$ L = - lr( y \cdot log(f(x)) + (1 - y) \cdot log(1 - f(x)) lr) $

This formulation works because:
- When $y = 1$, the second term vanishes (as $(1 - y) = 0$), leaving $-log(f(x))$.
- When $y = 0$, the first term vanishes (as $y = 0$), leaving $-log(1 - f(x))$.

=== Cost Function Definition

Using the simplified loss function, the cost function $J$ for the entire dataset is defined as:

$ J(w, b) = frac{1}{m} sum{i=1}^{m} - lr( y^{(i)} \cdot log(f(x^{(i)})) + (1 - y^{(i)}) \cdot log(1 - f(x^{(i)})) lr) $

Where:
- $m$ is the number of training examples.
- $f(x^{(i)})$ is the logistic function applied to the $i$-th training example.

We can rewrite it by factoring out the negative sign:

$ J(w, b) = - frac{1}{m} sum{i=1}^{m} lr( y^{(i)} \cdot log(f(x^{(i)})) + (1 - y^{(i)}) \cdot log(1 - f(x^{(i)})) lr) $

=== Properties of the Cost Function

- **Convexity**: This cost function is convex, meaning it has a single global minimum. This is crucial for gradient descent, ensuring it converges to the optimal solution.
- **Statistical Justification**: The function is derived using the principle of maximum likelihood estimation, which helps identify the most likely parameters for the logistic model.

== Example: Visualizing Cost for Two Parameter Choices

The lab exercise showed two different parameter choices ($w, b$), resulting in distinct costs. Below is a Python snippet to compute and visualize these differences:

#codeBlock(
```
python
  import numpy as np
  import matplotlib.pyplot as plt

  def logistic_function(x, w, b):
      return 1 / (1 + np.exp(-(w * x + b)))

  def cost_function(y, y_pred):
      return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

  x_train = np.linspace(-10, 10, 100)
  y_train = (x_train > 0).astype(int)  # Binary target: 1 for x > 0, else 0

  params = [
      {"w": 1, "b": 0},  # Choice 1
      {"w": 2, "b": -1}  # Choice 2
  ]

  costs = []
  for param in params:
      y_pred = logistic_function(x_train, param["w"], param["b"])
      costs.append(cost_function(y_train, y_pred))

  print(f"Costs for parameter choices: {costs}")
```
)

=== Notes

- The **blue decision boundary** with optimized parameters resulted in a lower cost compared to the **magenta decision boundary**, showing better model performance.
- Implementing this cost function in code will be essential for practical gradient descent applications.

In the next session, we'll explore how gradient descent can be applied to optimize this cost function for logistic regression.

= Demand Prediction with Neural Networks

== Introduction

Demand prediction is a key application of machine learning that helps businesses optimize inventory, marketing, and sales strategies. This handbook explores how neural networks model demand prediction using concepts from logistic regression and artificial neurons.

---

== Logistic Regression for Demand Prediction

Logistic regression models the probability of an event occurring, such as whether a product becomes a top seller. The model is represented by:

#simpleTable(
  columns: (1fr, 1fr),
  [*Concept*], [*Formula*],
  [*Model Output*], [
    $ a = frac{1}{1 + e^{-(w * x + b)}} $
  ],
  [*Parameters*], [
    $ w $: weight, $ b $: bias
  ],
  [*Activation*], [
    The predicted probability that the product is a top seller
  ],
)

Where:
- $x$ = product feature (e.g., price)
- $a$ = activation (output probability)

---

== Artificial Neurons as Building Blocks

An artificial neuron functions like a simple computer. It receives input features, processes them through an activation function, and outputs a prediction.

**Example:** Predicting whether a T-shirt becomes a top seller:
- Input: Price ($x$)
- Output: Probability of being a top seller ($a$)

---

== Neural Network Structure

Neural networks combine multiple artificial neurons to make complex predictions. Consider a T-shirt demand prediction model with the following features:

#simpleTable(
  columns: (1fr, 1fr),
  [*Feature*], [*Description*],
  [*Price*], [Cost of the T-shirt],
  [*Shipping Cost*], [Cost of delivery],
  [*Marketing Spend*], [Advertising expenses],
  [*Material Quality*], [Perceived fabric quality],
)

**Hidden Layer Neurons:**
1. **Affordability Neuron:** Inputs - Price, Shipping Cost
2. **Awareness Neuron:** Input - Marketing Spend
3. **Quality Perception Neuron:** Inputs - Price, Material Quality

The outputs of these neurons feed into a **Final Neuron**, predicting the overall likelihood of the T-shirt being a top seller.

---

=== Mathematical Representation

For each hidden neuron, the output $a_i$ is calculated as:

$ a_i = frac{1}{1 + e^{-(w_i x_i + b_i)}} $

The final prediction combines these activations:

$ \hat{y} = frac{1}{1 + e^{-(w_f \cdot a + b_f)}} $

Where:
- $a$ = vector of activations from hidden neurons
- $w_f$, $b_f$ = weights and bias of the final neuron


== Summary

- **Logistic Regression:** Models probability using a sigmoid function.
- **Artificial Neurons:** Simple units processing features to produce predictions.
- **Neural Networks:** Combine neurons in layers for complex tasks.

= Forward Propagation in Neural Networks

== Introduction

Forward propagation is the process of passing input data through a neural network to make predictions or inferences. It involves sequential calculations from the input layer through hidden layers to the output layer.

== Example: Handwritten Digit Recognition

Consider a binary classification problem distinguishing handwritten digits "0" and "1." We use a neural network with:

- **Input Layer:** An 8x8 grid (64 features).
- **Hidden Layers:** Two layers with 25 and 15 neurons, respectively.
- **Output Layer:** A single neuron indicating the probability of the digit being "1."

=== Step 1: From Input to First Hidden Layer

The first hidden layer calculates activations \(a^{(1)}\) using:

$ a^{(1)} = sigma(W^{(1)} x + b^{(1)}) $

Where:

- \(x\): Input features (64 pixels).
- \(W^{(1)}\): Weights connecting input to the first hidden layer.
- \(b^{(1)}\): Bias terms.
- \(sigma\): Activation function (e.g., ReLU or sigmoid).

#simpleTable(
  columns: (1fr, 1fr),
  [*Layer*], [*Calculation*],
  [Input], $x = a^{(0)}$,
  [Hidden Layer 1], $a^{(1)} = sigma(W^{(1)} x + b^{(1)})$,
)

=== Step 2: From First to Second Hidden Layer

Activations for the second hidden layer are calculated as:

$ a^{(2)} = sigma(W^{(2)} a^{(1)} + b^{(2)}) $

Where:

- \(W^{(2)}\): Weights connecting the first and second hidden layers.
- \(b^{(2)}\): Bias terms.

=== Step 3: Output Layer

The final prediction is computed using:

$ a^{(3)} = sigma(W^{(3)} a^{(2)} + b^{(3)}) $

Where \(a^{(3)}\) represents the predicted probability.

#simpleTable(
  columns: (1fr, 1fr),
  [*Layer*], [*Calculation*],
  [Hidden Layer 2], $a^{(2)} = sigma(W^{(2)} a^{(1)} + b^{(2)})$,
  [Output Layer], $a^{(3)} = sigma(W^{(3)} a^{(2)} + b^{(3)})$,
)

=== Summary

- **Forward Propagation:** Sequentially computes activations from input to output.
- **Use Case:** Inferencing using pre-trained models.

= Deep Learning Handbook: Image Recognition

== Neural Networks for Image Recognition

In computer vision, neural networks are used for tasks like facial recognition. A typical input is a grayscale image of size 1000x1000 pixels, represented as a 1000x1000 matrix of pixel intensity values ranging from 0 to 255. These values can be flattened into a vector of one million elements.

#simpleTable(
  columns: (1fr, 1fr),
  [*Concept*], [*Description*],
  [*Input Representation*], [
    X ={1000 x 1000 matrix of pixel intensities}
  ],
  [*Flattened Vector*], [
    X ={1,000,000-element vector}
  ],
)

=== Neural Network Architecture

A neural network processes the image through multiple layers:

1. **Input Layer:** The flattened image vector.
2. **Hidden Layers:** Extract features through learned filters.
3. **Output Layer:** Estimates probabilities of specific identities.


=== Feature Detection in Hidden Layers

Neurons in hidden layers learn different features:

1. **First Hidden Layer:** Detects edges and simple patterns.
2. **Second Hidden Layer:** Combines edges into facial parts.
3. **Third Hidden Layer:** Detects complete facial structures.

#simpleTable(
  columns: (1fr, 1fr),
  [*Layer*], [*Detected Features*],
  [*First Layer*], [Edges and lines],
  [*Second Layer*], [Facial parts like eyes and nose],
  [*Third Layer*], [Complete facial structures],
)


=== Transfer Learning Concept

The same network architecture can adapt to different tasks. If trained on car images, it will learn features like car edges, parts, and full car shapes.

= Deep Learning Handbook

== Neural Network Layers

Neural networks consist of interconnected layers of neurons. Each layer processes inputs and produces activations that are passed to the next layer.

=== Structure of a Neural Network Layer

A neural network layer takes inputs, processes them, and produces outputs through a set of neurons. Each neuron performs a weighted sum of the inputs, adds a bias, and applies an activation function.

#simpleTable(
  columns: (1fr, 1fr),
  [*Element*], [*Definition*],
  [*Input Vector*], [
    $ x = [x_1, x_2, ..., x_n]^T $
  ],
  [*Parameters*], [
    Weights $w_i$ and bias $b$
  ],
  [*Neuron Output*], [
    $ z = w^T x + b $
  ],
  [*Activation Function*], [
    $ a = g(z) = frac{1}{1 + e^{-z}} $
  ],
)

=== Example Calculation

Consider a hidden layer with three neurons and four input features:

1. First neuron:
   $ z_1 = w_1^T x + b_1 $, $ a_1 = g(z_1) $

2. Second neuron:
   $ z_2 = w_2^T x + b_2 $, $ a_2 = g(z_2) $

3. Third neuron:
   $ z_3 = w_3^T x + b_3 $, $ a_3 = g(z_3) $

The activations are passed as inputs to the next layer.

=== Notation and Layer Indexing

To distinguish layers, use superscripts:

- Layer 1 parameters: $w^{[1]}, b^{[1]}$
- Activations: $a^{[1]} = [a_1, a_2, a_3]^T$

The output of layer 1 becomes the input for layer 2.

=== Summary

- Each layer performs weighted sums and applies activation functions.
- Outputs of one layer become inputs to the next.
- Notation uses superscripts to indicate layer-specific parameters and activations.
