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

---

Visualizaciones Adicionales en el Laboratorio Opcional

En el laboratorio interactivo, puedes visualizar cómo diferentes configuraciones de \( w \) y \( b \) afectan el costo:

1. *Gráfico de Contorno Interactivo:* Puedes seleccionar puntos en el gráfico de contorno para ver cómo se visualiza la línea de \( f(x) \) correspondiente en el conjunto de datos.
2. *Superficie 3D Giratoria:* Explora la superficie 3D de \( J(w, b) \) y observa cómo se mueve el punto de menor costo con diferentes valores de los parámetros.

Estas herramientas permiten una comprensión visual y práctica de la función de costo y su minimización.

