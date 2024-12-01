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
