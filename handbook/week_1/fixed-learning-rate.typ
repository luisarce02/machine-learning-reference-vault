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

