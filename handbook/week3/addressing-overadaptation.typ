#import "../lib/template.typ": main
#import "../lib/simpleTable.typ": simpleTable
#import "../lib/codeBlock.typ": codeBlock

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
