
```typst
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

# Implementación del Gradiente Descendente
Para implementar el gradiente descendente en regresión lineal, seguimos estos pasos:

- **Inicialización**: Asignamos valores iniciales a w y b (e.g., w = -0.1 y b = 900).
- **Actualización**: En cada paso, calculamos las derivadas y actualizamos w y b:

  ```
  w := w - α * ∂J/∂w
  b := b - α * ∂J/∂b
  ```

  donde α es la tasa de aprendizaje.

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

```

