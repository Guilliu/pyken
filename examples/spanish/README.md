## <b>Ejemplos de uso</b>

Los siguientes notebooks de ejemplo pretenden mostrar distintos casos de uso:
-	**01 clase autoscorecard.ipynb**
-	**02 predicciones y overfitting en titanic.ipynb**
-	**03 agrupaciones interactivas y missings.ipynb**
-	**04 descifrando scorecard.ipynb**
-	**05 inferencia de denegados.ipynb**

`En el 01` se introduce la clase **autoscorecard**, la forma más automática de usar Pyken. Se genera un modelo automático con la configuración de hiperparámetros por defecto utilizando un dataset de juguete de scikit-learn sobre el cáncer de mama.

`En el 02` se muestra cómo utilizar el output de la clase autoscorecard para hacer **predicciones**. En concreto se utilizan dos datasets de Titanic de Kaggle: el original súper famoso y otro menos conocido ambientado en una nave espacial donde sus tripulantes pueden cambiar o no de dimensión. Se sacan modelos automáticos con la clase autoscorecard que se modifican ligeramente para reducir posibles efectos de **overfitting**. Generamos unas **submissions** que pueden subirse a Kaggle y ver en qué parte de la leaderboard nos situamos 😋.

`En el 03` cambiamos a otro dataset de infartos (sí, van todos de desgracias 😅) de Kaggle. Probamos a **cambiar las agrupaciones automáticas de manera interactiva** en variables de distinto tipo de datos y vemos cómo generar nuevas scorecards aplicando estas reagrupaciones manuales. Aprovechamos también para explicar cuál es el tratamiento que hace Pyken de los valores **missings**.

`En el 04` se da un enfoque algo más teórico. Desengranamos, paso a paso, el **funcionamiento interno de la clase autoscorecard** utilizando para ello el dataset de juguete del ejemplo 01. Se obtienen exactamente los mismos resultados.

`En el 05` usamos un nuevo dataset donde asignamos de manera aleatoria a una submuestra del 25% la etiqueta de denegados. Mostramos como desarrollar una scorecard utilizando el **parceling** como técnica de **inferencia de denegados**.

Se recomienda descargar los notebooks para visualizarlos mejor en JupyterLab o VSC (y así probar a ejecutarlos) ya que en GitHub su visualización es peor, sobre todo la parte de pintado de las scorecards en donde no se distingue tan bien las distintas variables involucradas.

