 # **Resumen del proyecto.**

Con los recientes avances en la capacidad de las GPU y las redes neuronales convolucionales, la visión por medio de un computador ha ganado un gran reconocimiento. Los computadores nos dan ahora una mayor precisión que los humanos para imágenes que son bastante complejas y tienen características que no son fácilmente diferenciables a simple vista, pero de nuevo los computadores tienen la destreza de averiguar los detalles de forma rigurosa.
Por esta razón se buscó una forma de poder clasificar unas fotos de cultivos con diferentes características que pueden pronosticar la salud de las siembras por medio la arquitectura de transfer learning _NASNetLarge_ con diferentes tecnicas de manejo de grandes volumenes de datos.

# **Sobre el Dataset.**

El conjunto de datos seleccionado es “Agriculture-Vision", tomado del portal de Amazon, el cual proporciona imágenes de 3.432 campos de cultivos de maíz y soya con nueve (9) características etiquetadas por agrónomos.
Este conjunto de datos está compuesto por:

-   94.986 imágenes 
-   70.500 de Entrenamiento 
-   24,486 de Prueba
-   512x512 pixeles
-   Cada imagen de campo contiene tres canales de color.

Dataset: https://registry.opendata.aws/intelinair_agriculture_vision/ 


# **Insights:** 

Gracias al procesamiento que fue realizado con TensorFlow y la lectura de las imágenes por medio de TFRecords se pudo encontrar que las etiquetas con las cuales se deben trabajar las clasificaciones de imágenes son 9 storm damage, drydown, endrow, wáter, nutrient deficient, double plant, waterway, weed cluster, planter skip. 
Para cada sección del campo agrícola fotografiado se presentan cinco variantes de la misma imagen distribuidas en los siguientes grupos: 
1.  RGB: Clasificada en el grupo basado en la adicción de colores lumínicos primarios (rojo, verde y azul). 
2.  NIR: Near infrared, correspondiente a la imagen del espectro infrarrojo. 
3.  Field labels: Contiene cuadriláteros demarcando el área de la fotografía que presenta una afectación característica en particular. 
4.  Field bounds: Contiene la imagen bicromática que discrimina que porción del área de la fotografía corresponde a cultivo y cual a algo diferente. (i.e. Una Carretera).

# **Construcción del modelo y su desempeño:**

Implementamos un clasificador de imágenes apoyados de la red neuronal de transfer learning NASNetLarge por la particularidad únicas de las imágenes del dataset de Agriculture-Vision, también se implemento la arquitectura resnet50 y mobilenet donde estas no dieron buenos resultados.
La estrategia que implementamos para el entrenamiento de los modelos fue que utilizamos un dataset en formato de directorio y un formato tipo tfrecord para ver las ventajas de uno con respecto al otro bajo el entrenamiento de diferentes mecanismos físicos (CPU – GPU).

En la primera iteración comparamos el tiempo de entrenamiento del modelo bajo el dataset en formato de directorio contra el formato tfrecord en una CPU. 
En la segunda iteración también comparamos el tiempo de entrenamiento del modelo bajo el dataset en formato de directorio contra el formato tfrecord en una GPU.
En la tercera iteración implementamos una estrategia de entrenamiento distribuido por medio de la API de tensorflow “tf.distribute.Strategy”. Esta nos proporciona una abstracción para distribuir su entrenamiento a través de múltiples unidades de procesamiento. El objetivo es permitir habilitar el entrenamiento distribuido utilizando las GPU disponibles en los servidores donde se entrenan los modelos, básicamente, copia todas las variables del modelo a cada procesador.
En nuestro caso utilizamos una estrategia llamada “MirroredStrategy” que es una de muchas estrategias de distribución disponible de la api de tensorflow. Cuando se entrena un modelo con múltiples GPUs, se puede utilizar la potencia de cálculo adicional de forma eficaz aumentando el tamaño del lote. En general, utilizando el mayor tamaño de lote que se ajuste a la memoria de la GPU y ajuste la tasa de aprendizaje en consecuencia del modelo.


