# Retrieval-Augmented ML Systems

Exposición del curso Aprendizaje Automático, Maestría en Ciencia de Datos y Analítica, Universidad EAFIT.

Autores: Santiago Betancur y Santiago Vélez.

## De qué trata

La exposición presenta los sistemas de machine learning aumentados con recuperación: modelos que, antes de responder, consultan una memoria externa en lugar de depender solo de lo que aprendieron en sus pesos.

Para mostrarlo construimos un ejemplo con una marca de motos inventada, Velora. Tiene un catálogo de 120 códigos de error repartidos en cinco categorías, y cada código tiene su falla, su síntoma y su solución. Como esa información no existe en internet, ningún modelo la conoce de antemano, y eso la hace un buen caso para recuperar antes de responder.

El sistema funciona así:

1. Un modelo de embeddings de Hugging Face (intfloat/multilingual-e5-small) convierte cada código en un vector de 384 dimensiones. Ese modelo no se entrena ni se ajusta.
2. Los vectores se guardan en una base de datos vectorial, en la tabla codigos_velora, con un índice HNSW.
3. Cuando llega una pregunta, el sistema recupera los códigos más parecidos. El primero es la respuesta, y los cinco primeros votan por la categoría (k-NN).

En la demo se comparan tres formas de buscar: sparse (BM25, por palabras), dense (embeddings, por significado) e hybrid (las dos combinadas con Reciprocal Rank Fusion). Al final se agregan los códigos de una moto nueva, la E-Volt, con un INSERT en la base de datos, y se muestra que el sistema los aprende sin reentrenar ningún modelo.

## Archivos

```
.
├── README.md
├── requirements.txt
├── Presentacion_RAG.pdf
├── CARGUE_RAG.ipynb
├── DEMO_RAG.ipynb
└── datos/
    ├── codigos_velora.csv
```

Presentacion_RAG.pdf tiene las diapositivas de la exposición.

CARGUE_RAG.ipynb prepara la base de datos. Crea la tabla codigos_velora, calcula los embeddings de los 120 códigos, los guarda y valida los tres métodos de búsqueda con preguntas de prueba. Si se vuelve a ejecutar, borra la tabla y la crea de nuevo.

DEMO_RAG.ipynb es la demo que se mostró en clase. Muestra qué hay en la base, cómo una pregunta se convierte en vector, un mapa del catálogo con PCA y UMAP, cómo se responde una pregunta, la comparación entre sparse, dense e hybrid, y el caso de la E-Volt: se pregunta por un código que no está, se suben los códigos nuevos y se vuelve a preguntar.

La carpeta datos tiene una copia en tablas de la información que usan los notebooks: los 132 códigos (120 del catálogo base y 12 de la E-Volt) y las preguntas de prueba con su código correcto. Los notebooks no leen estos archivos, porque los datos ya vienen dentro de ellos; están ahí para consultarlos fácilmente.

## Requisitos

Se necesita una base de datos vectorial: un Postgres con la extensión pgvector. Nosotros usamos Supabase, que tiene un plan gratuito y trae pgvector disponible.

La conexión a la base se pasa con el nombre SUPABASE_DB_URL. En Colab se guarda como un secreto; fuera de Colab, como variable de entorno. La cadena de conexión nunca se escribe en el código.

Las librerías de Python están en requirements.txt:

- sentence-transformers, para el modelo de embeddings
- psycopg y pgvector, para conectarse a la base y manejar vectores
- numpy, pandas y scikit-learn
- umap-learn, para el mapa en dos dimensiones
- matplotlib, para las gráficas

Los notebooks se hicieron en Google Colab con Python 3.11 o superior. En Colab la primera celda de cada notebook instala lo que falta.

## Cómo ejecutarlo

1. Crear un proyecto en Supabase y activar la extensión vector en Database, Extensions.
2. En el botón Connect del proyecto, copiar la cadena de conexión del Session pooler (puerto 5432) y reemplazar [YOUR-PASSWORD] por la contraseña de la base de datos.
3. En Colab, abrir el panel Secrets (el ícono de la llave), crear un secreto llamado SUPABASE_DB_URL con esa cadena y darle acceso al notebook.
4. Ejecutar primero CARGUE_RAG.ipynb completo. Este paso carga la base de datos.
5. Después ejecutar DEMO_RAG.ipynb. Sin el paso anterior la demo no funciona, porque la tabla no existe o está vacía.

DEMO_RAG.ipynb borra los códigos de la E-Volt al iniciar, así que se puede ejecutar varias veces y siempre empieza sin ellos.

Para correrlo fuera de Colab: instalar las dependencias con pip install -r requirements.txt y definir la variable de entorno SUPABASE_DB_URL antes de abrir los notebooks.

## Sobre los datos y los resultados

La marca Velora, sus modelos de moto y todos los códigos de error fueron inventados por nosotros para la exposición. Cualquier parecido con productos reales es coincidencia.

Como el catálogo es pequeño y las preguntas de prueba son pocas, los resultados sirven para ilustrar cómo funciona cada método, no como una medición general. Algunas cosas pueden cambiar un poco entre ejecuciones, por ejemplo la posición de los grupos en el mapa de UMAP.

## Referencias

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Khandelwal et al. (2019). Generalization through Memorization: Nearest Neighbor Language Models.
- Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering.
- Robertson y Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Cormack, Clarke y Büttcher (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.
- Malkov y Yashunin (2016). Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs.
- Wang et al. (2022). Text Embeddings by Weakly-Supervised Contrastive Pre-training.
- McInnes, Healy y Melville (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
- pgvector: https://github.com/pgvector/pgvector
