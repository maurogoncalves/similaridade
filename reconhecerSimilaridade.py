import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Função para carregar e preprocessar uma imagem
def process_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)

# Caminho das imagens
image_dir = "caminho/para/suas/imagens"
image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]

# Modelo pré-treinado para extração de características
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Extraindo embeddings
embeddings = []
for image_path in image_paths:
    image_array = process_image(image_path)
    feature_vector = model.predict(image_array)
    embeddings.append(feature_vector.flatten())

embeddings = np.array(embeddings)

# Similaridade entre imagens
similarity_matrix = cosine_similarity(embeddings)

# Agrupamento usando KMeans (opcional)
n_clusters = 3  # Defina o número de clusters desejados
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Exibir resultados
for cluster_id in range(n_clusters):
    cluster_images = [image_paths[i] for i in range(len(image_paths)) if clusters[i] == cluster_id]
    print(f"\nCluster {cluster_id + 1}:")
    for img in cluster_images:
        print(img)

# Exemplo de visualização de similaridade
plt.imshow(similarity_matrix, cmap='viridis')
plt.colorbar()
plt.title("Matriz de Similaridade")
plt.show()