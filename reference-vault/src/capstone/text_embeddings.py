import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df["text"].tolist()


def get_embeddings(texts, model="text-embedding-ada-002"):
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, model=model)
        embeddings.append(response["data"][0]["embedding"])
    return np.array(embeddings)


def classify_texts(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans


def visualize_clusters(embeddings, labels):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    plt.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis"
    )
    plt.title("Text Embedding Clusters")
    plt.show()


def summarize_clusters(texts, labels, n_clusters, model="gpt-4o-mini"):
    summaries = {}
    for i in range(n_clusters):
        cluster_texts = [texts[j] for j in range(len(texts)) if labels[j] == i]
        summary_prompt = f"Provide a summary of the following texts:\n{chr(10).join(cluster_texts[:10])}"
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize the following texts."},
                {"role": "user", "content": summary_prompt},
            ],
        )
        summaries[i] = response["choices"][0]["message"]["content"]
    return summaries


if __name__ == "__main__":
    api_key = "API_KEY"
    openai.api_key = api_key
    file_path = "./text_dataset.csv"
    n_clusters = 3

    texts = load_data(file_path)
    embeddings = get_embeddings(texts)
    labels, _ = classify_texts(embeddings, n_clusters)
    visualize_clusters(embeddings, labels)
    summaries = summarize_clusters(texts, labels, n_clusters)

    for cluster_id, summary in summaries.items():
        print(f"\nCluster {cluster_id} Summary:\n{summary}\n")
