import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt

# 1. データ読み込み
df = pd.read_csv("【Navi内のみ】ネットde診断アンケート_v5_20250721.csv")

# 2. モデル読み込み（日本語対応）
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 3. テキストをベクトル化
embeddings = model.encode(df['text'].tolist())

# 4. 次元削減（UMAP）
reducer = umap.UMAP(n_components=2)
reduced = reducer.fit_transform(embeddings)

# 5. クラスタリング（例：5クラスタ）
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(embeddings)

# 6. 可視化
plt.figure(figsize=(10, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10')
for i, txt in enumerate(df['text']):
    plt.annotate(txt[:10], (reduced[i, 0], reduced[i, 1]), fontsize=8)
plt.title("文の意味ベクトル分析（クラスタリング＋UMAP）")
plt.savefig("outputs/vector_plot.png")
plt.show()
