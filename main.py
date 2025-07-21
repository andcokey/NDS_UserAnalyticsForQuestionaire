import os
import glob
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# === 設定セクション ===
CSV_PATH = "data/【Navi内のみ】ネットde診断アンケート_v5_20250721.csv"
ENCODING = 'utf-8'
TEXT_COLUMNS = ['回答1', '回答2', '回答3', '回答4', '回答8', '回答9', '回答10']
N_REPRESENTATIVES = 5
N_KEYWORDS = 10
OUTPUT_SUMMARY_PATH = "outputs/cluster_summary.csv"

# 1. データ読み込み
try:
    df = pd.read_csv(CSV_PATH, encoding=ENCODING)
except FileNotFoundError:
    dirpath = os.path.dirname(CSV_PATH) or '.'
    candidates = glob.glob(os.path.join(dirpath, '*.csv'))
    suggestion = f"候補一覧: {candidates}" if candidates else f"{dirpath} に CSV が見つかりません。"
    raise FileNotFoundError(f"CSVファイルが見つかりません: {CSV_PATH}\n{suggestion}")
missing = [col for col in TEXT_COLUMNS if col not in df.columns]
if missing:
    raise KeyError(f"以下の列が見つかりません: {missing}。利用可能: {df.columns.tolist()}")

# 2. テキスト結合
df['text'] = df[TEXT_COLUMNS].fillna('').astype(str).agg(' '.join, axis=1)
texts = df['text'].tolist()

# 3. 文ベクトル化 + 正規化
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

# 4. 標準化 & PCA
scaler = StandardScaler()
scaled_emb = scaler.fit_transform(embeddings)
pca = PCA(n_components=min(50, scaled_emb.shape[1]), random_state=42)
reduced_pca = pca.fit_transform(scaled_emb)
print(f"PCA後次元: {reduced_pca.shape[1]}")

# 5. UMAP 可視化
umap_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_umap = umap_reducer.fit_transform(reduced_pca)

# 6. 最適クラスタ数探索 (シルエットスコア)
scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = km.fit_predict(reduced_pca)
    scores.append(silhouette_score(reduced_pca, labels_k))
best_k = range(2, 11)[int(np.argmax(scores))]
print(f"最適クラスタ数: {best_k}")

# 7. KMeans 実行 & UMAP プロット
kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)
labels = kmeans.fit_predict(reduced_pca)
df['cluster'] = labels
plt.figure(figsize=(10, 6))
sc = plt.scatter(embedding_umap[:,0], embedding_umap[:,1], c=labels, cmap='tab10', alpha=0.7)
plt.legend(*sc.legend_elements(), title='Cluster')
centroids_umap = umap_reducer.transform(kmeans.cluster_centers_)
plt.scatter(centroids_umap[:,0], centroids_umap[:,1], c='k', s=100, marker='X', label='Centroids')
for i, (x, y) in enumerate(centroids_umap):
    plt.text(x, y, f'C{i}', ha='center', va='center', color='white', fontweight='bold')
plt.title('UMAP + KMeans Clustering')
plt.tight_layout()
plt.savefig('outputs/vector_clusters.png')
plt.show()

# 8. クラスタ解釈サマリー
# 距離に基づく代表文抽出 & TF-IDF キーワード抽出
dist_matrix = euclidean_distances(reduced_pca, kmeans.cluster_centers_)
df['dist_to_centroid'] = [dist_matrix[i, lbl] for i, lbl in enumerate(labels)]
vectorizer = TfidfVectorizer(max_features=2000, token_pattern=r'(?u)\b\w+\b')
tfidf = vectorizer.fit_transform(texts)
terms = vectorizer.get_feature_names_out()
summary = []
for c in range(best_k):
    idxs = np.where(labels == c)[0]
    reps = df.loc[idxs].nsmallest(N_REPRESENTATIVES, 'dist_to_centroid')['text'].tolist()
    mean_tfidf = np.asarray(tfidf[idxs].mean(axis=0)).ravel()
    keywords = [terms[i] for i in mean_tfidf.argsort()[::-1][:N_KEYWORDS]]
    summary.append({'cluster': c, 'size': len(idxs), 'representatives': reps, 'keywords': keywords})
sum_df = pd.DataFrame([
    {
        'cluster': s['cluster'],
        'size': s['size'],
        'representatives': ' | '.join(s['representatives']),
        'keywords': ', '.join(s['keywords'])
    } for s in summary
])
sum_df.to_csv(OUTPUT_SUMMARY_PATH, index=False, encoding='utf-8-sig')
print(f"Cluster summary saved to {OUTPUT_SUMMARY_PATH}")
print(sum_df)
