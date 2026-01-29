# This is a project to analyse and figure out how to get 80% of the results with 20% of the effort for exams using math and machine learning
# importing all the necessary modules


import fitz
import re
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np
import hdbscan
from sklearn.preprocessing import MinMaxScaler

#We upload a pdf to the script, extract the necessary data, figure out the most important keywords and rank them by hyper-parameter metrics of frequency and importance

# 1. take text from pdf
def extract_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text() + "\n"
    return text

# 2. topic extraction
def extract_topics(text):
    # to split lines and punctuation
    raw = re.split(r"[.\n;:]", text)
    topics = [t.strip() for t in raw if 3 < len(t.split()) < 12]
    return list(set(topics))

# 3. bidirectional encoder representations from transformers
def keyword_filter(topics):
    kw_model = KeyBERT()
    final_topics = []
    for t in topics:
        kws = kw_model.extract_keywords(t, keyphrase_ngram_range=(1,2), stop_words='english')
        if kws:
            final_topics.append(kws[0][0])   # keep strongest keyword
    return list(set(final_topics))

# 4. making the words into vectors and doing HDBSCAN
def cluster_topics(topics):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(topics)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
    labels = clusterer.fit_predict(embeddings)

    return embeddings, labels

# 5. make a dataframe of topic,cluster
def pareto_score(topics, embeddings, labels):
    df = pd.DataFrame({
        "topic": topics,
        "cluster": labels
    })

    # distance analysis logic to classify subject relevance
    centroids = {}
    for c in set(labels):
        if c == -1: 
            continue
        idxs = np.where(labels == c)[0]
        centroids[c] = embeddings[idxs].mean(axis=0)

    centrality = []
    for i, t in enumerate(topics):
        c = labels[i]
        if c == -1:
            centrality.append(0.1)
            continue
        dist = np.linalg.norm(embeddings[i] - centroids[c])
        centrality.append(1 / (1 + dist))

    # calculating frequency 
    freq = []
    raw_counts = {t.lower(): 0 for t in topics}
    for t in topics:
        raw_counts[t.lower()] += 1
    for t in topics:
        freq.append(raw_counts[t.lower()])

    # Normalize everything
    scaler = MinMaxScaler()
    df["centrality"] = scaler.fit_transform(np.array(centrality).reshape(-1,1))
    df["frequency"] = scaler.fit_transform(np.array(freq).reshape(-1,1))

    # making paretto score on my normalised data
    df["pareto_score"] = 0.7 * df["centrality"] + 0.3 * df["frequency"]

    # ranking each topic
    df = df.sort_values("pareto_score", ascending=False).reset_index(drop=True)
    top20 = int(len(df) * 0.2)
    df["Top20Percent"] = df.index < top20

    return df


# This is where u link the pdf
pdf_path = "source.pdf"


text = extract_pdf(pdf_path)
topics = extract_topics(text)
keywords = keyword_filter(topics)
embeddings, labels = cluster_topics(keywords)
df = pareto_score(keywords, embeddings, labels)


#this is where we name and set the outputted dataframe into a csv file which we use for analysis
output_path = "output.csv"
df.to_csv(output_path, index=False)

df.head(50), output_path
