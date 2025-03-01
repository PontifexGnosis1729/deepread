import json
import logging
import numpy as np
import umap.umap_ as umap

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.documents import Document
from typing import Sequence


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def perform_topic_modeling(docs, embeddings, visualize=False):
    """
    Performs topic modeling using BERTopic on precomputed vector embeddings.

    :param embeddings: NumPy array or list of shape (n_samples, embedding_dim)
    :param visualize: Whether to generate topic visualizations
    :return: BERTopic model and topic assignments
    """

    # Train a BERTopic model
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=1729)
    topic_model = BERTopic(umap_model=umap_model, calculate_probabilities=True)

    # Fit BERTopic on precomputed embeddings
    print("Performing topic modeling on embeddings...")
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Fine-tune topic representations after training BERTopic
    vectorizer_model = CountVectorizer(ngram_range=(1, 5), stop_words="english")
    topic_model.update_topics(docs, vectorizer_model=vectorizer_model)

    # Show the top topics
    print("\n===== Top Topics Identified =====")
    print(topic_model.get_topic_info())

    # Visualizations (if enabled)
    if visualize:
        print("\nGenerating visualizations...")

        # Visualize the topic clusters in 2D
        topic_model.visualize_topics().show()

        # Visualize the topic heat matrix
        topic_model.visualize_heatmap().show()

        # Visualize the topic hierarchy
        topic_model.visualize_hierarchy().show()


        # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
        reduced_embeddings = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=1729).fit_transform(embeddings)

        # Visualize the documents
        topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings).show()

        # Visualize the document datamap
        #topic_model.visualize_document_datamap(docs, reduced_embeddings=reduced_embeddings)

    return topic_model, topics


def get_topic_modeling_info(docs: Sequence[Document], array_vectors: np.ndarray) -> list[Document]:
    # Perform topic modeling
    texts = [doc.page_content for doc in docs]
    topic_model, topics = perform_topic_modeling(texts, array_vectors)

    labeled_chunks_df = topic_model.get_document_info(texts)
    labeled_chunks_df["Representation"] = labeled_chunks_df["Representation"].apply(json.dumps)

    labeled_docs = [
        Document(
            page_content=doc.page_content, metadata={**doc.metadata, "topic_name": name, "topic_representation": representation}
        )
        for doc, name, representation in zip(docs, labeled_chunks_df["Name"], labeled_chunks_df["Representation"])
    ]

    return labeled_docs

