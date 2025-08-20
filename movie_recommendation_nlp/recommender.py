from sentence_transformers import SentenceTransformer
import tensorflow as tf

# Load once globally
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def recommend_movies(user_query, movies_norm, top_n=3):
    # Encode query
    query_emb = sentence_model.encode([user_query], convert_to_numpy=True)
    query_emb_tf = tf.convert_to_tensor(query_emb, dtype=tf.float32)
    query_norm = tf.nn.l2_normalize(query_emb_tf, axis=1)

    # Cosine similarity
    similarity = tf.matmul(movies_norm, query_norm, transpose_b=True)
    similarity = tf.squeeze(similarity, axis=1)

    # Top N indices
    top_indices = tf.argsort(similarity, direction='DESCENDING')[:top_n]

    return top_indices.numpy().tolist(), similarity.numpy().tolist()
