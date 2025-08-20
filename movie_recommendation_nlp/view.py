from rest_framework import viewsets, status
from rest_framework.response import Response
from movie_recommendation_nlp.model.model import Prediction
from movie_recommendation_nlp.serializer.serializer import PredictionSerializer
from movie_recommendation_nlp.recommender import recommend_movies
import os
# Example: load precomputed embeddings (movies_norm) at startup
import tensorflow as tf
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load movies dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'dataset', 'movies.csv')
df_movies = pd.read_csv(csv_path)  # must contain 'overview' column

# Precompute embeddings
model_path = os.path.join(BASE_DIR, 'dataset', 'model')
loaded_module = tf.saved_model.load(model_path)
movies_norm_loaded = loaded_module.embeddings # Access as a NumPy array

class PredictionViewSet(viewsets.ModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer

    def create(self, request, *args, **kwargs):
        query_text = request.data.get("query")
        top_n = request.data.get("top_n", 18)
        if not query_text:
            predictions = []
            for idx, record in enumerate(df_movies.to_dict(orient="records")):
                predictions.append({
                    "movie": record["original_title"],
                    "url": record["poster_path"],
                    "id": idx
                })
            return Response(predictions, status=status.HTTP_200_OK)

        # Run recommender
        top_indices, scores = recommend_movies(query_text, movies_norm_loaded, top_n=top_n)

        # Collect predictions
        predictions = []
        for idx in top_indices:
            predictions.append({
                "movie": df_movies.iloc[idx]["original_title"],
                "url": df_movies.iloc[idx]["poster_path"],
                "id": idx,
                "score": scores[idx]
            })

        # Save to DB
        prediction_obj = Prediction.objects.create(
            query=query_text,
            prediction=predictions
        )

        serializer = self.get_serializer(prediction_obj)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
