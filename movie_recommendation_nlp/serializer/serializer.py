from rest_framework import serializers
from movie_recommendation_nlp.model.model import Prediction


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'
