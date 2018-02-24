from api.models import Prediction
from rest_framework.serializers import ModelSerializer


class PredictionSerializer(ModelSerializer):
    class Meta:
        model = Prediction
        fields = ('id', 'result')
        extra_kwargs = {
            'id': {'read_only': True}
        }
