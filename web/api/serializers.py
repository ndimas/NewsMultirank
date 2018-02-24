from rest_framework.serializers import ModelSerializer

from web.api.models import Prediction


class PredictionSerializer(ModelSerializer):
    class Meta:
        model = Prediction
        fields = ('id', 'result')
        extra_kwargs = {
            'id': {'read_only': True}
        }
