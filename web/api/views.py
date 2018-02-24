from api.models import Prediction
from api.serializers import PredictionSerializer
from rest_framework.generics import ListCreateAPIView


class PredictionReadView(ListCreateAPIView):
    queryset = Prediction("OK")
    serializer_class = PredictionSerializer
