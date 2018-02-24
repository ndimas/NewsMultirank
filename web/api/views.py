from rest_framework.generics import ListCreateAPIView

from web.api.models import Prediction
from web.api.serializers import PredictionSerializer


class PredictionReadView(ListCreateAPIView):
    queryset = (Prediction("OK"))
    serializer_class = PredictionSerializer
