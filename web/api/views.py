from rest_framework.decorators import api_view
from rest_framework.response import Response

from prediction.predict import doPredict

@api_view(['GET', 'POST'])
def predict(request):
    print("inside predictr")
    result = doPredict("body", "title")
    return Response(request.data)
