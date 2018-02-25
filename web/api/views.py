from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from rest_framework.response import Response

from prediction.predict import doPredict

parser_classes = (JSONParser,)


@api_view(['GET', 'POST'])
def predict(request):

    title = request.data['title']

    body = request.data['body']

    print("head:" + title + " , body:" + body)

    result = doPredict(title, body)

    print("Rresult is" + result)

    # result_json = json.dumps({'result', result})

    return Response({"status": 200, "result": result})
