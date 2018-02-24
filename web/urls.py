from django.conf.urls import url
from web.api.views import PredictionReadView

urlpatterns = [
    url(r'^prediction/$', PredictionReadView.as_view(), name='prediction'),
]
