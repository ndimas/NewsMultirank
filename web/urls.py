from django.conf.urls import url

urlpatterns = [
    url(r'^prediction/$', PredictionReadView.as_view(), name='prediction'),
]
