from django.conf.urls import url
from web.api import views

urlpatterns = [
    url(r'^prediction/$', views.predict),
]
