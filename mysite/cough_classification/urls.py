from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('cough', views.cough, name='cough'),
    path('audio', views.cough, name='audio'),

]