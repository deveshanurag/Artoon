from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="ShopHome"),
    path("result/", views.result, name="Result"),
    path("test/", views.test, name="Test"),
]
