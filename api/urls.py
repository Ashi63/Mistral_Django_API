from django.urls import path
from . import views  # Import views from the current directory

urlpatterns = [
    path('extract-information/', views.extract_information, name='extract_information'),
]
