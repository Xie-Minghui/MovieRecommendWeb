from django.urls import path
from Users import views

urlpatterns = [
    path('<int:userID>/', views.getRecommendMovies),
]
