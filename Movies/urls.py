from django.urls import path
from Movies import views

urlpatterns = [
    path('<int:itemID>/', views.RecommendSimilaryItem),
    path('other/<int:itemID>/', views.RecommendOtherFavoriteMovies),
]
