from django.urls import path
from . import views
from django.urls import include, path
urlpatterns = [
    path('', views.index,name = 'list'),
    path('learning', views.learning,name = 'learning'),
    path('model', views.model,name = 'model'),
    path('about',views.about,name = 'about'),
    path(r'^admin_tools/', include('admin_tools.urls')),
]
