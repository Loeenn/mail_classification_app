from django.contrib import admin
from django.urls import path,include
from django.conf import settings
from django.conf.urls.static import static
#from . import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',include('main.urls')),
    #path("ajax_file_upload",views.ajax_file_upload)
    #path("ajax_file_upload_save",views.ajax_file_upload_save)
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)