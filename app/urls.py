from django.conf.urls import url
from . import views
app_name = 'app'

urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^fileupload/', views.fileUpload, name='app_file_upload'),
]