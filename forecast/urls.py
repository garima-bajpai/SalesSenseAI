from django.conf.urls import url
from . import views

app_name='forecast_app'

urlpatterns=[
    url(r'^forecast/fileupload/', views.fileUpload, name='file_upload'),
    url(r'^forecast/', views.forecast, name='forecast'),
    url(r'^tarselect/', views.targetSel, name='target_Sel'),
    url(r'^view/', views.selView, name='selections_view'),
    url(r'^selections/', views.selection, name='select_view'),
]