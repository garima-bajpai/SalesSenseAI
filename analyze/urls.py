from django.conf.urls import url
from . import views

app_name = 'analyze_app'

urlpatterns = [
    url(r"^type/",views.selChart, name='sel_type'),
    url(r"^plot/(?P<type>\w+)/", views.plot, name='plot_graph')
]