from django.shortcuts import render
from django.http import HttpRequest
from django.template import RequestContext
from .plot import *

def selChart(request):
    assert isinstance(request, HttpRequest)
    return render(request,
                  'analyze/selType.html',
                  context_instance=RequestContext(request,
                  									{
                  									"cat_names":request.session['cat_list'],
                                                    "num_names": request.session['num_list'],
                  									}
                                                  ))

def plot(request, type):
    assert isinstance(request, HttpRequest)
    typelist = ['line', 'pie','bar','scatter', 'histogram','distplot']
    if request.method == "GET":
        if type in typelist:
            if type == 'line':
                #data = request.GET.copy()
                url = linePlot(request)
                url = "https://dashboards.ly"+url
            elif type == 'pie':
                url = piePlot(request)
                url = "https://dashboards.ly" + url
            elif type=='bar':
                url = barPlot(request)
                url = "https://dashboards.ly" + url
            elif type == 'histogram':
                url = histPlot(request)
                url = "https://dashboards.ly" + url
            elif type == 'distplot':
                url = distPlot(request)
                url = "https://dashboards.ly" + url
            else:
                url = scatterPlot(request)
                url = "https://dashboards.ly" + url
            return render(request,
                          'analyze/plot.html',
                          context_instance=RequestContext(request,
                                                          {
                                                              "url": url
                                                          }
                                                          ))
        else:
            return render(request,
                          'temp/list.html',
                          context_instance=RequestContext(request,
                                                          {
                                                              "data": "Im out"
                                                          }
                                                          ))
    else:
        return render(request,
                      'temp/list.html',
                      context_instance=RequestContext(request,
                                                      {
                                                          "data": "hiii"
                                                      }
                                                      ))

