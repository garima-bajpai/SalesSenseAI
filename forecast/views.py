from django.shortcuts import render, redirect
from django.http import HttpRequest
from django.template import RequestContext
import cPickle
import numpy as np
import pandas as pd
from .trainData import trainingData
from .net import network
from .plot import linePlot
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime


from .forms import uploadFileForm

def selection(request):
    assert isinstance(request, HttpRequest)
    return render(request,
                  'forecast/selections.html',
                  context_instance=RequestContext(request,
                                                  {
                                                      'fields': request.session['column_names'],
                                                      'flag': True
                                                  }
                                                  )
                  )



def targetSel(request):
    assert isinstance(request,HttpRequest)
    if request.method == 'POST':
        request.session['predictors'] = request.POST.getlist('choices')
        not_predlist = [item for item in request.session['column_names'] if item not in request.session['predictors']]
        return render(request,
                      'forecast/selections.html',
                      context_instance=RequestContext(request,
                                                      {
                                                          'fields':not_predlist,
                                                          'flag':False
                                                      }
                                                      )
                      )


def selView(request):
    assert isinstance(request,HttpRequest)
    if request.method == 'POST':
        request.session['targets']=request.POST.get('choices')
        request.session['network'] = None
        return render(request,
                      'forecast/view.html',
                      context_instance=RequestContext(request,
                                                      {
                                                          'predictors':request.session['predictors'],
                                                          'targets':request.session['targets']
                                                      })
                      )




def forecast(request):
    assert isinstance(request, HttpRequest)
    if request.session['network'] == None:
        try:
            df = cPickle.loads(str(request.session['dataframe']))
            td = trainingData(df)
            td.preprocess(request.session['predictors'], request.session['targets'])
            nn = network(td.trainX,td.trainY)
            p = nn.predict(td.testX)
            errormabs = (1-mean_absolute_error(td.testY,p))*100
            errorm = (1-mean_squared_error(td.testY,p))*100
            td.out_mapper.inverse_transform(p)
            request.session['network'] =  cPickle.dumps(nn)
            request.session['transformers']= cPickle.dumps(td)
        except RuntimeError:
            print "Something went wrong with the network"
        return redirect('forecast_app:file_upload')
    else:
        return redirect('forecast_app:file_upload')


# Create your views here.
def fileUpload(request):
    assert isinstance(request, HttpRequest)
    if request.method == 'POST':
        form = uploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['docFile']
            tar_df = pd.read_csv(file,parse_dates=[x for x in range(0,4)], infer_datetime_format=True)
            tar_cols = tar_df.columns.values.tolist()

            if set(request.session['predictors']).issubset(set(tar_cols)):
                drop_list=[]
                for name in tar_cols:
                    if name not in request.session['predictors']:
                        drop_list.append(tar_cols.index(name))
                #can fail if list is empty!!!
                tar_df.drop(tar_df.columns[drop_list], axis=1, inplace=True)
                tar_df.dropna(axis=1,how='any',inplace = True)
                td = cPickle.loads(str(request.session['transformers']))
                nn = cPickle.loads(str(request.session['network']))
                p = nn.predict(np.array(td.in_mapper.transform(tar_df),np.float32))
                p1 = td.out_mapper.inverse_transform(p)
                name = "Predicted"+ " "+ str(request.session['targets'])
                tar_df[name] = p1
                date_df = tar_df.select_dtypes(include=['datetime64[ns]']).astype(datetime.date)
                date_df = date_df.astype(str)
                date_df = date_df.ix[:,0].tolist()
                url = linePlot(date_df,p)
                url = "https://dashboards.ly" + url
                tblhead = "<th>"+name+"</th>"
                return render(request,
                              'temp/list2.html',

                              context_instance=RequestContext(request,
                                                              {
                                                                  'url': url,
                                                                  'data': tar_df.to_html(#float_format=lambda x: '%10.2f' % x,\
                                                                                         formatters={name:lambda x: '%.3f' % x},
                                                                                         max_rows=20)\
                                                                                        .replace('<table border="1" class="dataframe">','<table id = "datatable-responsive" \
                                                                                        class="table table-striped table-bordered dt-responsive responsive-utilities \
                                                                                        jambo_table nowrap dataTable no-footer dtr-inline collapsed">')\
                                                                                        .replace(tblhead,"<th><u>"+name+"</u></th>")

                                                              })
                              )
            else:
                #pop message file does not contain same fields
                return render(request,
                              'temp/list.html',

                              context_instance=RequestContext(request,
                                                              {
                                                                  'data':"invalid fields in uploaded files!!!"

                                                              })
                              )
        else:
            type = request.FILES['docFile'].content_type
            size = request.FILES['docFile'].size
            return render(request,
                          # 'analyze/selections.html',
                          'temp/list.html',
                          context_instance=RequestContext(request,
                                                          {
                                                              # 'fields':request.session['column_names'],
                                                              # 'flag':True
                                                              'data': "invalid type {} with size {}".format(type, size)
                                                          })
                          )
    else:
        form = uploadFileForm()
        return render(request,
                      'forecast/upload.html',
                      context_instance=RequestContext(request,
                                                      {
                                                          'form': form
                                                      }
                                                      )
                      )


    #return redirect('forecast_app:file_upload')
