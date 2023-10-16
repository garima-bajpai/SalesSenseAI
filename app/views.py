from django.shortcuts import render
from django.http import HttpRequest
from django.template import RequestContext
import cPickle
import pandas as pd
from .forms import uploadFileForm
# Create your views here.

def home(request):
    assert isinstance(request, HttpRequest)
    return render(request,
                  'app/index.html',
                  context_instance=RequestContext(request,
                                                  {
                                                      'title':'Home Page',
                                                  }
                                                  )
                  )


def fileUpload(request):
    assert isinstance(request, HttpRequest)
    if request.method == 'POST':
        form = uploadFileForm(request.POST,request.FILES)
        if form.is_valid():
            file = request.FILES['docFile']
            df = pd.read_csv(file)
            cat_list=[]
            num_list=[]
            for name in df.columns.values.tolist():
                if df[name].dtype == 'O':
                    cat_list.append(name)
                else:
                    num_list.append(name)
            request.session['cat_list']=cat_list
            request.session['num_list']=num_list
            request.session['dataframe'] = cPickle.dumps(df)
            request.session['column_names'] = df.columns.values.tolist()
            return render(request,
                          'app/twoway.html',
                          context_instance=RequestContext(request,
                                                          {
                                                              #'fields':request.session['column_names'],
                                                              #'flag':True
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
                      'app/upload.html',
                      context_instance=RequestContext(request,
                                                      {
                                                       'form':form
                                                      }
                                                      )
                      )

