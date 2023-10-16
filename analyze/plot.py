import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import cPickle
import requests, json
from plotly.tools import FigureFactory as FF

def linePlot(request):
    df = cPickle.loads(str(request.session['dataframe']))
    trace = go.Scatter(
        x=df[request.GET['x_axis']],
        y=df[request.GET['y_axis']]
    )
    plotdata = [trace]
    url = py.plot(plotdata, "temp/line-plot", auto_open=False)
    dashboard_json = {
        "rows": [
            [{"plot_url": url}]
        ],
        "banner": {
            "visible": True,
            "backgroundcolor": "#2A3F54",
            "textcolor": "white",
            "title": "{} vs {}".format(request.GET['x_axis'],request.GET['y_axis'] ),
            "links": []
        },
        "requireauth": False,
        "auth": {
            "username": "Acme Corp",
            "passphrase": ""
        }
    }
    response = requests.post('https://dashboards.ly/publish',
                             data={'dashboard': json.dumps(dashboard_json)},
                             headers={'content-type': 'application/x-www-form-urlencoded'})
    response.raise_for_status()
    dashboard_url = response.json()['url']
    return dashboard_url



def piePlot(request):
    df = cPickle.loads(str(request.session['dataframe']))
    df1 = df.groupby(request.GET['group_by']).agg({request.GET['x_axis']: 'sum'})
    fig = {
        'data': [{'labels': df[request.GET['group_by']].unique().tolist(),
                  'values': df1[request.GET['x_axis']].tolist(),
                  'type': 'pie'}],
    }
    url = py.plot(fig, filename='Pie Chart Example',auto_open=False)

    dashboard_json = {
        "rows": [
            [{"plot_url": url}]
        ],
        "banner": {
            "visible": True,
            "backgroundcolor": "#2A3F54",
            "textcolor": "white",
            "title": "{} group_by {}".format(request.GET['x_axis'], request.GET['group_by']),
            "links": []
        },
        "requireauth": False,
        "auth": {
            "username": "Acme Corp",
            "passphrase": ""
        }
    }
    response = requests.post('https://dashboards.ly/publish',
                             data={'dashboard': json.dumps(dashboard_json)},
                             headers={'content-type': 'application/x-www-form-urlencoded'})
    response.raise_for_status()
    dashboard_url = response.json()['url']
    return dashboard_url


def barPlot(request):
    df = cPickle.loads(str(request.session['dataframe']))
    df1 = df.groupby(request.GET['group_by']).sum()
    trace1 = go.Bar(
        x=df[request.GET['group_by']].unique().tolist(),
        y=df1[request.GET['x_axis']].tolist(),
        name=request.GET['x_axis'],
        marker=dict(
            color='rgb(55, 83, 109)'
        )
    )

    trace2 = go.Bar(
        x=df[request.GET['group_by']].unique().tolist(),
        y=df1[request.GET['y_axis']].tolist(),
        name=request.GET['y_axis'],
        marker=dict(
            color='rgb(26, 118, 255)'
        )
    )

    trace3 = go.Bar(
        x=df[request.GET['group_by']].unique().tolist(),
        y=df1[request.GET['z_axis']].tolist(),
        name=request.GET['z_axis'],
        marker=dict(
            color='rgb(142, 124, 195)'
        )
    )

    layout = go.Layout(
        title='Comparison of {}, {} and {} vs {}'.format(request.GET['x_axis'],request.GET['y_axis'],request.GET['z_axis'],request.GET['group_by']),
        xaxis=dict(
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title='USD (millions)',
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )
    data = [trace1, trace2, trace3]
    fig = go.Figure(data=data, layout=layout)

    url = py.plot(fig, filename='style-bar',auto_open=False)

    dashboard_json = {
        "rows": [
            [{"plot_url": url}]
        ],
        "banner": {
            "visible": False,
            "backgroundcolor": "#2A3F54",
            "textcolor": "white",
            #"title": "{} group_by {}".format(request.GET['x_axis'], request.GET['group_by']),
            "links": []
        },
        "requireauth": False,
        "auth": {
            "username": "Acme Corp",
            "passphrase": ""
        }
    }
    response = requests.post('https://dashboards.ly/publish',
                             data={'dashboard': json.dumps(dashboard_json)},
                             headers={'content-type': 'application/x-www-form-urlencoded'})
    response.raise_for_status()
    dashboard_url = response.json()['url']
    return dashboard_url


def scatterPlot(request):
    df = cPickle.loads(str(request.session['dataframe']))
    trace0 = go.Scattergl(
        y=df[request.GET['x_axis']],
        mode='markers',
        name=request.GET['x_axis'],
        marker=dict(
            size='13',
            #color=np.random.randn(500),  # set color equal to a variable

        ))
    trace1 = go.Scattergl(
        y=df[request.GET['y_axis']],
        mode='markers',
        name=request.GET['y_axis'],
        marker=dict(
            size='13',
            # color=np.random.randn(500),  # set color equal to a variable

        ))
    data=[trace0,trace1]
    layout = dict(title = '{} vs {}'.format(request.GET['x_axis'],request.GET['y_axis']),

                  )
    fig = go.Figure(data=data, layout=layout)
    url = py.plot(fig, filename='Scatte example',auto_open=False)

    dashboard_json = {
        "rows": [
            [{"plot_url": url}]
        ],
        "banner": {
            "visible": False,
            "backgroundcolor": "#2A3F54",
            "textcolor": "white",
            #"title": "{} group_by {}".format(request.GET['x_axis'], request.GET['y_axis']),
            "links": []
        },
        "requireauth": False,
        "auth": {
            "username": "Acme Corp",
            "passphrase": ""
        }
    }
    response = requests.post('https://dashboards.ly/publish',
                             data={'dashboard': json.dumps(dashboard_json)},
                             headers={'content-type': 'application/x-www-form-urlencoded'})
    response.raise_for_status()
    dashboard_url = response.json()['url']
    return dashboard_url


def histPlot(request):
    df = cPickle.loads(str(request.session['dataframe']))

    x0=df[request.GET['x_axis']]
    x1=df[request.GET['y_axis']]


    trace1 = go.Histogram(
        x=x0,
        histnorm='count',
        name=request.GET['x_axis'],
        autobinx=False,
        xbins=dict(
            start=-3.2,
            end=2.8,
            size=0.2
        ),
        marker=dict(
            color='rgb(55, 83, 109)',
            line=dict(
                color='grey',
                width=0
            )
        ),
        opacity=0.75
    )
    trace2 = go.Histogram(
        x=x1,
        name=request.GET['y_axis'],
        autobinx=False,
        xbins=dict(
            start=-1.8,
            end=4.2,
            size=0.2
        ),
        marker=dict(
            color='rgb(26, 118, 255)'
        ),
        opacity=0.75
    )
    data = [trace1, trace2]
    layout = go.Layout(
        title='{} vs {}'.format(request.GET['x_axis'],request.GET['y_axis']),
        xaxis=dict(
            title=request.GET['x_axis']
        ),
        yaxis=dict(
            title=request.GET['x_axis']
        ),
        barmode='overlay',
        bargap=0.25,
        bargroupgap=0.3
    )
    fig = go.Figure(data=data, layout=layout)
    url = py.plot(fig, filename='style-histogram', auto_open=False)

    dashboard_json = {
        "rows": [
            [{"plot_url": url}]
        ],
        "banner": {
            "visible": False,
            "backgroundcolor": "#2A3F54",
            "textcolor": "white",
            # "title": "{} group_by {}".format(request.GET['x_axis'], request.GET['y_axis']),
            "links": []
        },
        "requireauth": False,
        "auth": {
            "username": "Acme Corp",
            "passphrase": ""
        }
    }
    response = requests.post('https://dashboards.ly/publish',
                             data={'dashboard': json.dumps(dashboard_json)},
                             headers={'content-type': 'application/x-www-form-urlencoded'})
    response.raise_for_status()
    dashboard_url = response.json()['url']
    return dashboard_url

def distPlot(request):
    df = cPickle.loads(str(request.session['dataframe']))
    x0 = df[request.GET['x_axis']]
    x1 = df[request.GET['y_axis']]
    x2 = df[request.GET['z_axis']]

    data=[x0,x1,x2]
    labels = [request.GET['x_axis'],request.GET['y_axis'],request.GET['z_axis']]
    colors = ['rgb(0, 0, 100)', 'rgb(0, 200, 200)','rgb(0, 200, 100)']
    fig = FF.create_distplot(
        data, labels, bin_size=.2, colors=colors)
    fig['layout'].update(title='Distplot of {} vs {} vs {}'.format(request.GET['x_axis'], \
                                                                   request.GET['y_axis'], \
                                                                   request.GET['z_axis']))
    url = py.plot(fig, filename='Distplot with Normal Curve', auto_open=False)
    dashboard_json = {
        "rows": [
            [{"plot_url": url}]
        ],
        "banner": {
            "visible": False,
            "backgroundcolor": "#2A3F54",
            "textcolor": "white",
            # "title": "{} group_by {}".format(request.GET['x_axis'], request.GET['y_axis']),
            "links": []
        },
        "requireauth": False,
        "auth": {
            "username": "Acme Corp",
            "passphrase": ""
        }
    }
    response = requests.post('https://dashboards.ly/publish',
                             data={'dashboard': json.dumps(dashboard_json)},
                             headers={'content-type': 'application/x-www-form-urlencoded'})
    response.raise_for_status()
    dashboard_url = response.json()['url']
    return dashboard_url
