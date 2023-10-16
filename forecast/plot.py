from django.http import HttpRequest
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import cPickle
import plotly.tools as tls
import requests, json

def linePlot(x_dataf, y_data):

    trace1 = go.Bar(
        x=x_dataf,
        y=y_data
    )

    plotdata = [trace1]
    url1 = py.plot({'data':plotdata,
                    'layout': {
        'title': 'growth',
        'margin': {'l': 30, 'r': 30, 'b': 30, 't': 60}
        }
        },filename="temp/line-plot", auto_open=False)

    trace2 = go.Scatter(
        x=x_dataf,
        y=y_data,
        mode='markers'
    )
    plotdata2 = [trace2]
    url2 = py.plot({'data': plotdata2,
                    'layout': {
                        'title': 'growth2',

                        'margin': {'l': 30, 'r': 30, 'b': 30, 't': 60}
                    }
                    }
                   , "temp/line2-plot", auto_open=False)

    trace3 = go.Histogram(
        x=x_dataf,
        opacity=0.75
    )

    plotdata3 = [trace3]
    url3 = py.plot({'data': plotdata3,
                    'layout': {
                        'title': 'growth3',
                        'margin': {'l': 30, 'r': 30, 'b': 30, 't': 60}
                    }
                    }
                   , "temp/line3-plot", auto_open=False)

    dashboard_json = {
        "rows": [
            [{"plot_url": url1}],
             [{"plot_url": url2},{"plot_url": url3}]
        ],
        "banner": {
            "visible": True,
            "backgroundcolor": "#2A3F54",
            "textcolor": "white",
            "title": "Forecasted Results",
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