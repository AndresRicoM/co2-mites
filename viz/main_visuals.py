import dash
import dash_daq as daq
import dash_html_components as html

external_stylesheets = ['https://gitlab.cba.mit.edu/classes/863.20/CBA/cbasite/-/blob/master/people/andresrico/stylesheet.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    daq.NumericInput(
        id='my-numeric-input',
        value=0
    ),
    html.Div(id='numeric-input-output')
])


@app.callback(
    dash.dependencies.Output('numeric-input-output', 'children'),
    [dash.dependencies.Input('my-numeric-input', 'value')])

def update_output(value):
    return 'The value is {}.'.format(value)


if __name__ == '__main__':
    app.run_server(debug=True)
