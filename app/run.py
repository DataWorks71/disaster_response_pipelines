import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
my_table_name = 'ModelTrainData'
my_database_filepath = 'data/DisasterResponse.db'

engine = create_engine('sqlite:///../' + my_database_filepath)

df = pd.read_sql_table(my_table_name, engine, index_col='index')


# load model
my_model_filepath = 'models/classifier.pkl'
model = joblib.load('../' + my_model_filepath)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # TODO: Below is an other example - modify to extract data for your own visuals
    cat_freq = df[df.columns[4:]].sum() / len(df) 
    cat_names = list(df.columns[4:])

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # Below is a horizontal frequency barplot of selected variabels
        {
            'data': [
                Bar(
                    y=['related', 'aid_related', 'infrastructure_related', 'weather_related', 'request', 'offer', 'direct_report'],
                    x=cat_freq[['related', 'aid_related', 'infrastructure_related', 'weather_related', 'request', 'offer', 'direct_report']],
                    orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Frequency of Selected Categories<br>(label skewness)',
                'subtitle': 'Frequency of Categories (positive label skewness)',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Percentage(positive labels)"
                }
            }
        },
        
        # Below is a horizontal frequency barplot of weather_related variabels
        {
            'data': [
                Bar(
                    y=['weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather'],
                    x=cat_freq[['weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather']],
                    orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Frequency of Weather Related Categories<br>(label skewness)',
                'subtitle': 'Frequency of Categories (positive label skewness)',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Percentage(positive labels)"
                }
            }
        },
        
        # Below is a horizontal frequency barplot of infrastructure_related variabels
        {
            'data': [
                Bar(
                    y=['infrastructure_related', 'transport', 'buildings', 'electricity',
                       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure'],
                    x=cat_freq[['infrastructure_related', 'transport', 'buildings', 'electricity',
                                'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure']],
                    orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Frequency of Infrastructure Related Categories<br>(label skewness)',
                'subtitle': 'Frequency of Categories (positive label skewness)',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Percentage(positive labels)"
                }
            }
        },
        
        # Below is a horizontal frequency barplot of aid_related variabels
        {
            'data': [
                Bar(
                    y=['aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
                       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid'],
                    x=cat_freq[['aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                                'security', 'military', 'child_alone', 'water', 'food', 'shelter',
                                'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid']],
                    orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Frequency of Aid Related Categories<br>(label skewness)',
                'subtitle': 'Frequency of Categories (positive label skewness)',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Percentage(positive labels)"
                }
            }
        }
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()