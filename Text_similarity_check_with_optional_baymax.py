# Import required libraries
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go
from transformers import pipeline
import spacy  # For text similarity calculation

# Load spaCy language model
nlp = spacy.load("en_core_web_md")

# Define COLORS and STYLES 
COLORS = {
    'primary': '#355c7d',  
    'accent': '#c5e1e5',   
    'text': '#333'        
}

STYLES = {
    'container': {
        'padding': '40px',
        'background-color': '#e0f7f9'  
    },
    'card': {
        'background-color': 'white',
        'padding': '30px',
        'border-radius': '15px',
        'box-shadow': '0 4px 8px rgba(0,0,0,0.2)',
        'width': '70%',
        'margin': 'auto'
    },
    'header': {
        'background-color': COLORS['primary'],
        'padding': '10px',  
        'border-radius': '10px',
        'color': 'white',
        'text-align': 'center',
        'font-size': '1.6em',  
        'font-weight': 'bold'
    },
    'subheader': {
        'text-align': 'center',
        'color': 'white',
        'margin-top': '-5px', 
        'font-size': '0.85em',
        'opacity': '0.9'
    },
    'input': {
        'width': '100%',
        'padding': '10px',
        'margin': '10px 0',
        'border-radius': '8px',
        'border': '1px solid #355c7d',
        'font-size': '0.85em' 
    },
    'input_group': {
        'background-color': '#aad8e6',
        'padding': '10px',
        'border-radius': '10px',
        'margin-bottom': '0px'  
    },
    'input_container': {
        'display': 'flex',
        'justify-content': 'space-between',
        'gap': '5px'  # Reduced gap between text input boxes
    },
    'button': {
        'background-color': COLORS['primary'],
        'color': 'white',
        'padding': '6px 18px',
        'border': 'none',
        'border-radius': '10px',
        'cursor': 'pointer',
        'font-size': '0.9em',  # Adjusted font size for button
        'font-weight': 'bold',
        'transition': 'all 0.3s ease',
        'margin-top': '20px',
        'text-align': 'center'
    },
    'output': {
        'background-color': COLORS['accent'],
        'padding': '20px',
        'border-radius': '10px',
        'margin-top': '20px',
        'text-align': 'center',
        'font-size': '0.85em',
        'font-weight': 'bold'
    },
    'chat_button': {
        'position': 'fixed',
        'bottom': '20px',
        'left': '20px',
        'background-color': COLORS['primary'],
        'color': 'white',
        'padding': '10px 20px',
        'border': 'none',
        'border-radius': '50%',
        'cursor': 'pointer',
        'font-size': '20px',
        'box-shadow': '0 4px 8px rgba(0,0,0,0.2)'
    },
    'chat_window': {
        'position': 'fixed',
        'bottom': '80px',
        'left': '20px',
        'background-color': 'white',
        'border-radius': '8px',
        'box-shadow': '0 4px 8px rgba(0,0,0,0.2)',
        'padding': '15px',
        'width': '300px',
        'display': 'none'
    },
    'chat_output': {
        'max-height': '200px',
        'overflow-y': 'auto',
        'background-color': COLORS['accent'],
        'padding': '10px',
        'border-radius': '5px',
        'margin-bottom': '10px'
    },
}

app = Dash(__name__, suppress_callback_exceptions=True)

# Load the language model for chat
chat_model = pipeline("text-generation", model="gpt2")

app.layout = html.Div(
    style=STYLES['container'],
    children=[
        html.Div(
            style=STYLES['card'],
            children=[
                # Header
                html.Div(
                    children=[
                        html.H1("Text Similarity Calculator"),
                        html.P("Compare the similarity between two texts using advanced NLP techniques Developed by Team Baymax", style=STYLES['subheader'])
                    ],
                    style=STYLES['header']
                ),
                
                # Input Section
                html.Div([
                    html.Div([
                        html.Label("First Text", style={'font-weight': 'bold', 'color': COLORS['primary'], 'font-size': '0.85em'}),
                        dcc.Textarea(
                            id='input-text1',
                            placeholder='Enter your first text here...',
                            style={'height': '100px', **STYLES['input']}
                        ),
                    ], style=STYLES['input_group']),
                    
                    html.Div([
                        html.Label("Second Text", style={'font-weight': 'bold', 'color': COLORS['primary'], 'font-size': '0.85em'}),
                        dcc.Textarea(
                            id='input-text2',
                            placeholder='Enter your second text here...',
                            style={'height': '100px', **STYLES['input']}
                        ),
                    ], style=STYLES['input_group']),
                ], style=STYLES['input_container']),
                
                # Similarity Method Selection
                html.Div([
                    html.Label("Select Similarity Method", 
                               style={'font-weight': 'bold', 'color': COLORS['primary'], 'margin-right': '15px', 'font-size': '0.85em'}),
                    dcc.Dropdown(
                        id='similarity-method',
                        options=[
                            {'label': 'Default spaCy Similarity', 'value': 'default'},
                            {'label': 'Word Vector Similarity', 'value': 'word_vector'},
                            {'label': 'Token Overlap Similarity', 'value': 'token_overlap'}
                        ],
                        value='default',
                        style={'width': '250px', 'font-size': '0.85em'}
                    )
                ], style={'margin': '20px 0', 'text-align': 'center'}),
                
                # Calculate Button
                html.Div(
                    html.Button(
                        'Calculate Similarity',
                        id='calculate-button',
                        style=STYLES['button']
                    ),
                    style={'text-align': 'center'}
                ),
                
                # Output Section
                html.Div(
                    id='output-container',
                    style=STYLES['output'],
                    children=[
                        dcc.Loading(
                            id="loading",
                            type="circle",
                            children=[
                                html.Div(id='output-similarity'),
                                dcc.Graph(id='similarity-graph')
                            ]
                        )
                    ]
                )
            ]
        ),
        
        # Floating Chat Button
        html.Button("💬", id="chat-button", style=STYLES['chat_button']),
        
        # Chat Window (Initially Hidden)
        html.Div(
            id="chat-window",
            style=STYLES['chat_window'],
            children=[
                html.Div(id='chat-output', style=STYLES['chat_output']),
                dcc.Textarea(
                    id='chat-input',
                    placeholder='Type your message here...',
                    style={'width': '100%', 'height': '50px', 'margin-top': '10px', 'font-size': '0.85em'}
                ),
                html.Button(
                    'Send', id='send-chat-button', style=STYLES['button']
                ),
            ]
        )
    ]
)

# Callback for Similarity Calculation
@app.callback(
    [Output('output-similarity', 'children'),
     Output('similarity-graph', 'figure')],
    [Input('calculate-button', 'n_clicks')],
    [State('input-text1', 'value'),
     State('input-text2', 'value'),
     State('similarity-method', 'value')]
)
def update_output(n_clicks, text1, text2, method):
    if not n_clicks or not text1 or not text2:
        return "Please enter text in both fields and click Calculate.", {}

    # Calculate similarity using spaCy
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    similarity = doc1.similarity(doc2)

    # Create visualization
    figure = {
        'data': [
            go.Bar(
                x=['Similarity Score'],
                y=[similarity],
                marker_color=COLORS['primary'],
                width=[0.4]
            )
        ],
        'layout': go.Layout(
            title={'text': f'Similarity Score: {similarity:.4f}', 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            yaxis={'title': 'Score', 'range': [0, 1], 'gridcolor': COLORS['accent']},
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
    }
    
    return html.Div([
        html.H3(f"Similarity Score: {similarity:.4f}", style={'color': COLORS['primary'], 'margin-bottom': '10px'}),
        html.P(f"Method used: spaCy Model Similarity", style={'color': COLORS['text']})
    ], style={'text-align': 'center'}), figure

# Callback to Toggle Chat Window
@app.callback(
    Output('chat-window', 'style'),
    [Input('chat-button', 'n_clicks')],
    [State('chat-window', 'style')]
)
def toggle_chat_window(n_clicks, style):
    if n_clicks:
        style['display'] = 'block' if style['display'] == 'none' else 'none'
    return style

# Callback for Chat Interactions
@app.callback(
    Output('chat-output', 'children'),
    [Input('send-chat-button', 'n_clicks')],
    [State('chat-input', 'value'), State('chat-output', 'children')]
)
def chat_with_llm(n_clicks, user_input, chat_history):
    if not n_clicks or not user_input:
        return chat_history or []

    response = chat_model(user_input, max_length=50, num_return_sequences=1)
    response_text = response[0]['generated_text']
    chat_history = chat_history or []
    chat_history.extend([
        html.P(f"User: {user_input}", style={'font-weight': 'bold'}),
        html.P(f"LLM: {response_text}", style={'color': COLORS['text']})
    ])
    return chat_history

if __name__ == '__main__':
    app.run_server(debug=True)
