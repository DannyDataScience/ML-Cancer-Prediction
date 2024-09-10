import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

@st.cache_data
def get_clean_data(file_path="data/data.csv"):
    """
    Load and clean the dataset:
    - Replace spaces with underscores in column names.
    - Map 'diagnosis' to binary values (M:1, B:0).
    """
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.replace(' ', '_')
    data = data.drop(columns=['Unnamed:_32', 'id'], errors='ignore')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def add_sidebar(data):
    """
    Add sliders to the sidebar for the input features, returning a dictionary of the selected values.
    """
    st.sidebar.header('Cell Nuclei Details')
    
    input_dict = {
        col: st.sidebar.slider(col.replace("_", " ").capitalize(), 
                               min_value=float(0), 
                               max_value=float(data[col].max()), 
                               value=float(data[col].mean())) 
        for col in data.columns if 'diagnosis' not in col
    }
    return input_dict

def scale_values(input_dict, data):
    """
    Scale the input values based on the dataset's min and max for each feature.
    """
    return {key: (value - data[key].min()) / (data[key].max() - data[key].min()) 
            for key, value in input_dict.items()}

def get_radar_chart(input_data, data):
    """
    Create and return a radar chart based on the scaled input data.
    Adjusts the chart's theme based on the app's current theme (light/dark).
    """
    theme_base = st.get_option("theme.base")
    pio.templates.default = "plotly_dark" if theme_base == 'dark' else "plotly_white"
    
    scaled_data = scale_values(input_data, data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 
                  'Compactness', 'Concavity', 'Concave points', 'Symmetry', 'Fractal dimension']
    
    def extract_values(suffix):
        return [scaled_data.get(f'{cat.lower().replace(" ", "_")}_{suffix}', 0) for cat in categories]

    traces = ['mean', 'se', 'worst']
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(go.Scatterpolar(
            r=extract_values(trace),
            theta=categories,
            fill='toself',
            name=trace.capitalize()
        ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    return fig

def add_predictions(input_data):
    """
    Load model and scaler, make predictions, and display the results.
    """
    model = pickle.load(open("model/model.pkl", 'rb'))
    scaler = pickle.load(open("model/scaler.pkl", 'rb'))

    input_array = scaler.transform(np.array(list(input_data.values())).reshape(1, -1))
    prediction = model.predict(input_array)[0]
    prob_benign, prob_malignant = model.predict_proba(input_array)[0] * 100

    st.subheader("Cell cluster prediction")
    color, text = ('#049470', 'Benign') if prediction == 0 else ('#a94030', 'Malignant')
    
    st.markdown(f'''
        <div style="background-color: {color}; border-radius: 10px; padding: 10px; text-align: center;">
            <p style="color:white;">{text}</p>
        </div>
    ''', unsafe_allow_html=True)
    
    st.markdown(f"**Probability of being benign**: {prob_benign:.2f}%")
    st.markdown(f"**Probability of being malignant**: {prob_malignant:.2f}%")
    st.write("This app assists in diagnosis, not a substitute for professional opinion.")

def main():
    st.set_page_config(page_title="Breast Cancer Predictor", page_icon="female-doctor", layout="wide", initial_sidebar_state='expanded')
    data = get_clean_data()
    input_data = add_sidebar(data)

    st.title("Breast Cancer Predictor")
    st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")

    col1, col2 = st.columns([4, 1])
    with col1:
        radar_chart = get_radar_chart(input_data, data)
        st.plotly_chart(radar_chart)
    
    with col2:
        add_predictions(input_data)

if __name__ == '__main__':
    main()


