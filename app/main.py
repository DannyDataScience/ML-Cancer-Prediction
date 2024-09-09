import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    
    data.columns = data.columns.str.replace(' ', '_')
    data = data.drop(['Unnamed:_32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    
    return data

def add_sidebar():
    st.sidebar.header('Cell Nuclei Details')

    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave_points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave_points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave_points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):
    # Detectar el tema actual
    theme_base = st.get_option("theme.base")
    
    # Configurar el template de Plotly según el tema
    if theme_base == 'dark':
        pio.templates.default = "plotly_dark"
    else:
        pio.templates.default = "plotly_white"

    input_data = get_scaled_values(input_data)

    fig = go.Figure()

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave points', 'Symmetry', 'Fractal Dimension']

    def get_values(suffix):
        values = []
        for category in categories:
            key = f'{category.lower().replace(" ", "_")}_{suffix}'
            values.append(input_data.get(key, 0))
        return values

    mean_values = get_values('mean')
    se_values = get_values('se')
    worst_values = get_values('worst')

    fig.add_trace(go.Scatterpolar(
        r=mean_values,
        theta=categories,
        fill='toself',
        name='Mean'
    ))
    fig.add_trace(go.Scatterpolar(
        r=se_values,
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=worst_values,
        theta=categories,
        fill='toself',
        name='Worst Values'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", 'rb'))
    scaler = pickle.load(open("model/scaler.pkl", 'rb'))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")

    if prediction[0] == 0:
        st.markdown(
            f'''
            <div style="background-color: #049470; border-radius: 10px; padding: 10px; text-align: center; display: inline-block; width: auto; max-width: 100%; margin: 0px 20px 20px 0px;">
                <p style="color:white; margin: 0; display: inline;">Benign</p>
            </div>
            ''',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'''
            <div style="background-color: #a94030; border-radius: 10px; padding: 10px; text-align: center; display: inline-block; width: auto; max-width: 100%; margin: 0px 20px 20px 0px;">
                <p style="color:white; margin: 0; display: inline;">Malignant</p>
            </div>
            ''',
            unsafe_allow_html=True
        ) 

    prob_benign = (model.predict_proba(input_array_scaled)[0][0]) * 100

    prob_malignant = (model.predict_proba(input_array_scaled)[0][1]) * 100

    st.markdown(f'<p style="display:inline;">Probability of being benign: </p>'
            f'<p style="color:#AA41FB; display:inline;">{round(prob_benign, 2)}%</p>', unsafe_allow_html=True)

    st.markdown(f'<p style="display:inline;">Probability of being malignant: </p>'
            f'<p style="color:#AA41FB; display:inline;">{round(prob_malignant, 2)}%</p>', unsafe_allow_html=True)
    
    st.write(" ")
    
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="female-doctor",
        layout="wide",
        initial_sidebar_state='expanded'
    )

    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    
    with col2:
        with st.container(border=True):
            add_predictions(input_data)
 

if __name__ == '__main__':
    main()


