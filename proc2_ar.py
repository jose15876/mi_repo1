import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             recall_score, roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configuración de la app
st.set_page_config(page_title="Clasificación de Atletas", page_icon=':runner:', layout='centered')

# Cargar datos
@st.cache_data
def load_data():
    file_path = "D:/Usuarios/jcirujanom01/Desktop/2ºEval/atletas_250_redondeado.csv"
    data = pd.read_csv(file_path)
    return data

data = load_data()

# Página principal
st.title("Clasificación de Atletas: Fondistas vs Velocistas")

# Sidebar para navegación
page = st.sidebar.selectbox("Seleccione una página", ["Preprocesamiento", "Modelado", "Predicción"])

# Pestaña de Preprocesamiento
if page == "Preprocesamiento":
    st.header("Preprocesamiento de Datos")

    st.subheader("Datos Crudos")
    st.write(data.head())

    st.subheader("Valores Faltantes")
    st.write(data.isna().sum())

    data_clean = data.dropna()
    st.write("Datos después de eliminar NaN:", data_clean.shape)

    with st.expander("📦 Detección de Outliers"):
        fig, ax = plt.subplots()
        sns.boxplot(data=data_clean, x='Tipo de Atleta', y='Peso', ax=ax)
        st.pyplot(fig)

    with st.expander("📊 Distribuciones de Variables"):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        sns.histplot(data_clean, x='Edad', hue='Tipo de Atleta', kde=True, ax=axes[0, 0])
        sns.histplot(data_clean, x='Peso', hue='Tipo de Atleta', kde=True, ax=axes[0, 1])
        sns.histplot(data_clean, x='Frecuencia Cardiaca Basal', hue='Tipo de Atleta', kde=True, ax=axes[1, 0])
        sns.histplot(data_clean, x='IMC', hue='Tipo de Atleta', kde=True, ax=axes[1, 1])
        st.pyplot(fig)

    with st.expander("📈 Correlación entre Variables"):
        numeric_data = data_clean.select_dtypes(include=[np.number])
        corr = numeric_data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)

# Pestaña de Modelado
elif page == "Modelado":
    st.header("Modelado y Evaluación")

    data_clean = data.dropna()
    le = LabelEncoder()
    data_clean['Tipo de Atleta_encoded'] = le.fit_transform(data_clean['Tipo de Atleta'])

    feature_cols = ['Edad', 'Peso', 'Frecuencia Cardiaca Basal', 'IMC',
                    'Volumen Sistólico', 'Umbral de Lactato', 'Fibras Lentas', 'Fibras Rápidas']
    X = data_clean[feature_cols]
    y = data_clean['Tipo de Atleta_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Regresión Logística": LogisticRegression(),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=3),
        "SVM": SVC(probability=True)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        results.append({"Modelo": name, "Accuracy": acc, "Recall": rec, "AUC": roc_auc})

        with st.expander(f"📌 Matriz de Confusión - {name}"):
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Realidad')
            ax.set_title(f'Matriz de Confusión - {name}')
            st.pyplot(fig)

        with st.expander(f"📈 Curva ROC - {name}"):
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel('FPR (Tasa de Falsos Positivos)')
            ax.set_ylabel('TPR (Tasa de Verdaderos Positivos)')
            ax.set_title(f'Curva ROC - {name}')
            ax.legend(loc='lower right')
            st.pyplot(fig)

        with st.expander(f"📉 Curva Precisión-Recall - {name}"):
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            fig, ax = plt.subplots()
            ax.plot(recall, precision, color='blue', lw=2)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precisión')
            ax.set_title(f'Curva Precisión-Recall - {name}')
            st.pyplot(fig)

    # Visualización del Árbol de Decisión
    with st.expander("🌳 Árbol de Decisión"):
        model_tree = DecisionTreeClassifier(max_depth=3)
        model_tree.fit(X_train_scaled, y_train)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(model_tree, filled=True, feature_names=feature_cols, class_names=le.classes_, ax=ax, rounded=True, fontsize=10)
        st.pyplot(fig)

    results_df = pd.DataFrame(results)
    st.subheader("📊 Comparación de Métricas")
    st.table(results_df)

# Pestaña de Predicción
elif page == "Predicción":
    st.header("Predicción de Tipo de Atleta")

    edad = st.sidebar.slider("Edad", 15, 50, 25, 1)
    peso = st.sidebar.slider("Peso (kg)", 50, 120, 70, 1)
    imc = st.sidebar.slider("IMC", 15.0, 35.0, 22.0, 0.1)
    fc_basal = st.sidebar.slider("Frecuencia Cardiaca Basal", 40, 100, 60, 1)
    volumen_sistolico = st.sidebar.slider("Volumen Sistólico", 50, 150, 80, 1)
    umbral_lactato = st.sidebar.slider("Umbral de Lactato", 2.0, 10.0, 4.0, 0.1)
    fibras_lentas = st.sidebar.slider("Fibras Lentas (%)", 0, 100, 50, 1)
    fibras_rapidas = st.sidebar.slider("Fibras Rápidas (%)", 0, 100, 50, 1)

    data_clean = data.dropna()
    le = LabelEncoder()
    data_clean['Tipo de Atleta_encoded'] = le.fit_transform(data_clean['Tipo de Atleta'])

    feature_cols = ['Edad', 'Peso', 'Frecuencia Cardiaca Basal', 'IMC',
                    'Volumen Sistólico', 'Umbral de Lactato', 'Fibras Lentas', 'Fibras Rápidas']
    X = data_clean[feature_cols]
    y = data_clean['Tipo de Atleta_encoded']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(probability=True)
    model.fit(X_scaled, y)

    if st.sidebar.button("Predecir tipo de atleta"):
        input_data = scaler.transform([[edad, peso, fc_basal, imc,
                                        volumen_sistolico, umbral_lactato, fibras_lentas, fibras_rapidas]])
        proba = model.predict_proba(input_data)

        predicted_class = np.argmax(proba)
        predicted_class_label = le.inverse_transform([predicted_class])[0]
        predicted_prob = proba[0][predicted_class]

        st.subheader("🏁 Resultado de la Predicción")
        st.success(f"El atleta es probablemente **{predicted_class_label}** con una probabilidad del **{predicted_prob:.2%}**")

        st.subheader(f"🏃‍♂️ Fondista - Probabilidad: {proba[0][le.transform(['Fondista'])[0]]:.2%}")
        st.subheader(f"🏃‍♀️ Velocista - Probabilidad: {proba[0][le.transform(['Velocista'])[0]]:.2%}")

        # Gráfico de probabilidades sin expander
        fig, ax = plt.subplots()
        bars = ax.barh(le.classes_, proba[0], color=['#1890ff', '#faad14'])
        ax.set_xlabel('Probabilidad')
        ax.set_title('Distribución de Probabilidades por Tipo de Atleta')
        st.pyplot(fig)



