import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("df_limpio.csv")
label_tipo = LabelEncoder()
label_vecindario = LabelEncoder()
df["tipo_de_cuarto_encoded"] = label_tipo.fit_transform(df["tipo_de_cuarto"])
df["vecindario_encoded"] = label_vecindario.fit_transform(df["vecindario"])
X = df[["tipo_de_cuarto_encoded", "vecindario_encoded"]].values
y = df["precio"].values
model = KNeighborsRegressor(n_neighbors=2)
model.fit(X, y)
if "predicciones" not in st.session_state:
    st.session_state["predicciones"] = pd.DataFrame(columns=[
        "tipo_de_cuarto", "vecindario", "precio_predicho"
    ])
st.title("🔮 Predicción de Precio de Cuarto (KNN)")
st.write("Seleccione el tipo de cuarto y el vecindario para predecir el precio.")
tipo_sel = st.selectbox("Tipo de cuarto:", df["tipo_de_cuarto"].unique())
vec_sel = st.selectbox("Vecindario:", df["vecindario"].unique())
test_point = np.array([[
    label_tipo.transform([tipo_sel])[0],
    label_vecindario.transform([vec_sel])[0]
]])
pred = model.predict(test_point)[0]
st.subheader("💰 Precio Predicho")
st.success(f"**${pred:.2f}**")
if st.button("Guardar predicción"):
    nueva_fila = {
        "tipo_de_cuarto": tipo_sel,
        "vecindario": vec_sel,
        "precio_predicho": pred
    }
    st.session_state["predicciones"] = pd.concat(
        [st.session_state["predicciones"], pd.DataFrame([nueva_fila])],
        ignore_index=True
    )
    st.success("Predicción guardada!")

st.subheader("📊 Visualización de datos y vecinos más cercanos")

fig, ax = plt.subplots(figsize=(8, 5))

scatter = ax.scatter(
    X[:, 0], X[:, 1], c=y, cmap='viridis', s=100
)
ax.scatter(test_point[0, 0], test_point[0, 1], c='red', s=200, marker='X')

neighbors_idx = model.kneighbors(test_point, return_distance=False)
for i in neighbors_idx[0]:
    ax.scatter(X[i, 0], X[i, 1], edgecolor='black', s=150, alpha=0.5)

ax.set_xlabel("tipo_de_cuarto (codificado)")
ax.set_ylabel("vecindario (codificado)")
ax.set_title("KNN - Vecinos Más Cercanos")
fig.colorbar(scatter, label="Precio")

st.pyplot(fig)

st.subheader("📄 Historial de predicciones")
st.dataframe(st.session_state["predicciones"])
