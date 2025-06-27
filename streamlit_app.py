# Revisi untuk menghindari error NaN saat mapping dan sorting dengan lebih aman

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Prediksi Kanker Payudara", layout="wide")
st.title("🔬 Prediksi Kekambuhan Kanker Payudara dengan Naive Bayes")

@st.cache_data
def load_raw_data():
    df = pd.read_csv("breast-cancer.data", header=None)
    df.columns = [
        'Class', 'Age', 'Menopause', 'TumorSize', 'InvNodes',
        'NodeCaps', 'DegMalig', 'Breast', 'BreastQuad', 'Irradiat'
    ]
    return df

df = load_raw_data()

# Mapping user-friendly
menopause_map = {
    "premeno": "Belum Menopause",
    "ge40": "Menopause ≥ 40 th",
    "lt40": "Menopause < 40 th"
}
degmalig_map = {
    "1": "Rendah (1)",
    "2": "Sedang (2)",
    "3": "Tinggi (3)"
}
breast_map = {
    "left": "Kiri",
    "right": "Kanan"
}
quad_map = {
    "left_low": "Kiri-Bawah",
    "left_up": "Kiri-Atas",
    "right_low": "Kanan-Bawah",
    "right_up": "Kanan-Atas",
    "central": "Tengah"
}

# Tampilkan data
st.subheader("📊 Preview Dataset")
st.dataframe(df.head())

# Encode
df_encoded = df.copy()
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df_encoded.drop("Class", axis=1)
y = df_encoded["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.subheader("📈 Evaluasi Model")
col1, col2 = st.columns(2)
with col1:
    st.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
with col2:
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Input form
st.subheader("🧪 Prediksi Kemungkinan Kekambuhan")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    age = col1.selectbox("Usia", sorted(df["Age"].dropna().unique()))

    meno_values = df["Menopause"].dropna().unique()
    meno_opts = sorted([menopause_map[v] for v in meno_values if v in menopause_map])
    meno = col1.selectbox("Status Menopause", meno_opts)

    size = col1.selectbox("Ukuran Tumor", sorted(df["TumorSize"].dropna().unique()))
    nodes = col2.selectbox("Jumlah Kelenjar Getah Bening Positif", sorted(df["InvNodes"].dropna().unique()))
    nodecaps = col2.selectbox("Kapsul Terinfeksi", sorted(df["NodeCaps"].dropna().unique()))

    deg_values = df["DegMalig"].dropna().astype(str).unique()
    deg_opts = sorted([degmalig_map[v] for v in deg_values if v in degmalig_map])
    deg = col2.selectbox("Derajat Keganasan", deg_opts)

    breast_values = df["Breast"].dropna().unique()
    breast_opts = sorted([breast_map[v] for v in breast_values if v in breast_map])
    breast = col3.selectbox("Sisi Payudara", breast_opts)

    quad_values = df["BreastQuad"].dropna().unique()
    quad_opts = sorted([quad_map[v] for v in quad_values if v in quad_map])
    quad = col3.selectbox("Kuadran Tumor", quad_opts)

    irradiat = col3.selectbox("Pernah Radiasi?", sorted(df["Irradiat"].dropna().unique()))

    rs = st.number_input("Recurrence Score (0–100)", min_value=0, max_value=100, step=1)

    submitted = st.form_submit_button("Prediksi")

    if submitted:

        # Mapping balik
        meno_rev = {v: k for k, v in menopause_map.items()}
        deg_rev = {v: k for k, v in degmalig_map.items()}
        breast_rev = {v: k for k, v in breast_map.items()}
        quad_rev = {v: k for k, v in quad_map.items()}

        input_dict = {
            "Age": age,
            "Menopause": meno_rev[meno],
            "TumorSize": size,
            "InvNodes": nodes,
            "NodeCaps": nodecaps,
            "DegMalig": deg_rev[deg],
            "Breast": breast_rev[breast],
            "BreastQuad": quad_rev[quad],
            "Irradiat": irradiat
        }
        encoded_input = [label_encoders[k].transform([str(v)])[0] for k, v in input_dict.items()]
        prediction = model.predict([encoded_input])[0]
        result_label = label_encoders["Class"].inverse_transform([prediction])[0]
        st.success(f"🧬 Hasil Prediksi: **{result_label.upper()}**")
        
        st.info(f"📈 Recurrence Score (RS): {rs}")
        if rs <= 25:
            st.write("✅ Risiko kambuh rendah–sedang. Keuntungan kemoterapi kecil.")
        else:
            st.write("⚠️ Risiko kambuh tinggi. Pertimbangkan kemoterapi.")