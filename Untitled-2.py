import streamlit as st
import os
import numpy as np
import gdown
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess


class CornDiseaseApp:
    def __init__(self):
        self.model_dir = "models"
        self.model_urls = {
            "resnet": "https://drive.google.com/uc?id=1jVmr1kHY8cDSYgJEnIQ-OhqcUV8cj-qM",
            "vgg": "https://drive.google.com/uc?id=1kKUN75slUQtEsqv8tULqAC-HIVy_OBU8",
            "inception": "https://drive.google.com/uc?id=12YT-eiq09i3B8gY60KBjkOnJFHEKBIob"
        }
        self.model_filenames = {
            "resnet": "resnet_best_model.h5",
            "vgg": "vgg_best_model.h5",
            "inception": "inception_best_model.h5"
        }
        self.image_target_sizes = {
            "resnet": (224, 224),
            "vgg": (224, 224),
            "inception": (224, 224)
        }
        self.class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
        self.models = self.load_models()

    @st.cache_resource(show_spinner=False, hash_funcs={"_thread.RLock": lambda _: None})
    def load_models(self):
        os.makedirs(self.model_dir, exist_ok=True)
        models = {}
        status = st.empty()

        for name in self.model_filenames:
            model_path = os.path.join(self.model_dir, self.model_filenames[name])

            if not os.path.exists(model_path):
                status.info(f"⬇️ Mengunduh model {name}...")
                try:
                    gdown.download(
                        url=self.model_urls[name],
                        output=model_path,
                        quiet=True,
                        fuzzy=True
                    )
                    status.success(f"✅ Model {name} berhasil diunduh.")
                except Exception as e:
                    status.error(f"Gagal mengunduh model {name}")
                    st.exception(e)
                    return None

            try:
                status.info(f"🧠 Memuat model {name}...")
                model = load_model(model_path)
                models[name] = model
                status.success(f"✅ Model {name} berhasil dimuat.")
            except Exception as e:
                status.error(f"Gagal memuat model {name}")
                st.exception(e)
                return None

        status.success("🎉 Semua model berhasil dimuat!")
        return models

    def preprocess_image(self, image_path, model_name):
        size = self.image_target_sizes.get(model_name)
        img = Image.open(image_path).convert("RGB")
        img = img.resize(size)
        img_array = np.expand_dims(np.array(img), axis=0)

        if model_name == "resnet":
            return resnet_preprocess(img_array)
        elif model_name == "vgg":
            return vgg_preprocess(img_array)
        elif model_name == "inception":
            return inception_preprocess(img_array)
        else:
            return img_array / 255.0

    def predict(self, image_path):
        predictions = []
        for name, model in self.models.items():
            img_processed = self.preprocess_image(image_path, name)
            pred = model.predict(img_processed)
            predictions.append(pred)

        ensemble = np.mean(predictions, axis=0)
        idx = np.argmax(ensemble)
        return self.class_names[idx], float(np.max(ensemble) * 100)

    def run(self):
        st.set_page_config(page_title="Deteksi Penyakit Jagung", page_icon="🌽")

        st.title("🌽 Deteksi Penyakit Daun Jagung")
        st.markdown("""
        Aplikasi ini menggunakan teknik *ensemble Convolutional Neural Network* (CNN)
        (ResNet50, VGG16, dan Inception V3) untuk mendeteksi penyakit daun jagung.
        """)

        if self.models is None or len(self.models) != len(self.model_filenames):
            st.error("❌ Gagal memuat model. Periksa koneksi atau struktur file.")
            st.stop()

        uploaded = st.file_uploader(
            "Unggah gambar daun jagung:",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:
            st.image(uploaded, caption="Gambar yang Diunggah", use_column_width=True)
            path = os.path.join(self.model_dir, "temp_image.jpg")
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())

            with st.spinner("Melakukan prediksi..."):
                try:
                    label, confidence = self.predict(path)
                    st.success("✅ Prediksi selesai!")
                    st.subheader(f"Hasil Deteksi: **{label}**")
                    st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")
                except Exception as e:
                    st.error("❌ Gagal memproses gambar.")
                    st.exception(e)
                finally:
                    if os.path.exists(path):
                        os.remove(path)


# ===============================
# 🚀 Jalankan Aplikasi
# ===============================
if __name__ == "__main__":
    app = CornDiseaseApp()
    app.run()
