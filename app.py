from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
# Import model dan scaler yang sesuai
from tensorflow.keras.models import load_model # Jika pakai .h5
import pickle # Jika pakai .pkl untuk model
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

# Path ke model dan scaler
MODEL_PATH = 'model_new1.pkl' # Ganti jika pakai .pkl
SCALER_X_PATH = 'scaler_x.joblib'
SCALER_Y_PATH = 'scaler_y.joblib'

# Load model dan scaler saat aplikasi pertama kali dijalankan
try:
    # Jika menggunakan format .h5
    # model = load_model(MODEL_PATH)
    # Jika menggunakan format .pkl untuk model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    print("Model dan scaler berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model atau scaler: {e}")
    model = None
    scaler_x = None
    scaler_y = None

# Tentukan timesteps yang sama dengan saat melatih model
TIMESTEPS = 120 # Sesuaikan dengan nilai di Colab Anda

# Fungsi untuk membuat sequence input untuk prediksi
def create_sequence_for_prediction(last_data_scaled, next_day_scaled):
    """
    Membuat sequence input untuk memprediksi satu hari ke depan.
    last_data_scaled: Array numpy (timesteps, jumlah_fitur) dari data yang di-scaling
    next_day_scaled: Array numpy (1, jumlah_fitur) dari data hari berikutnya yang di-scaling
    """
    # Geser data lama ke atas dan tambahkan data hari berikutnya di akhir
    new_sequence = np.vstack((last_data_scaled[1:], next_day_scaled))
    return new_sequence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler_x is None or scaler_y is None:
        return jsonify({'error': 'Model atau scaler gagal dimuat.'}), 500

    try:
        data = request.json # Asumsikan input datang dalam format JSON
        user_input_date_str = data.get('target_date') # Tanggal target dari input user

        if not user_input_date_str:
             return jsonify({'error': 'Tanggal target tidak ditemukan dalam request.'}), 400

        target_date = datetime.strptime(user_input_date_str, '%Y-%m-%d')

        # Ambil data history yang dibutuhkan (minimal timesteps hari terakhir)
        # --- PENTING: Anda perlu memiliki akses ke data history asli di aplikasi Flask ---
        # Ini bisa berarti:
        # 1. Memuat file dataset_clean.csv lagi
        # 2. Menyimpan sebagian data history terakhir saat melatih model dan meloadnya di sini
        # Pilihan 1 lebih mudah jika file tidak terlalu besar.

        # Contoh menggunakan Pilihan 1: Load dataset lagi (pastikan path benar)
        df_history = pd.read_csv('dataset_clean.csv')
        df_history['Tanggal'] = pd.to_datetime(df_history['Tanggal'])
        df_history['Hari'] = df_history['Tanggal'].dt.day
        df_history['Bulan'] = df_history['Tanggal'].dt.month
        df_history['Tahun'] = df_history['Tanggal'].dt.year

        input_features = ['Hari', 'Bulan', 'Tahun']
        history_data = df_history[input_features].values

        # Ambil timesteps hari terakhir dari data history
        if len(history_data) < TIMESTEPS:
             return jsonify({'error': f'Data history kurang dari {TIMESTEPS} hari yang dibutuhkan.'}), 400

        last_history_data = history_data[-TIMESTEPS:]

        # Scaling data history terakhir
        last_history_scaled = scaler_x.transform(last_history_data)

        predicted_dates = []
        predicted_prices_scaled = [] # Simpan prediksi scaled dulu
        predicted_prices_real = []   # Simpan prediksi real

        current_input_sequence = last_history_scaled.copy()
        current_date = df_history['Tanggal'].iloc[-1] # Tanggal terakhir dari data history

        # Lakukan forecasting harian hingga mencapai tanggal target
        while current_date < target_date:
            # Prediksi satu hari ke depan
            pred_scaled = model.predict(np.array([current_input_sequence]), verbose=0)
            predicted_prices_scaled.append(pred_scaled[0]) # Ambil hasil prediksi scaled

            # Update tanggal saat ini
            current_date += timedelta(days=1)
            predicted_dates.append(current_date)

            # Siapkan input untuk prediksi hari berikutnya
            next_date_features = [current_date.day, current_date.month, current_date.year]
            next_scaled = scaler_x.transform([next_date_features]) # Scaling fitur tanggal berikutnya
            current_input_sequence = create_sequence_for_prediction(current_input_sequence, next_scaled) # Update sequence

        # Inverse transform semua hasil prediksi scaled
        predicted_prices_real = scaler_y.inverse_transform(np.array(predicted_prices_scaled))

        # Format hasil untuk response
        results = []
        for i in range(len(predicted_dates)):
            results.append({
                'tanggal': predicted_dates[i].strftime('%Y-%m-%d'),
                'harga_prediksi': float(predicted_prices_real[i][0]) # Ambil nilai float
            })

        return jsonify({'predictions': results})

    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return jsonify({'error': 'Terjadi kesalahan saat melakukan prediksi.'}), 500

if __name__ == '__main__':
    # Untuk development
    app.run(debug=True)

    # Untuk deployment (contoh dengan gunicorn)
    # from gunicorn.app.base import BaseApplication
    # class FlaskApplication(BaseApplication):
    #     def __init__(self, app, options=None):
    #         self.options = options or {}
    #         self.application = app
    #         super().__init__()
    #     def load_config(self):
    #         config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
    #         for key, value in config.items():
    #             self.cfg.set(key.lower(), value)
    #     def load(self):
    #         return self.application
    # if __name__ == '__main__':
    #     options = {
    #         'bind': '0.0.0.0:5000', # Atau port lain
    #         'workers': 1 # Sesuaikan dengan kebutuhan
    #     }
    #     FlaskApplication(app, options).run()