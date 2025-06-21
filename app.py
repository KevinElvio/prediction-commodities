from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime, timedelta
import os

app = Flask(__name__)


MODEL_PATH = 'model_new1.pkl'
SCALER_X_PATH = 'scaler_x.joblib'
SCALER_Y_PATH = 'scaler_y.joblib'

try:
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

TIMESTEPS = 120

def create_sequence_for_prediction(last_data_scaled, next_day_scaled):
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
        data = request.json
        user_input_date_str = data.get('target_date')

        if not user_input_date_str:
             return jsonify({'error': 'Tanggal target tidak ditemukan dalam request.'}), 400

        target_date = datetime.strptime(user_input_date_str, '%Y-%m-%d')

        df_history = pd.read_csv('dataset_clean.csv')
        df_history['Tanggal'] = pd.to_datetime(df_history['Tanggal'])
        df_history['Hari'] = df_history['Tanggal'].dt.day
        df_history['Bulan'] = df_history['Tanggal'].dt.month
        df_history['Tahun'] = df_history['Tanggal'].dt.year

        input_features = ['Hari', 'Bulan', 'Tahun']
        history_data = df_history[input_features].values

        if len(history_data) < TIMESTEPS:
             return jsonify({'error': f'Data history kurang dari {TIMESTEPS} hari yang dibutuhkan.'}), 400

        last_history_data = history_data[-TIMESTEPS:]

        last_history_scaled = scaler_x.transform(last_history_data)

        predicted_dates = []
        predicted_prices_scaled = [] 
        predicted_prices_real = []   

        current_input_sequence = last_history_scaled.copy()
        current_date = df_history['Tanggal'].iloc[-1] 


        while current_date < target_date:
            
            pred_scaled = model.predict(np.array([current_input_sequence]), verbose=0)
            predicted_prices_scaled.append(pred_scaled[0]) 

   
            current_date += timedelta(days=1)
            predicted_dates.append(current_date)

          
            next_date_features = [current_date.day, current_date.month, current_date.year]
            next_scaled = scaler_x.transform([next_date_features]) 
            current_input_sequence = create_sequence_for_prediction(current_input_sequence, next_scaled) 
        
        predicted_prices_real = scaler_y.inverse_transform(np.array(predicted_prices_scaled))

       
        results = []
        for i in range(len(predicted_dates)):
            results.append({
                'tanggal': predicted_dates[i].strftime('%Y-%m-%d'),
                'harga_prediksi': float(predicted_prices_real[i][0]) 
            })

        return jsonify({'predictions': results})

    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return jsonify({'error': 'Terjadi kesalahan saat melakukan prediksi.'}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
