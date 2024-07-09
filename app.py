from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('fuzzy_cmeans_model.pkl', 'rb') as model_file:
    cntr = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    luas_bangunan = float(request.form['luas_bangunan'])
    luas_tanah = float(request.form['luas_tanah'])
    kamar_tidur = float(request.form['kamar_tidur'])
    kamar_mandi = float(request.form['kamar_mandi'])
    garasi = float(request.form['garasi'])

    # Prepare the data for clustering
    data = np.array([[luas_bangunan, luas_tanah, kamar_tidur, kamar_mandi, garasi]])

    # Normalize the data (assuming normalization was used in training)
    data_normalized = (data - data.min()) / (data.max() - data.min())

    # Predict the cluster for the input data
    u, _, _, _, _, _, _ = cntr
    u_pred = np.dot(u, data_normalized.T).argmax(axis=0)[0]

    return render_template('index.html', result=u_pred)

if __name__ == '__main__':
    app.run(debug=True)