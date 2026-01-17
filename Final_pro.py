from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model_filename = 'riverwater1_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the scaler
scaler_filename = 'scaler1.pkl'
scaler = joblib.load(scaler_filename)

@app.route('/')
def index():
    return render_template('water_predict.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("FORM DATA RECEIVED:", request.form)

        ph = float(request.form['ph'])
        hardness = float(request.form['Hardness'])
        solids = float(request.form['Solids'])
        chloramines = float(request.form['Chloroamines'])
        sulfate = float(request.form['Sulfate'])
        conductivity = float(request.form['Conductivity'])
        organic_carbon = float(request.form['Organic_Carbon'])
        trihalomethanes = float(request.form['Trihalomethanes'])
        turbidity = float(request.form['Turbidity'])

        features = [[
            ph, hardness, solids, chloramines,
            sulfate, conductivity, organic_carbon,
            trihalomethanes, turbidity
        ]]

        print("FEATURES:", features)

        scaled_features = scaler.transform(features)
        print("SCALED:", scaled_features)

        prediction = model.predict(scaled_features)
        print("PREDICTION:", prediction)

        if prediction[0] == 1:
            output = "Water is Good – You can Drink ✅"
        else:
            output = "Do Not Drink This Water ❌"

        return render_template('results_water.html', prediction_text=output)

    except Exception as e:
        print("🔥 ERROR:", e)
        return render_template(
            'results_water.html',
            prediction_text=f"ERROR: {e}"
        )

if __name__ == '__main__':
    app.run(debug=True,port=5007)
