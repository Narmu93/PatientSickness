from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
data = pd.read_csv('disease_data1.csv')
data.columns = data.columns.str.strip()

# Encode categorical columns
symptom_columns = ['Symptom1', 'Symptom2', 'Symptom3']
target_column = 'Target(Disease)'

symptom_encoder = LabelEncoder()
target_encoder = LabelEncoder()

# Encode symptoms
for col in symptom_columns:
    data[col] = symptom_encoder.fit_transform(data[col])

# Encode target
data[target_column] = target_encoder.fit_transform(data[target_column])

# Prepare data for training
X = data[symptom_columns]
y = data[target_column]

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Symptoms list and disease decoding
symptoms = sorted(symptom_encoder.classes_)  # Decode symptoms for dropdown
disease_decoder = {index: label for index, label in enumerate(target_encoder.classes_)}

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        age = request.form.get('age')  # Age isn't used directly in this example
        symptom1 = request.form.get('symptom1')
        symptom2 = request.form.get('symptom2')
        symptom3 = request.form.get('symptom3')

        # Encode symptoms for model input
        symptom1_encoded = symptom_encoder.transform([symptom1])[0]
        symptom2_encoded = symptom_encoder.transform([symptom2])[0]
        symptom3_encoded = symptom_encoder.transform([symptom3])[0]

        # Prepare input for the model
        sample_input = [[symptom1_encoded, symptom2_encoded, symptom3_encoded]]

        # Predict the disease
        prediction_encoded = model.predict(sample_input)[0]
        predicted_disease = disease_decoder[prediction_encoded]

        return render_template('index.html', symptoms=symptoms, prediction=predicted_disease)
    except Exception as e:
        # Handle generic errors without revealing the traceback
        return render_template('index.html', symptoms=symptoms, error="Invalid input or prediction failed.")

if __name__ == '__main__':
    app.run(debug=True)
