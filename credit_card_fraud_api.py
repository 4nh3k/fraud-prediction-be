from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from flask_cors import CORS, cross_origin

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

# Helper function for standard scaling the features
def standard_scale_predict(features):
    """Standard scaling for features (including 'Time' and 'Amount')"""
    scaler = StandardScaler()
    time_amount_cols = ['Time', 'Amount']
    features[time_amount_cols] = scaler.fit_transform(features[time_amount_cols])
    return features

# Endpoint for making predictions
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Load the trained model
        model = tf.keras.models.load_model('credit_card_fraud_model.h5')
        print(model.summary())

        # Get the input data directly (assuming input is sent as a form or as key-value pairs)
        # get data from request body
        print(request)
        data = request.get_json()
        
        # data looks like this: {"data":"1,-1.35835406159823,-1.34016307473609,1.77320934263119,0.379779593034328,-0.503198133318193,1.80049938079263,0.791460956450422,0.247675786588991,-1.51465432260583,0.207642865216696,0.624501459424895,0.066083685268831,0.717292731410831,-0.165945922763554,2.34586494901581,-2.89008319444231,1.10996937869599,-0.121359313195888,-2.26185709530414,0.524979725224404,0.247998153469754,0.771679401917229,0.909412262347719,-0.689280956490685,-0.327641833735251,-0.139096571514147,-0.0553527940384261,-0.0597518405929204,378.66"}
        # please extract time from first element, and amount from last element
        # and the rest of the elements are the features
        data = data['data']
        print(data)
        data = data.split(',')
        features = [float(i) for i in data]
        print(features)
                
        # Convert the feature list into a pandas DataFrame for easier scaling
        feature_df = pd.DataFrame([features], columns= ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])

        # Standard scale the 'features' (including Time and Amount)
        feature_df_scaled = standard_scale_predict(feature_df)

        # Make the prediction
        prediction = model.predict(feature_df_scaled)
        pred = 1 if prediction[0][0] > 0.5 else 0

        # Return the prediction (0 for normal, 1 for fraud)
        return jsonify({"data": pred}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint for training the model
@app.route('/train', methods=['POST'])
@cross_origin()
def train():
    try:
        # Load and prepare the dataset
        data = pd.read_csv("C:\\Users\\Tien Anh\\Downloads\\creditcard.csv")
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Standard scaling for training data (including 'Time' and 'Amount')
        scaler = StandardScaler()
        time_amount_cols = ['Time', 'Amount']
        X[time_amount_cols] = scaler.fit_transform(X[time_amount_cols])

        # Split the data into training and testing
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Create and compile the model
        model = tf.keras.models.Sequential([
            layers.Dense(30, input_shape=(30,), activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

        # Save the trained model
        model.save('credit_card_fraud_model.h5')

        return jsonify({"message": "Model trained and saved successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
