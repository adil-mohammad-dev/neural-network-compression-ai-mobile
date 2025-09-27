import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
from imblearn.over_sampling import SMOTE


# Load dataset
df = pd.read_csv('nn_compression.csv')  # Replace with your actual file
print(df['Technique'].unique())

# Feature columns
features = ['Model', 'Use_Case', 'Pre_Compression_Accuracy (%)',
            'Post_Compression_Accuracy (%)', 'Model_Size_MB_Before',
            'Model_Size_MB_After', 'Inference_Time_Before (ms)',
            'Inference_Time_After (ms)', 'Energy_Consumption_After (mWh)','Device_Type','Network_Availability' ]

# Target variable
target = 'Technique' # Example target (you can change it)

# Encode categorical features
label_encoder = LabelEncoder()


df['Technique'] = label_encoder.fit_transform(df['Technique'])
print(df['Technique'])

df['Model'] = label_encoder.fit_transform(df['Model'])
df['Use_Case'] = label_encoder.fit_transform(df['Use_Case'])
df['Device_Type'] = label_encoder.fit_transform(df['Device_Type'])

df['Network_Availability'] = label_encoder.fit_transform(df['Network_Availability'])
df.dropna(inplace=True)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Flask setup
app = Flask(__name__)
app.secret_key = '371023ed2754119d0e5d086d2ae7736b'




@app.route('/')
def home():
    return render_template('home.html')



@app.route('/run_rf')
def run_rf():
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)*100
    joblib.dump(rf_model,'rf_model.joblib')
    return render_template('dashboard.html', algorithm='Random Forest', accuracy=accuracy)

@app.route('/run_dt')
def run_dt():
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)*100
    
    joblib.dump(dt_model,'dt_model.joblib')
    return render_template('dashboard.html', algorithm='Decision Tree', accuracy=accuracy)

@app.route('/run_mlp')
def run_mlp():
    
    df.drop(columns=['Location'],inplace=True)

    
    y = df['Technique']  
    X = df.drop(columns=['Technique'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sm = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Train MLP model
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), 
                          activation='relu',
                          solver='adam',
                          learning_rate='adaptive',
                          max_iter=500,
                            verbose=True,
                          random_state=42)
    mlp_model.fit(X_train_scaled, y_train_resampled)

    # Predict and evaluate
    y_pred = mlp_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) * 100
    joblib.dump(mlp_model,'mlp_model.joblib')
    return render_template('dashboard.html', algorithm='Neural Network', accuracy=accuracy)
    
@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/submit_signup', methods=['POST'])
def submit_signup():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    
    # You can also add logic here to save the user to a database
    
    message = f"Registration successful for {username}!"
    return render_template('signup.html', message=message, username=username)


@app.route('/submit_login', methods=['POST'])
def submit_login():
    username = request.form.get('username')
    password = request.form.get('password')
    session['username'] = username
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        M = request.form['model']
        U = request.form['Use_Case']
        P = float(request.form['pre_accuracy'])
        Po = float(request.form['post_accuracy'])
        MB = float(request.form['size_before'])
        MA = float(request.form['size_after'])
        Inb = float(request.form['inference_before'])
        Ina = float(request.form['inference_after'])
        EnA = float(request.form['energy_consumption'])
        De = request.form['Device_Type']
        Net = request.form['network_availability']

    
        data={'Model':M,'Use_Case':U,'Pre_Compression_Accuracy (%)':P,
              'Post_Compression_Accuracy (%)':Po,'Model_Size_MB_Before':MB,
              'Model_Size_MB_After':MA,'Inference_Time_Before (ms)':Inb,'Inference_Time_After (ms)':Ina,
              'Energy_Consumption_After (mWh)':EnA,'Device_Type':De,'Network_Availability':Net}

        # Prepare DataFrame
        input_df = pd.DataFrame([data])
        print(input_df.dtypes)
        # Choose model
        model=joblib.load('rf_model.joblib')
        prediction = model.predict(input_df)[0]

        
        print(prediction)
        return render_template('result.html', prediction=prediction)
    return render_template('predict.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
