import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Dummy dataset ---
data = {
    'payload_mass': [5000, 2000, 6000, 8000, 1000, 3000, 7500, 2500],
    'orbit': [1, 0, 1, 1, 0, 0, 1, 0],  
    'flight_number': [1, 5, 10, 15, 20, 25, 30, 35],
    'reused': [0, 1, 1, 0, 0, 1, 1, 0],
    'success': [0, 1, 1, 1, 0, 1, 1, 0]  
}

df = pd.DataFrame(data)

# --- Split data ---
X = df[['payload_mass', 'orbit', 'flight_number', 'reused']]
y = df['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- Train model ---
model = RandomForestClassifier()
model.fit(X_train, y_train)

# --- Test ---
y_pred = model.predict(X_test)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 2))

# --- Predict new rocket launch ---
new_launch = pd.DataFrame({'payload_mass':[4500], 'orbit':[1], 'flight_number':[40], 'reused':[1]})
prediction = model.predict(new_launch)
print("🚀 Predicted Launch Result:", "Success ✅" if prediction[0]==1 else "Failure ❌")
