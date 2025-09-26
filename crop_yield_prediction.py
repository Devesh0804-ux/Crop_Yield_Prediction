# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
data = pd.read_csv("dataset.csv")  # Ensure you have columns like: Temp,Rainfall,Humidity,Soil_pH,Fertilizer,Yield,Crop

# Step 3: Preprocessing
X = data.drop("Yield", axis=1)
y = data["Yield"]

# Step 4: Divide & Conquer Approach
if 'Crop' in X.columns:
    subsets = X['Crop'].unique()
else:
    subsets = ['All']

results = pd.DataFrame(columns=['Actual', 'Predicted', 'Crop'])

for crop in subsets:
    if crop != 'All':
        subset_X = X[X['Crop'] == crop].drop('Crop', axis=1)
        subset_y = y[X['Crop'] == crop]
    else:
        subset_X = X
        subset_y = y
    
    # Split subset
    X_train, X_test, y_train, y_test = train_test_split(subset_X, subset_y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Save results
    temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    temp_df['Crop'] = crop
    results = pd.concat([results, temp_df], ignore_index=True)

# Step 5: Evaluation
rmse = mean_squared_error(results['Actual'], results['Predicted'], squared=False)
r2 = r2_score(results['Actual'], results['Predicted'])
print(f"RMSE: {rmse:.2f}, R² Score: {r2:.2f}")

# Step 6: Visualization
plt.figure(figsize=(10,6))
sns.scatterplot(data=results, x='Actual', y='Predicted', hue='Crop', palette='tab10')
plt.plot([results['Actual'].min(), results['Actual'].max()],
         [results['Actual'].min(), results['Actual'].max()], 'k--', lw=2)
plt.title("Actual vs Predicted Crop Yield")
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.legend(title='Crop')
plt.show()
