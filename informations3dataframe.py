
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Step 1: Read Excel file as a DataFrame
excel_file_path = 'final_results1.xlsx'  # Replace with your actual file path
df_results = pd.read_excel(excel_file_path)

# Calculate the overall mean of "Mean_Pixel_Value"
overall_mean_pixel_value = df_results['Mean_Pixel_Value'].mean()

# Calculate the residual for each instance
df_results['Pixel_Value_Residual'] = np.absolute(df_results['Mean_Pixel_Value'] - overall_mean_pixel_value)

# Select the features for anomaly detection
X = df_results[['Mean_Distance', 'Pixel_Value_Residual']]

# Fit the Isolation Forest model
clf = IsolationForest(contamination=0.25)  # You can adjust the contamination parameter
df_results['Is_Anomaly'] = clf.fit_predict(X)

# Examine the correlation between "Pixel_Value_Residual" and being an anomaly
corr_matrix = df_results[['Mean_Distance', 'Pixel_Value_Residual', 'Is_Anomaly']].corr()

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Filter non-anomaly data for polynomial regression
non_anomaly_data = df_results[df_results['Is_Anomaly'] == 1]

# Fit polynomial regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(non_anomaly_data[['Image_Index']])
regressor = LinearRegression()
regressor.fit(X_poly, non_anomaly_data['Mean_Distance'])

# Predicting the values
non_anomaly_data['Predicted_Mean_Distance'] = regressor.predict(X_poly)

# Visualize the results
plt.figure(figsize=(12, 6))

# Plot the actual Mean Distance values
plt.scatter(df_results['Image_Index'], df_results['Mean_Distance'], c=df_results['Is_Anomaly'], cmap='coolwarm', label='Anomoly Values')

# Plot the predicted Mean Distance values for non-anomalies
plt.plot(non_anomaly_data['Image_Index'], non_anomaly_data['Predicted_Mean_Distance'], color='green', linewidth=2, label='Polynomial Regression (Non-Anomalies)')

plt.xlabel('Image Index')
plt.ylabel('Mean Distance')
plt.title('Anomaly Detection and Polynomial Regression')
plt.legend()
plt.show()

# Calculate the square area (square of mean distance)
non_anomaly_data['Square_Area'] = non_anomaly_data['Predicted_Mean_Distance'] ** 2

# Plot the square area against image index
plt.figure(figsize=(12, 6))
plt.plot(non_anomaly_data['Image_Index'], non_anomaly_data['Square_Area'], label='Square Area (Square of Mean Distance)', color='purple', linewidth=2)
plt.xlabel('Image Index')
plt.ylabel('Square Area')
plt.title('Square Area Changes using Polynomial Regression')
plt.legend()
plt.show()

# Predicting the values
non_anomaly_data['Predicted_Mean_Distance'] = regressor.predict(X_poly)
non_anomaly_data = non_anomaly_data.reset_index()

percentiles = [0, 25, 50, 75, 95]

# Calculate the indices corresponding to the percentiles
indices = [int(percentile / 100 * len(non_anomaly_data)) for percentile in percentiles]

# Extract the 'Square_Area' values at the specified percentiles
percentile_values = non_anomaly_data['Square_Area'].iloc[indices].tolist()

# Calculate changes compared to the start
changes_compared_to_start = (np.array(percentile_values) - percentile_values[0]) / percentile_values[0] * 100

# Create a DataFrame to display the results
result_df = pd.DataFrame({
    'Percentile': percentiles,
    'Square_Area_at_Percentile': percentile_values,
    'Change_from_Start': changes_compared_to_start
})

# Display the result DataFrame
print(result_df)

# Create a DataFrame for the table
table_data = {'Percentile': percentiles, 'Square_Area': percentile_values, 'Change Compared to Start (%)': changes_compared_to_start}
table_df = pd.DataFrame(table_data)

# Print the table using tabulate
table_str = tabulate(table_df, headers='keys', tablefmt='fancy_grid', showindex=False)

# Display the table
print("Table showing Square Area at different percentiles:")
print(table_str)