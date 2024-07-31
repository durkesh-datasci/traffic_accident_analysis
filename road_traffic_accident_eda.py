import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium

#I have done this in google colab, make sure you pass your dataset correctly.
# Load the RTA Dataset
file_path = '/content/RTA Dataset.csv'  # replace with your actual file path
df = pd.read_csv(file_path)

# Ensure the necessary columns are present
required_columns = ['Time', 'Road_surface_conditions', 'Weather_conditions', 'Accident_severity', 'Area_accident_occured']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"The following required columns are missing: {missing_columns}")

# Data Cleaning and Preprocessing
df.dropna(subset=required_columns, inplace=True)
df['Time'] = pd.to_datetime(df['Time']).dt.hour

# Set the background color to black for visualizations
plt.style.use('dark_background')

# Exploratory Data Analysis (EDA)
# Distribution of Road Conditions
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Road_surface_conditions', palette='viridis')
plt.title('Distribution of Road Surface Conditions')
plt.xlabel('Road Surface Condition')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')
plt.show()

# Distribution of Weather Conditions
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Weather_conditions', palette='viridis')
plt.title('Distribution of Weather Conditions')
plt.xlabel('Weather')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')
plt.show()

# Distribution of Time of Day
plt.figure(figsize=(10, 6))
sns.histplot(df['Time'], bins=24, kde=False, color='cyan')
plt.title('Distribution of Accidents by Time of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()

# Accident Severity by Road Condition
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Road_surface_conditions', hue='Accident_severity', palette='viridis')
plt.title('Accident Severity by Road Surface Condition')
plt.xlabel('Road Surface Condition')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')
plt.legend(title='Accident Severity', loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
plt.show()

# Accident Severity by Weather
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Weather_conditions', hue='Accident_severity', palette='viridis')
plt.title('Accident Severity by Weather')
plt.xlabel('Weather')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')
plt.legend(title='Accident Severity', loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
plt.show()

# Accident Hotspots Visualization
# Creating a heatmap of accidents by area (if no lat/lon is available)
plt.figure(figsize=(12, 8))
sns.countplot(y='Area_accident_occured', data=df, order=df['Area_accident_occured'].value_counts().index, palette='viridis')
plt.title('Accidents by Area')
plt.xlabel('Number of Accidents')
plt.ylabel('Area')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()

# Identify Patterns
# Group by road condition, weather, and time of day
patterns = df.groupby(['Road_surface_conditions', 'Weather_conditions', 'Time']).size().reset_index(name='count')
print(patterns)

# Save the pattern data for further analysis
patterns.to_csv('accident_patterns.csv', index=False)