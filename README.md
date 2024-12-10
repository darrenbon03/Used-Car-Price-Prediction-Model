Car Price Prediction with Inflation Adjustment
Description
This project predicts future car prices for specific makes and models by accounting for inflation. The machine learning models (Random Forest and Gradient Boosting) are trained on historical car price data. The program provides a visual representation of predicted prices over a specified range of years (2025–2030).

Prerequisites
Before running the program, ensure you have the following:

Python: Version 3.8 or higher.

Required Libraries:

pandas
numpy
matplotlib
scikit-learn
joblib
You can install these libraries by running:

bash
Copy code
pip install pandas numpy matplotlib scikit-learn joblib
Dataset:

Ensure the dataset (synthetic_car_prices.csv) is located in the specified path:
bash
Copy code
C:/Users/OWNER/Desktop/cs projects for resume/pythonProject/synthetic_car_prices.csv
The dataset should have the following columns:
car_make
vehicle_style
year
msrp
How to Run the Program
Clone or Download the Repository:

Clone this repository or download the code files.
Verify Dataset Path:

Confirm the dataset path in the code matches the location of synthetic_car_prices.csv.
Run the Program:

Execute the program by running the following command in the terminal:
bash
Copy code
python main.py
Follow User Prompts:

The program will display a list of available car makes. Enter the car make (e.g., bmw).
The program will then display the models available for the selected make. Enter the desired model (e.g., sedan).
The program will predict and display prices for the selected car over the years 2025–2030 and show a graph of the predictions.
Outputs
Predicted Prices Table:

A table of predicted prices for the selected car make and model for each year from 2025 to 2030.
Graph:

A graph displaying predicted prices over the years.
Saved Model:

If a model is trained, it will be saved to best_model.pkl for future use.
Notes
If the dataset or required columns are missing, the program will raise an error.
To use a different dataset, update the file path and ensure the dataset has the required columns.
The program loads a pre-trained model if available (best_model.pkl). If not found, it trains a new model.
