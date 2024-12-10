import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


INFLATION_RATE = 0.03  # 3% annual inflation rate


def adjust_price_by_inflation(price, start_year, end_year):
    years = end_year - start_year
    return price * ((1 + INFLATION_RATE) ** years)


def load_data(csv_path):
    print("Loading data...")
    data = pd.read_csv(csv_path)
    data['car_make'] = data['car_make'].str.lower()
    data['vehicle_style'] = data['vehicle_style'].str.lower()
    print("Data loaded successfully!")
    return data


def preprocess_data(data):
    print("Preprocessing data...")
    if "year" not in data.columns or "msrp" not in data.columns:
        raise ValueError("Required columns ('year', 'msrp') not found.")

    data["Inflation_Adjusted_MSRP"] = data.apply(
        lambda row: adjust_price_by_inflation(row["msrp"], row["year"], 2024), axis=1
    )
    data = data.dropna(subset=["Inflation_Adjusted_MSRP"])
    X = data.drop(["msrp", "car_make", "vehicle_style", "Inflation_Adjusted_MSRP"], axis=1, errors="ignore")
    y = np.log(data["Inflation_Adjusted_MSRP"] + 1)
    X = pd.get_dummies(X, drop_first=True)
    print("Preprocessing completed!")
    return X, y, data


def train_models(X_train, y_train):
    print("Training models...")
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    print("Model training completed!")
    return trained_models


def evaluate_models(models, X_test, y_test):
    print("Evaluating models...")
    for name, model in models.items():
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")


def predict_price_over_years(model, input_data, car_make, car_model):
    filtered_data = input_data[
        (input_data["car_make"] == car_make) & (input_data["vehicle_style"] == car_model)
    ]
    if filtered_data.empty:
        print(f"No data found for make '{car_make}' and model '{car_model}'.")
        return None

    base_price = filtered_data["msrp"].mean()
    years = range(2025, 2031)
    predicted_prices = [adjust_price_by_inflation(base_price, 2024, year) for year in years]

    plt.figure(figsize=(10, 6))
    plt.plot(years, predicted_prices, marker='o', linestyle='-', color='blue')
    plt.title(f"Predicted Prices for {car_make.capitalize()} {car_model.capitalize()} (2025–2030)")
    plt.xlabel("Year")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.show()

    return dict(zip(years, predicted_prices))


def display_available_options(input_data):
    print("\nAvailable car makes:")
    available_makes = input_data["car_make"].dropna().unique()
    for make in available_makes:
        print(f"- {make.capitalize()}")

    car_make = input("\nEnter the car make: ").strip().lower()
    if car_make not in available_makes:
        print("Car make not found in the dataset.")
        return None, None

    available_models = input_data[input_data["car_make"] == car_make]["vehicle_style"].dropna().unique()
    print(f"\nAvailable models for {car_make.capitalize()}:")
    for model in available_models:
        print(f"- {model.capitalize()}")

    car_model = input("\nEnter the car model: ").strip().lower()
    return car_make, car_model


def main():
    csv_path = "C:/Users/OWNER/Desktop/cs projects for resume/pythonProject/synthetic_car_prices.csv"
    data = load_data(csv_path)

    X, y, input_data = preprocess_data(data)

    if X is None or y is None:
        print("Error during preprocessing. Exiting.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_filename = "best_model.pkl"
    try:
        model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")
    except FileNotFoundError:
        models = train_models(X_train, y_train)
        evaluate_models(models, X_test, y_test)
        model = models["GradientBoosting"]  # Choose the best model
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}")

    car_make, car_model = display_available_options(input_data)

    if car_make and car_model:
        predicted_prices = predict_price_over_years(model, input_data, car_make, car_model)
        if predicted_prices:
            print("\nPredicted prices over years:")
            for year, price in predicted_prices.items():
                print(f"{year}: ${price:,.2f}")
        else:
            print("Prediction could not be completed. Please check the input and try again.")

if __name__ == "__main__":
    main()
