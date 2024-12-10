# Car Price Prediction with Inflation Adjustment

## Pre-Trained Model
To use the pre-trained model instead of training a new one, download the `best_model.pkl` file from the following link:

[Download best_model.pkl](https://drive.google.com/file/d/1YOHWSmPlKK8Y4z1xujilawVc8Lg_aHDj/view?usp=drive_link)

Place the file in the project directory where the `main.py` script is located.

---

## Description
This project predicts future car prices for specific makes and models by accounting for inflation. The program utilizes machine learning models (Random Forest and Gradient Boosting) trained on historical car price data. A visual representation of predicted prices over a specified range of years (2025–2030) is also provided.

## Features
- Predicts car prices for any make and model in the dataset.
- Visualizes predicted prices over the years 2025–2030.
- Automatically adjusts prices for inflation using a 3% annual inflation rate.
- Saves trained models for future use.

---

## Prerequisites
Before running the program, ensure you have the following:

### Software Requirements
- **Python**: Version 3.8 or higher.

### Required Libraries
Install the necessary libraries using the following command:
```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

### Dataset Requirements
- **Dataset Name**: `synthetic_car_prices.csv`
- **File Path**: Ensure the dataset is located at:
  ```bash
  C:/Users/OWNER/Desktop/cs projects for resume/pythonProject/synthetic_car_prices.csv
  ```
- **Required Columns**:
  - `car_make`
  - `vehicle_style`
  - `year`
  - `msrp`

---

## How to Run the Program

### Step 1: Clone or Download the Repository
Clone this repository or download the code files.

### Step 2: Verify Dataset Path
Ensure the dataset path in the `main.py` file matches the location of your `synthetic_car_prices.csv` file.

### Step 3: Execute the Program
Run the program using the following command:
```bash
python main.py
```

### Step 4: Follow the User Prompts
- The program will display a list of available car makes. Enter a car make (e.g., `bmw`).
- It will then display the available models for the selected make. Enter the desired model (e.g., `sedan`).
- The program will predict and display prices for the selected car over the years 2025–2030 and show a graph of the predictions.

---

## Outputs

### Predicted Prices Table
- A table of predicted prices for the selected car make and model for each year from 2025 to 2030.

### Graph
- A line graph displaying predicted prices over the years 2025–2030.

### Saved Model
- If a model is trained, it will be saved to `best_model.pkl` for future use.

---

## Notes
- If the dataset or required columns are missing, the program will raise an error.
- To use a different dataset, update the file path and ensure the dataset contains the required columns.
- The program will load a pre-trained model (`best_model.pkl`) if available. If not, it will train a new model.

---

## Example
### User Input
```plaintext
Available car makes:
- Audi
- Chevrolet
- Volkswagen
- Bmw
- Kia
- Ford
- Mercedes
- Honda
- Toyota
- Hyundai

Enter the car make: bmw

Available models for Bmw:
- Suv
- Hatchback
- Coupe
- Sedan

Enter the car model: sedan
```

### Output
#### Predicted Prices Table
```plaintext
Predicted prices over years:
2025: $29,131.86
2026: $30,005.82
2027: $30,905.99
2028: $31,833.17
2029: $32,788.17
2030: $33,771.81
```

#### Graph
- A graph visualizing the price predictions from 2025 to 2030 will be displayed.

---

## Video Demo
Watch the video demonstration of the project in action:

[![Video Demo](https://img.youtube.com/vi/your_video_id/maxresdefault.jpg)](your_video_link)

---

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue for suggestions or improvements.

---

## License
This project is licensed under the MIT License.
