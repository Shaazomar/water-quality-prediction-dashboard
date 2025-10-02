# 💧 Water Quality Prediction Dashboard

## Project Overview
This project is a web application to predict water quality based on key physical and chemical features using a machine learning model. It provides easy-to-use interfaces to input water sample data, explain predictions with SHAP visuals, compare multiple samples, and monitor live simulated sensor readings.

## Features
- **Prediction:** Input water parameters and get predicted water quality categories.
- **Explainability:** Understand why the model predicts a certain class through SHAP waterfall and bar plots.
- **Comparison Mode:** Upload CSV files with multiple samples to compare predictions side by side.
- **Live Monitoring:** Simulate real-time water sensor readings and view trends and prediction counts.

## Technologies Used
- Programming Language: Python
- Web Framework: Streamlit
- ML Model: Random Forest Classifier (pretrained and loaded via joblib)
- Explainability: SHAP library
- Visualization: Matplotlib, Streamlit Charts
- Data: Water quality datasets for model training

## Installation

1. Clone this repository:

git clone https://github.com/yourusername/water-quality-prediction-dashboard.gitcd water-quality-prediction-dashboard


2. Create and activate a virtual environment (optional but recommended):

python3 -m venv venvsource venv/bin/activate  # On Windows use `venv\Scripts\activate`


3. Install required packages:
pip install -r requirements.txt


## Usage

Run the Streamlit app:
"streamlit run app.py".


Open your browser and go to `http://localhost:8501` to access the dashboard.

## File Structure

- `app.py` - Main Streamlit app script.
- `style.css` - Custom CSS for UI styling.
- `water_quality_rf_model.pkl` - Pretrained random forest model.
- `waterDataset.csv` - Dataset used for training and reference.
- `README.md` - Project overview and usage instructions.

## How to Contribute

Feel free to fork and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.

## Contact

For any questions, please contact Shaaz Omar (mailto:shazomar041@gmail.com).


