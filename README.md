# AI-Based Anomaly Detection Platform

A powerful web-based platform for detecting anomalies in datasets using advanced machine learning algorithms. Built with Streamlit and scikit-learn.

## Features

- **Interactive Dashboard**: View key metrics, recent activity, and platform overview
- **Dataset Upload & Analysis**: Upload CSV files and perform anomaly detection
- **Multiple Detection Methods**: Choose between Isolation Forest and One-Class SVM algorithms
- **Real-time Visualizations**: Interactive charts showing anomaly distributions and trends
- **Prediction Logs**: Comprehensive logging of all detection runs with downloadable reports
- **Data Persistence**: Uploaded datasets are saved for future reference

## Requirements

- Python 3.8 or higher
- Required Python packages:
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - plotly

## Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser and navigate to the local URL (usually `http://localhost:8501`)
3. Upload a CSV dataset in the "Upload & Analyze" tab
4. Select numeric columns for analysis
5. Choose a detection method and adjust parameters
6. Click "Detect Anomalies" to run the analysis
7. View results, visualizations, and download reports

## Project Structure

```
AI-Anomaly-Detection/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── sample_dataset.csv     # Sample dataset for testing
├── README.md             # This file
├── data/                 # Directory for uploaded datasets
└── .git/                 # Git repository
```

## Anomaly Detection Methods

### Isolation Forest
- Unsupervised algorithm that isolates anomalies by randomly partitioning data
- Effective for high-dimensional datasets
- Good at detecting anomalies without prior knowledge of normal behavior

### One-Class SVM
- Supervises learning algorithm that learns a decision boundary around normal data
- Useful when you have only normal data for training
- Effective for novelty detection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

Built with ❤️ using Streamlit and scikit-learn