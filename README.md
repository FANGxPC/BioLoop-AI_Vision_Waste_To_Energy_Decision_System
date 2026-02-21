# HackSphere: AI-Vision_Waste_To_Energy_Decision_System

[![Python-Version-3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit-Version-1.32.2](https://img.shields.io/badge/Streamlit-1.32.2-orange.svg)](https://streamlit.io/)
[![Ultralytics-YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://ultralytics.com/)

An intelligent web application for optimizing Waste-to-Energy (WtE) plant operations. This tool uses computer vision to analyze incoming waste, calculate its energy potential in real-time, and provide actionable recommendations for processing.


## Key Features

- **AI Waste Analysis:** Uses a YOLOv11 model to instantly identify and segment different types of waste (plastic, paper, organic, etc.) from an uploaded image.
- **Dynamic Energy Calculation:** Calculates the Lower Heating Value (LHV) of the waste batch, providing a precise measure of its energy potential in MJ/kg.
- **Weather-Aware Logic:** Automatically adjusts moisture content calculations based on real-time weather data (e.g., "Monsoon" vs. "Dry Season"), leading to more accurate energy yield predictions.
- **Decision Support:** Provides clear, color-coded actions based on the final energy value:
    - **✅ Direct Combustion:** For high-energy, low-moisture waste.
    - **⚠️ Pre-Drying Required:** For energy-viable waste that is too wet for efficient combustion.
    - **🚨 Reject / Biogas:** For low-energy or energy-negative waste that is unsuitable for incineration.
- **Operational Dashboard:** A user-friendly interface built with Streamlit, featuring:
    - Side-by-side comparison of raw and AI-annotated images.
    - Interactive charts showing waste composition.
    - A detailed breakdown of the energy calculation.
- **History & Reporting:** Keeps a running log of all scans and decisions, which can be downloaded as a CSV report.

## How It Works

1.  **Upload Image:** The user uploads a photo of a waste batch.
2.  **AI Segmentation:** The YOLOv11 model processes the image to determine the percentage composition of different materials.
3.  **Energy Calculation:** The `energy_math` module calculates the total energy potential (LHV), factoring in the material type, composition percentage, and current weather conditions.
4.  **Actionable Insight:** The app displays the final energy value and recommends the best processing action.

## Tech Stack

- **ML / Computer Vision:** Ultralytics YOLOv8, OpenCV, Pillow
- **Web Framework:** Streamlit
- **Data & Plotting:** Pandas, Plotly Express
- **APIs:** Requests (for OpenWeatherMap)

## Setup and Installation

1.  **Prerequisites:**
    - Python 3.9+
    - Git

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/HackSphere.git
    cd HackSphere
    ```

3.  **Create a Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download Model Weights:**
    Ensure you have the trained model file `best.pt`. Place it in the following directory:
    ```
    ./weights/wasteland_model/WtE_Predictor/v3_final_refinement/weights/best.pt
    ```
    *If you don't have this file, you may need to train the model first or download pre-trained weights if available.*

## Running the Application

Once the setup is complete, run the Streamlit app from your terminal:

```bash
streamlit run app.py
```

The application will open in your web browser.

## Configuration

The application uses the OpenWeatherMap API to fetch weather data. The API key is currently hardcoded in `src/weather_api.py`. For production use, it is highly recommended to store it as an environment variable.

1.  Create a `.env` file in the root directory.
2.  Add your API key to the file:
    ```
    OPENWEATHER_API_KEY="your_api_key_here"
    ```
3.  Modify `src/weather_api.py` to load the key from the environment.

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── src/
│   ├── vision_engine.py    # Waste detection and composition logic
│   ├── energy_math.py      # LHV and energy potential calculations
│   └── weather_api.py      # Fetches weather data
├── weights/
│   └── .../best.pt         # Trained YOLO model weights
└── garbage_testing_dataset/ # Sample images for testing
```
