# StockAnalysisProject 📈

**A Python-based tool for analyzing and visualizing stock data, integrating technical indicators and AI-powered insights.**

---

## 🚀 Features

- Fetches historical stock data via Alpha Vantage API
- Converts raw JSON to annotated `pandas.DataFrame`
- Visualizes:
  - Daily closing price
  - Trading volume
  - 7-day and 20-day moving averages
- Generates AI insights on stock trends using Google’s Gemini model
- Optionally wraps the pipeline in a **Streamlit app** for interactive exploration

---

## 🔧 Installation

```bash
git clone https://github.com/RoyMus/StockAnalysisProject.git
cd StockAnalysisProject

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\activate   # Windows

pip install -r requirements.txt
