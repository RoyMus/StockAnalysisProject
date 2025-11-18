# StockAnalysisProject 📈

**A Python-based tool for analyzing and visualizing stock data, integrating technical indicators and AI-powered insights.**

---

## Features

- Fetches historical stock data via Alpha Vantage API
- Converts raw JSON to annotated `pandas.DataFrame`
- Visualizes:
  - Daily closing price
  - Trading volume
  - 7-day and 20-day moving averages
- Generates AI insights on stock trends using Google’s Gemini model
- Optionally wraps the pipeline in a **Streamlit app** for interactive exploration

---

## Installation

```bash
git clone https://github.com/RoyMus/StockAnalysisProject.git
cd StockAnalysisProject

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\activate   # Windows

pip install -r requirements.txt
```
---
Dependencies include: pandas, matplotlib, requests, pytz, google-generativeai, Pillow, streamlit
---
## ⚙️ Configuration
Alpha Vantage API Key

Sign up at AlphaVantage.co

Set it in stock_utility_handler.py:

python
Copy
Edit
API_KEY = "YOUR_ALPHA_VANTAGE_KEY"
Google Gemini API Key

Obtain from Google AI Studio

Set it in ai_insights_handler.py:
API_KEY = "YOUR_GOOGLE_AI_KEY"

---

## Project Structure
<pre>
.
├── stock_utility_handler.py    # Data fetching & visualization
├── ai_insights_handler.py      # Wraps Gemini model for insight generation
├── marketapp.py                # (Optional) Streamlit web interface
├── requirements.txt            # Python dependencies
└── README.md                   # This file
</pre>
---

##  Disclaimer
This tool is for educational and informational purposes only.
It does not constitute financial advice. Use insights responsibly, and consult a licensed financial advisor before making investment decisions.
---
## License
MIT License
---

Version	Date	Changes
0.1.0	2025‑07‑10	Initial release with core features
---

