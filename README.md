# StockAnalysisProject 📈

**A Python-based tool that tries to predict stock prices using Linear Regression**
URL: https://roymus-stockanalysisproject-main-e46is0.streamlit.app/
---

## Features

- Fetches historical stock data via YFinance API
- Tries to predict x days forward using Linear Regression on the relationship between X price, Y time

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




