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

Dependencies include: pandas, matplotlib, requests, pytz, google-generativeai, Pillow, streamlit

⚙️ Configuration
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

python
Copy
Edit
API_KEY = "YOUR_GOOGLE_AI_KEY"
📦 Project Structure
text
Copy
Edit
.
├── stock_utility_handler.py    # Data fetching & visualization
├── ai_insights_handler.py      # Wraps Gemini model for insight generation
├── marketapp.py                # (Optional) Streamlit web interface
├── requirements.txt            # Python dependencies
└── README.md                   # This file
🧩 Usage
As a script
bash
Copy
Edit
python marketapp.py
This will:

Prompt for stock ticker and market (e.g., AAPL, NASDAQ)

Generate a plot image (plots/<ticker>_<market>.png)

Obtain AI commentary on the plot

Render results in Streamlit

As modules
python
Copy
Edit
from stock_utility_handler import StockAPI, StockAnalyzer
from ai_insights_handler import AIInsights

api = StockAPI(ALPHA_VANTAGE_KEY)
raw = api.get_stock_info("AAPL", "NASDAQ")

analyzer = StockAnalyzer()
df = analyzer.json_to_dataframe(raw, "AAPL", "NASDAQ")
analyzer.plot_stock_data(df, "AAPL", "NASDAQ", "out.png")

ai = AIInsights(GOOGLE_AI_KEY)
insights = ai.get_ai_insights("out.png", "AAPL", "NASDAQ")
print(insights)
🌟 Contribute
Contributions are welcome! You could help by:

🧠 Adding new technical indicators

🌍 Supporting additional markets (e.g. BSE, LSE)

💡 Improving visualization (e.g. candlestick charts)

🎨 Enhancing the Streamlit app UI/UX

🧪 Implementing more AI analysis modes (e.g. sentiment, fundamental)

⚠️ Disclaimer
This tool is for educational and informational purposes only.
It does not constitute financial advice. Use insights responsibly, and consult a licensed financial advisor before making investment decisions.

📄 License
MIT License

📞 Contact
Questions, feedback, or suggestions?
Reach out to @RoyMus via GitHub Discussions or Issues.

📝 Changelog
Version	Date	Changes
0.1.0	2025‑07‑10	Initial release with core features

markdown
Copy
Edit

---

### ✅ Next Steps

- Update the **Requirements** section if you add/remove dependencies
- Add screenshots or demo GIFs (e.g., from `marketapp.py`)
- Populate the **Changelog** as you evolve the project
- Consider adding:
  - A `Dockerfile`
  - CI tests
  - Coverage report
  - GitHub Actions workflow

This README offers a clean, structured launchpad—readable, informative, and welcoming for collaborators. Let me know if you’d like help integrating additional details or assets!
::contentReference[oaicite:0]{index=0}
