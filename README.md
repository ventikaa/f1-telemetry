cat > README.md << 'EOF'
# F1 Telemetry Analytics & Strategy Dashboard

Interactive multi-season F1 telemetry analysis system with ML-powered lap time prediction and pit stop strategy simulation.

## Live Demo
[→ Launch Dashboard](https://your-app.streamlit.app)

## What it does
- Analyzes 59,000+ laps across 5 seasons (2018–2023)
- Predicts lap times using a Random Forest model (MAE: 0.841s)
- Simulates pit stop strategies and finds the optimal pit window
- Visualizes tire degradation, pace consistency, and driver comparisons

## Architecture
FastF1 API → pipeline.py → features.py → model.py → Streamlit dashboard

## Stack
| Layer | Tool |
|---|---|
| Data ingestion | FastF1 |
| Feature engineering | pandas / numpy |
| ML model | scikit-learn (Random Forest) |
| Strategy simulation | Custom Python |
| Dashboard | Streamlit + Plotly |

## Setup

```bash
git clone https://github.com/ventikaa/f1-telemetry
cd f1-telemetry
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline (only needed to refresh data)
python src/pipeline.py
python src/features.py
python src/model.py

# Launch dashboard
streamlit run dashboard/app.py
```

## Features
- **Overview** — KPIs, driver comparison, tire usage breakdown
- **Telemetry** — lap time progression, degradation curves, gap to leader
- **Insights** — auto-generated performance insights per race
- **Strategy Sim** — pit lap slider with full window sweep and optimal lap finder
EOF
