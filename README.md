# Soccer Forecast Agent

Multi-agent AI system for Premier League forecasting, qualitative news synthesis, and alert generation.

## Main Run Commands

Run the full pipeline with verbose LLM logging:

```bash
DEBUG_LLM=1 .venv/bin/python -m soccer_forecast_agent.main
```

Run the pipeline normally:

```bash
.venv/bin/python -m soccer_forecast_agent.main
```

## Notebook Workflow

Regenerate the notebook from source:

```bash
.venv/bin/python make_notebook.py
```

Execute the notebook into a fresh artifact:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute demo.ipynb --output demo_executed.ipynb
```

## PDF Export

Export the executed notebook directly to PDF:

```bash
.venv/bin/jupyter nbconvert --to webpdf demo_executed.ipynb --output demo_final_report --no-input
```

If `webpdf` is unavailable or Chromium is missing, export to HTML instead:

```bash
.venv/bin/jupyter nbconvert --to html demo_executed.ipynb --output demo_final_report --no-input
```

Then open `demo_final_report.html` in Chrome and print to PDF.

If Playwright Chromium is missing, install it with:

```bash
.venv/bin/python -m playwright install chromium
```

## Notes

- Live notebook cells depend on external APIs and can fall back to cached or stored results if quotas are exhausted.
- The alert spam-suppression check depends on the SQLite database state in `soccer_forecast.db`.
- For the safest demo run, make sure `.env` contains valid API keys before running the live pipeline.
