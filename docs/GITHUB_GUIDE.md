# GitHub proof-of-work checklist

1. Create a public repository named `ai-iot-predictive-maintenance`.
2. Extract this ZIP. Review the README and ensure generated data is clearly marked synthetic.
3. From the project directory:
   ```bash
   git init
   git add .
   git commit -m "Build IoT predictive maintenance demo"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/ai-iot-predictive-maintenance.git
   git push -u origin main
   ```
4. Run the app and capture screenshots of fleet KPIs, risk ranking, and sensor trends. Add them to `docs/screenshots/` and embed them in README only after they are real screenshots from your app.
5. Add a short demo GIF/video (30–60 seconds), a clear problem statement, architecture diagram, setup steps, data dictionary, evaluation, and limitations.
6. Make meaningful commits as you implement improvements rather than one giant commit. Suggested sequence: `add synthetic telemetry generator`, `train baseline classifier`, `build streamlit dashboard`, `document limitations and setup`, `add tests and CI`.
7. Pin the repository on your profile. Add repository topics: `predictive-maintenance`, `iot`, `machine-learning`, `python`, `streamlit`, `industrial-analytics`.
8. In your resume, describe what you actually did. Example: “Built a Python/Streamlit predictive-maintenance demo using simulated IoT telemetry; trained and evaluated a Random Forest classifier; surfaced machine-level risk and sensor trends in an interactive dashboard.” Do not claim industrial deployment or real-world accuracy.

## Suggested evidence to publish
- Screenshot of the working dashboard
- A brief architecture diagram
- Metrics JSON and explanation of why ROC-AUC alone is insufficient
- Test output (`pytest -q`)
- A short note on how you would validate with real equipment data
