# Concrete Compressive Strength Prediction

End-to-end machine-learning project for **Computational Intelligence (2307311L)**, T.Y. B.Tech, Division C, 2026-27.

> Given the amounts of cement, slag, fly ash, water, superplasticizer, coarse and fine aggregate in a concrete mix and its
> curing age, predict the **compressive strength in MPa** – and serve the model through a web app and a REST API.

**SDG addressed:** SDG 9 – Industry, Innovation and Infrastructure (also SDG 12). Screening mixes on a computer means fewer
trial batches and less material waste, and it makes it easier to test mixes in which fly ash or slag replace part of the cement.

## Team

| Name | PRN | Contribution |
|---|---|---|
| *(fill in)* | *(fill in)* | e.g. Data collection, EDA, preprocessing |
| *(fill in)* | *(fill in)* | e.g. Model training and hyper-parameter tuning |
| *(fill in)* | *(fill in)* | e.g. Evaluation, testing |
| *(fill in)* | *(fill in)* | e.g. Deployment, Docker, GitHub, documentation |

## Results

Dataset: UCI Concrete Compressive Strength, 1,030 rows → **1,005 after removing 25 duplicate rows**; 80 % train (804) / 20 % test (201).

| Model (tuned, with engineered features) | CV RMSE (MPa) | Test R² | Test RMSE (MPa) | Test MAE (MPa) |
|---|---|---|---|---|
| **XGBoost (selected)** | **4.21** | **0.941** | **4.21** | **2.66** |
| Gradient Boosting | 4.23 | 0.937 | 4.33 | 2.73 |
| SVR | 4.96 | 0.922 | 4.84 | 2.93 |
| Random Forest | 4.96 | 0.921 | 4.85 | 3.42 |
| Decision Tree | 6.51 | 0.871 | 6.21 | 4.61 |
| Linear Regression (baseline) | 6.81 | 0.816 | 7.40 | 5.70 |

- Selection used **cross-validation only**; the test set was used once. Cross-validated R² of the selected model is ≈ 0.93 (the more cautious estimate).
- **Feature engineering** (water/binder ratio, water/cement ratio, total binder, log-age) lifted Linear Regression from CV R² 0.59 to 0.82 and KNN from 0.65 to 0.84.
- Each prediction comes with a 90 % range of about ±6.5 MPa (it contained 93 % of the test mixes).

## Project structure

```
concrete-strength-ml/
├── data/                  Concrete_Data.xls + original readme (UCI, CC BY 4.0)
├── notebooks/
│   └── concrete_strength_pipeline.ipynb   EDA → preprocessing → models → tuning → evaluation → saves the model
├── src/
│   ├── data.py            loading, renaming, duplicate removal
│   ├── features.py        feature engineering (a scikit-learn transformer, saved inside the pipeline)
│   └── predict.py         input validation + prediction logic shared by the app, the API and the tests
├── app/streamlit_app.py   web app
├── api/main.py            FastAPI REST API
├── models/                concrete_pipeline.joblib (full pipeline) + metadata.json
├── reports/figures/       all plots (use them in the report, PPT and poster)
├── tests/                 pytest: validation, model behaviour, API, UI smoke test
├── Dockerfile, docker-compose.yml, requirements*.txt
```

## Run it

Requires **Python 3.11+** (developed on 3.13).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

| Task | Command |
|---|---|
| Re-run the whole analysis and re-create the model (about 3 minutes) | `jupyter nbconvert --to notebook --execute --inplace notebooks/concrete_strength_pipeline.ipynb` (or open the notebook and *Run All*) |
| Web app | `streamlit run app/streamlit_app.py` → http://localhost:8501 |
| REST API | `uvicorn api.main:app --reload` → docs at http://127.0.0.1:8000/docs |
| Tests | `pytest` |

If you see a pickle / version error when loading the model, your library versions differ from `requirements.txt`: run the notebook once to regenerate `models/`.

### REST API example

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"cement":280,"slag":0,"fly_ash":0,"water":190,"superplasticizer":0,"coarse_agg":970,"fine_agg":800,"age":28}'
# {"strength_mpa":25.2,"low_mpa":18.9,"high_mpa":31.9,"category":"Medium (20-40 MPa)", ...}
```
Invalid input (missing, non-numeric or outside the training range) returns HTTP 422 with a readable message.

## Deployment

**Option A – Streamlit Community Cloud (free, gives a public link for the demo)**
1. Push this folder to a public GitHub repository (include `models/` and `reports/figures/`).
2. Go to https://share.streamlit.io → *New app* → choose the repo and branch, set **Main file path** to `app/streamlit_app.py`.
3. In *Advanced settings* choose the newest Python version offered (3.11 or later) and deploy. It installs `requirements.txt` automatically. If the app reports a model-loading error, regenerate `models/` by running the notebook with the same library versions as the deployment and commit the new files.

**Option B – Docker (also covers the "scalability / environment setup" part of the rubric)**
```bash
docker build -t concrete-strength .
docker run -p 8501:8501 concrete-strength          # web app on http://localhost:8501
docker compose up --build                          # web app (8501) + REST API (8000)
```
The container image holds only what is needed to serve the model (no notebooks, tests or raw data); the dependency layer is cached for fast rebuilds, and the API and web app can be scaled independently.

## Testing and error handling

- 36 automated tests: input validation (missing / non-numeric / NaN / out-of-range), model sanity (strength rises with age, falls with more water), minimum-quality guard, API status codes, and a UI smoke test.
- The app limits every input to the range seen in training, shows clear messages instead of tracebacks, and warns about unusual water/binder ratios.

## Limitations

- Laboratory data from a single study; reliable only inside the training ranges (the app enforces them). Long curing ages (> 90 days) come from fewer, different mixes.
- Cement type, aggregate size and curing conditions are not in the data.
- A screening tool – not a substitute for laboratory tests or design standards such as IS 456.

## Data citation

I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks,"
*Cement and Concrete Research*, Vol. 28, No. 12, pp. 1797-1808 (1998).
Dataset: Yeh, I-C. (1998). *Concrete Compressive Strength* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PK67 (CC BY 4.0).
