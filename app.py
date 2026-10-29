
import os
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random.pkl"
DATASET_DIR = BASE_DIR / "data"
TARGET_COLUMN = "market_condition"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
dataset = None
feature_names = []
categorical_columns = set()
categorical_options = {}
startup_error = None


# ============================================================
# 2. LOAD MODEL AND DATASET
# ============================================================

def load_resources():
    """Load deployment files without downloading anything."""

    global model, dataset, feature_names
    global categorical_columns, categorical_options

    # Load the trained model from the project directory.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "random.pkl was not found in the project root."
        )

    # Only load pickle files that you trust.
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    if not hasattr(model, "predict"):
        raise TypeError("random.pkl does not contain a prediction model.")

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "The model does not contain feature_names_in_. "
            "Verify how the model was trained and saved."
        )

    feature_names = list(model.feature_names_in_)

    # Read the dataset bundled with the deployment.
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            "Dataset folder not found. Create a data folder "
            "and add your dataset CSV."
        )

    csv_files = list(DATASET_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found inside the data folder."
        )

    # If multiple CSV files exist, choose the largest.
    csv_path = max(csv_files, key=lambda path: path.stat().st_size)
    dataset = pd.read_csv(csv_path)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found "
            f"in {csv_path.name}."
        )

    missing_features = [
        feature
        for feature in feature_names
        if feature not in dataset.columns
    ]

    if missing_features:
        raise ValueError(
            f"Dataset is missing model features: {missing_features}"
        )

    # Identify raw categorical columns.
    categorical_columns = {
        feature
        for feature in feature_names
        if (
            pd.api.types.is_object_dtype(dataset[feature])
            or isinstance(dataset[feature].dtype, pd.CategoricalDtype)
            or isinstance(dataset[feature].dtype, pd.StringDtype)
        )
    }

    # Reconstruct sorted LabelEncoder-style category mappings.
    # These mappings must match the original training procedure.
    categorical_options = {}

    for feature in categorical_columns:
        values = (
            dataset[feature]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        values = sorted(values)

        if not values:
            raise ValueError(
                f"No category values found for '{feature}'."
            )

        categorical_options[feature] = values

    logger.info("Model loaded successfully.")
    logger.info("Dataset loaded from %s", csv_path)
    logger.info("Number of model features: %s", len(feature_names))


try:
    load_resources()
except Exception as exc:
    startup_error = str(exc)
    logger.exception("Application initialization failed.")


# ============================================================
# 3. HTML TEMPLATE, CSS AND JAVASCRIPT
# ============================================================

PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <title>MarketPulse AI | Market Predictor</title>

    <style>
        :root {
            --bg: #f4f7fc;
            --surface: #ffffff;
            --ink: #17213b;
            --muted: #69758c;
            --line: #e2e8f0;
            --primary: #4f46e5;
            --blue: #2563eb;
            --navy: #101a35;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            background: var(--bg);
            color: var(--ink);
            font-family: Inter, "Segoe UI", Arial, sans-serif;
        }

        header {
            background: var(--navy);
            color: white;
            padding: 19px 5%;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 15px;
        }

        .brand {
            font-size: 21px;
            font-weight: 850;
            letter-spacing: -.5px;
        }

        .brand span {
            color: #a5b4fc;
        }

        .header-note {
            color: #cbd5e1;
            font-size: 11px;
            letter-spacing: 1.1px;
        }

        main {
            max-width: 1240px;
            margin: 32px auto;
            padding: 0 22px 40px;
        }

        .hero {
            color: white;
            padding: 38px;
            border-radius: 24px;
            background: linear-gradient(
                125deg,
                #172554 0%,
                #3730a3 60%,
                #2563eb 100%
            );
            box-shadow: 0 18px 45px #17255418;
            margin-bottom: 24px;
        }

        .eyebrow {
            color: #c7d2fe;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1.7px;
            text-transform: uppercase;
        }

        .hero h1 {
            font-size: clamp(29px, 4vw, 43px);
            line-height: 1.15;
            letter-spacing: -1.2px;
            margin: 14px 0;
        }

        .hero p {
            max-width: 680px;
            color: #e0e7ff;
            font-size: 14px;
            line-height: 1.8;
            margin-bottom: 0;
        }

        .layout {
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(270px, .65fr);
            align-items: start;
            gap: 22px;
        }

        .card {
            min-width: 0;
            padding: 26px;
            border-radius: 20px;
            border: 1px solid var(--line);
            background: var(--surface);
            box-shadow: 0 8px 28px #15213b08;
        }

        .card h2 {
            font-size: 20px;
            margin: 0 0 7px;
        }

        .subtitle {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.7;
            margin: 0 0 22px;
        }

        .section-label {
            margin: 26px 0 13px;
            font-size: 11px;
            font-weight: 850;
            letter-spacing: 1.2px;
            color: var(--primary);
            text-transform: uppercase;
        }

        .fields {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }

        .field {
            min-width: 0;
        }

        label {
            display: block;
            margin-bottom: 7px;
            font-size: 12px;
            font-weight: 750;
            color: #34415c;
        }

        input, select {
            width: 100%;
            min-width: 0;
            border: 1px solid #dbe2ef;
            border-radius: 10px;
            padding: 12px;
            font-size: 13px;
            color: var(--ink);
            background: white;
            outline: none;
            transition: border .2s, box-shadow .2s;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px #4f46e51c;
        }

        .sample-buttons {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
        }

        button, .reset {
            border: 0;
            border-radius: 11px;
            padding: 13px 16px;
            font-size: 13px;
            font-weight: 800;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            transition: transform .15s, filter .15s;
        }

        button:hover, .reset:hover {
            transform: translateY(-2px);
        }

        .sample-buttons button {
            min-width: 0;
            padding: 13px 8px;
            line-height: 1.5;
            background: #eef2ff;
            color: #3730a3;
            border: 1px solid #c7d2fe;
        }

        .sample-buttons .bull {
            background: #ecfdf3;
            border-color: #a7f3d0;
            color: #047857;
        }

        .sample-buttons .recession {
            background: #fff1f2;
            border-color: #fecdd3;
            color: #be123c;
        }

        .sample-buttons .stable {
            background: #eff6ff;
            border-color: #bfdbfe;
            color: #1d4ed8;
        }

        .actions {
            display: flex;
            gap: 12px;
            margin-top: 24px;
        }

        .predict-button {
            flex: 1;
            color: white;
            background: linear-gradient(110deg, var(--primary), var(--blue));
        }

        .reset {
            color: #334155;
            background: #eef2f7;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .result-card {
            border: 1px solid #c7d2fe;
            background: linear-gradient(145deg, #ffffff, #f5f6ff);
        }

        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 10px;
            background: #e0e7ff;
            color: #3730a3;
            font-size: 10px;
            font-weight: 850;
            letter-spacing: .8px;
        }

        .result-value {
            font-size: clamp(23px, 3vw, 31px);
            font-weight: 900;
            letter-spacing: -.7px;
            color: #3730a3;
            overflow-wrap: anywhere;
            margin: 17px 0 10px;
        }

        .result-description {
            font-size: 13px;
            line-height: 1.8;
            color: var(--muted);
        }

        .info {
            margin-top: 18px;
            padding: 15px;
            border-radius: 12px;
            background: #f8fafc;
            color: var(--muted);
            font-size: 12px;
            line-height: 1.8;
        }

        .error {
            border: 1px solid #fecaca;
            border-radius: 12px;
            padding: 15px;
            color: #b42318;
            background: #fff1f0;
            font-size: 13px;
            line-height: 1.7;
            overflow-wrap: anywhere;
        }

        footer {
            margin-top: 28px;
            text-align: center;
            color: var(--muted);
            font-size: 12px;
            line-height: 1.8;
        }

        @media (max-width: 850px) {
            .layout {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 540px) {
            main {
                margin-top: 20px;
                padding: 0 13px 30px;
            }

            header {
                padding: 16px;
            }

            .header-note {
                display: none;
            }

            .hero {
                padding: 25px;
                border-radius: 18px;
            }

            .card {
                padding: 19px;
            }

            .fields, .sample-buttons {
                grid-template-columns: 1fr;
            }

            .actions {
                flex-direction: column;
            }

            .reset {
                min-height: 44px;
            }
        }
    </style>
</head>

<body>

<header>
    <div class="brand">MarketPulse <span>AI</span></div>
    <div class="header-note">MARKET CONDITION INTELLIGENCE</div>
</header>

<main>

    <section class="hero">
        <div class="eyebrow">Predictive analytics workspace</div>
        <h1>Understand the market<br>before the next move.</h1>
        <p>
            Explore market-condition predictions using company, hiring,
            financial, and workforce indicators.
        </p>
    </section>

    {% if startup_error %}

        <section class="card">
            <h2>Application setup required</h2>
            <p class="subtitle">
                The model or dataset could not be initialized.
            </p>

            <div class="error">{{ startup_error }}</div>

            <div class="info">
                Check that random.pkl is in the project root and the
                dataset CSV is included in the data folder.
            </div>
        </section>

    {% else %}

        <div class="layout">

            <section class="card">
                <h2>Market profile</h2>

                <p class="subtitle">
                    Load a random example or enter the feature values
                    manually, then generate a prediction.
                </p>

                <form method="post" action="/">

                    <div class="section-label">Quick-fill sample data</div>

                    <div class="sample-buttons">
                        <button
                            class="bull"
                            type="submit"
                            name="sample_class"
                            value="Bull Market"
                            formnovalidate>
                            ↗ Random Bull Market
                        </button>

                        <button
                            class="recession"
                            type="submit"
                            name="sample_class"
                            value="Recession"
                            formnovalidate>
                            ↘ Random Recession
                        </button>

                        <button
                            class="stable"
                            type="submit"
                            name="sample_class"
                            value="Stable"
                            formnovalidate>
                            → Random Stable
                        </button>
                    </div>

                    {% set ns = namespace(section='') %}

                    {% for feature in features %}

                        {% if feature in [
                            'company_name', 'industry', 'country',
                            'company_size'
                        ] %}
                            {% set section = 'Company profile' %}

                        {% elif feature in [
                            'layoffs_count', 'layoff_percentage',
                            'reason_for_layoffs', 'open_roles',
                            'hiring_trend', 'remote_jobs_percentage',
                            'top_hiring_role'
                        ] %}
                            {% set section = 'Hiring & workforce' %}

                        {% elif feature in [
                            'ai_automation_impact', 'ai_replacement_risk',
                            'ai_adoption_level', 'employee_sentiment',
                            'job_security_score'
                        ] %}
                            {% set section = 'AI & employee signals' %}

                        {% else %}
                            {% set section = 'Financial & time indicators' %}
                        {% endif %}

                        {% if section != ns.section %}
                            {% if not loop.first %}
                                </div>
                            {% endif %}

                            <div class="section-label">{{ section }}</div>
                            <div class="fields">

                            {% set ns.section = section %}
                        {% endif %}

                        <div class="field">
                            <label for="{{ feature }}">
                                {{ feature.replace('_', ' ').title() }}
                            </label>

                            {% if feature in categorical_options %}

                                <select
                                    id="{{ feature }}"
                                    name="{{ feature }}"
                                    required>

                                    <option
                                        value=""
                                        disabled
                                        {% if not submitted.get(feature) %}
                                            selected
                                        {% endif %}>
                                        Select {{ feature.replace('_', ' ') }}
                                    </option>

                                    {% for option in categorical_options[feature] %}
                                        <option
                                            value="{{ option }}"
                                            {% if submitted.get(feature) == option %}
                                                selected
                                            {% endif %}>
                                            {{ option }}
                                        </option>
                                    {% endfor %}

                                </select>

                            {% else %}

                                <input
                                    id="{{ feature }}"
                                    name="{{ feature }}"
                                    type="number"
                                    step="any"
                                    required
                                    value="{{ submitted.get(feature, '') }}"
                                    placeholder="Enter {{ feature.replace('_', ' ') }}">

                            {% endif %}
                        </div>

                    {% endfor %}

                    </div>

                    <div class="actions">
                        <button
                            class="predict-button"
                            type="submit"
                            name="predict"
                            value="1">
                            Generate prediction →
                        </button>

                        <a class="reset" href="/">Reset form</a>
                    </div>

                </form>
            </section>

            <aside>

                <section class="card result-card">
                    <span class="pill">MODEL OUTPUT</span>

                    {% if prediction is not none %}

                        <div class="result-value">{{ prediction }}</div>

                        <p class="result-description">
                            This is the class returned by your trained
                            model for the submitted feature values.
                        </p>

                    {% else %}

                        <h2 style="margin-top: 18px">
                            Your prediction appears here
                        </h2>

                        <p class="result-description">
                            Load a sample or complete the market profile,
                            then click Generate prediction.
                        </p>

                    {% endif %}
                </section>

                {% if error %}
                    <section class="error" style="margin-top: 16px">
                        {{ error }}
                    </section>
                {% endif %}

                <section class="card" style="margin-top: 18px">
                    <h2>About this model</h2>

                    <p class="subtitle">
                        MarketPulse AI uses your existing trained
                        Random Forest classifier. It does not retrain it.
                    </p>

                    <div class="info">
                        The quick-fill buttons select actual rows from
                        the corresponding dataset class. The selected
                        class is not forced as the prediction.
                    </div>
                </section>

            </aside>
        </div>

    {% endif %}

    <footer>
        MarketPulse AI · Model predictions are not financial advice.
    </footer>

</main>
</body>
</html>
"""


# ============================================================
# 4. FLASK ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():
    submitted = {}
    prediction = None
    error = None

    if request.method == "POST" and startup_error is None:
        try:
            sample_class = request.form.get("sample_class", "").strip()

            # ------------------------------------------------
            # RANDOM SAMPLE: FILL THE FORM ONLY
            # ------------------------------------------------
            if sample_class:

                target_values = (
                    dataset[TARGET_COLUMN].astype(str).str.strip()
                )

                available_classes = set(target_values.unique())

                if sample_class not in available_classes:
                    raise ValueError(
                        f"'{sample_class}' was not found in the dataset. "
                        f"Available classes: {sorted(available_classes)}"
                    )

                matching_rows = dataset[
                    target_values == sample_class
                ].dropna(subset=feature_names)

                if matching_rows.empty:
                    raise ValueError(
                        f"No complete feature rows are available "
                        f"for '{sample_class}'."
                    )

                # Randomly choose one real row from the requested class.
                sample = matching_rows.sample(n=1).iloc[0]

                for feature in feature_names:
                    value = sample[feature]

                    if feature in categorical_columns:
                        value = str(value)

                        if value not in categorical_options[feature]:
                            raise ValueError(
                                f"Category mapping mismatch for '{feature}'."
                            )

                    submitted[feature] = str(value)

                # The user clicks Generate prediction next.
                prediction = None

            # ------------------------------------------------
            # MANUAL INPUT: GENERATE A MODEL PREDICTION
            # ------------------------------------------------
            elif request.form.get("predict"):

                row = {}

                for feature in feature_names:
                    raw = request.form.get(feature, "").strip()

                    if not raw:
                        raise ValueError(
                            f"Please provide a value for '{feature}'."
                        )

                    submitted[feature] = raw

                    if feature in categorical_columns:

                        allowed = categorical_options.get(feature, [])

                        if raw not in allowed:
                            raise ValueError(
                                f"Unsupported category '{raw}' "
                                f"for '{feature}'."
                            )

                        # Reconstruct the sorted LabelEncoder mapping.
                        row[feature] = allowed.index(raw)

                    else:
                        try:
                            value = float(raw)
                        except ValueError:
                            raise ValueError(
                                f"'{feature}' must be a valid number."
                            )

                        if not np.isfinite(value):
                            raise ValueError(
                                f"'{feature}' must be a finite number."
                            )

                        row[feature] = value

                # Preserve the model's original feature order.
                input_df = pd.DataFrame(
                    [[row[name] for name in feature_names]],
                    columns=feature_names
                )

                result = model.predict(input_df)
                prediction = str(result[0])

            else:
                error = (
                    "Choose a random sample button or submit "
                    "the prediction form."
                )

        except Exception as exc:
            error = str(exc)
            logger.exception("Request processing failed.")

    return render_template_string(
        PAGE,
        features=feature_names,
        categorical_options=categorical_options,
        submitted=submitted,
        prediction=prediction,
        error=error,
        startup_error=startup_error,
    )


# ============================================================
# 5. LOCAL DEVELOPMENT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
