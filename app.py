# from flask import Flask, render_template, request, redirect, url_for
# import joblib
# import pandas as pd
# from datetime import datetime

# # Optional timezone-aware now() helper (uses zoneinfo if available)
# try:
#     from zoneinfo import ZoneInfo

#     def get_now():
#         return datetime.now(ZoneInfo("Asia/Kolkata"))
# except Exception:
#     def get_now():
#         return datetime.now()

# app = Flask(__name__)

# # Make now() available in templates so base.html can call {{ now().year }}
# @app.context_processor
# def inject_now():
#     return {"now": get_now}

# # Load model + features once (ensure paths are correct)
# model = joblib.load("rf_productivity_model.joblib")
# feature_names = joblib.load("rf_productivity_features.joblib")

# def predict_productivity(raw):
#     """
#     raw: dict with all feature values (including 'date' as a string)
#     returns: float prediction
#     """
#     df_raw = pd.DataFrame([raw])
#     df = df_raw.copy()

#     # Parse date flexibly (accepts yyyy-mm-dd from HTML date inputs and other common formats)
#     try:
#         df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
#     except Exception:
#         df["date"] = pd.to_datetime(df["date"], errors="coerce")

#     # If parsing failed, drop the date column to avoid errors
#     if df["date"].isnull().any():
#         df = df.drop(columns=["date"], errors="ignore")
#     else:
#         df = df.drop(columns=["date"], errors="ignore")

#     # One-hot encode categorical columns present in the model pipeline
#     df = pd.get_dummies(df, columns=["quarter", "department", "day"], drop_first=True)

#     # Reindex to match training features; fill missing columns with 0
#     df_processed = df.reindex(columns=feature_names, fill_value=0)

#     # Predict and return as float
#     pred = model.predict(df_processed)
#     return float(pred[0])

# def categorize_productivity(p):
#     """Return a human-friendly label for a numeric prediction score."""
#     try:
#         p = float(p)
#     except Exception:
#         return "Unknown"
#     if p < 0.5:
#         return "Low productivity"
#     elif p < 0.75:
#         return "Medium productivity"
#     else:
#         return "High productivity"

# @app.route("/")
# def home():
#     return render_template("home.html")

# @app.route("/about")
# def about():
#     return render_template("about.html")

# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         # Collect form values (add validation as needed)
#         try:
#             raw = {
#                 "date": request.form.get("date", ""),
#                 "quarter": request.form.get("quarter", ""),
#                 "department": request.form.get("department", ""),
#                 "day": request.form.get("day", ""),
#                 "team": int(request.form.get("team") or 0),
#                 "targeted_productivity": float(request.form.get("targeted_productivity") or 0.0),
#                 "smv": float(request.form.get("smv") or 0.0),
#                 "wip": int(request.form.get("wip") or 0),
#                 "over_time": int(request.form.get("over_time") or 0),
#                 "incentive": int(request.form.get("incentive") or 0),
#                 "idle_time": float(request.form.get("idle_time") or 0.0),
#                 "idle_men": int(request.form.get("idle_men") or 0),
#                 "no_of_style_change": int(request.form.get("no_of_style_change") or 0),
#                 "no_of_workers": int(request.form.get("no_of_workers") or 0),
#             }
#         except ValueError as e:
#             # If conversion fails, re-render the form with an error message (you can implement flashing)
#             return render_template("predict.html", error=f"Invalid input: {e}")

#         # Compute prediction
#         try:
#             pred = predict_productivity(raw)
#         except Exception as e:
#             # If prediction fails, show an error in the predict page (or handle differently)
#             return render_template("predict.html", error=f"Prediction error: {e}")

#         # Category label
#         label = categorize_productivity(pred)

#         # Render output template directly (safer than redirect with query params)
#         # We pass: prediction_score (float), prediction_notes (string), result_id (None for now)
#         prediction_notes = f"Category: {label}"
#         return render_template(
#             "output.html",
#             prediction_score=pred,
#             prediction_notes=prediction_notes,
#             result_id=None,
#         )

#     # GET -> show the prediction form
#     return render_template("predict.html")

# @app.route("/output")
# def output():
#     # Support direct GET to /output if someone uses it; try to read query params
#     prediction_score = request.args.get("prediction_score")
#     prediction_notes = request.args.get("prediction_notes", "")
#     result_id = request.args.get("result_id")

#     # Convert prediction_score to float if provided
#     try:
#         prediction_score = float(prediction_score) if prediction_score is not None else None
#     except Exception:
#         prediction_score = None

#     return render_template(
#         "output.html",
#         prediction_score=prediction_score,
#         prediction_notes=prediction_notes,
#         result_id=result_id,
#     )

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
from datetime import datetime
import traceback

app = Flask(__name__)

# --- helper to make now() available in templates (same one we used before) ---
try:
    from zoneinfo import ZoneInfo
    def get_now():
        return datetime.now(ZoneInfo("Asia/Kolkata"))
except Exception:
    def get_now():
        return datetime.now()

@app.context_processor
def inject_now():
    return {"now": get_now}

# --- load model/features (keep your original filenames) ---
model = joblib.load("rf_productivity_model.joblib")
feature_names = joblib.load("rf_productivity_features.joblib")

def predict_productivity(raw):
    df_raw = pd.DataFrame([raw])
    df = df_raw.copy()

    # parse date flexibly
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isnull().any():
        df = df.drop(columns=["date"], errors="ignore")
    else:
        df = df.drop(columns=["date"], errors="ignore")

    df = pd.get_dummies(df, columns=["quarter", "department", "day"], drop_first=True)
    df_processed = df.reindex(columns=feature_names, fill_value=0)
    return float(model.predict(df_processed)[0])

def categorize_productivity(p):
    try:
        p = float(p)
    except Exception:
        return "Unknown"
    if p < 0.5:
        return "Low productivity"
    elif p < 0.75:
        return "Medium productivity"
    else:
        return "High productivity"

# --- Health route to verify server is running ---
@app.route("/health")
def health():
    return "OK", 200

# --- Keep compatibility if templates link to /prediction ---
@app.route("/prediction")
def prediction_redirect():
    return redirect(url_for("predict"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         print("DEBUG: /predict POST received")
#         print("Form keys:", list(request.form.keys()))
#         # print posted values (be careful in prod with sensitive info)
#         for k, v in request.form.items():
#             print(f" - {k} = {v}")

#         try:
#             raw = {
#                 "date": request.form.get("date", ""),
#                 "quarter": request.form.get("quarter", ""),
#                 "department": request.form.get("department", ""),
#                 "day": request.form.get("day", ""),
#                 "team": int(request.form.get("team") or 0),
#                 "targeted_productivity": float(request.form.get("targeted_productivity") or 0.0),
#                 "smv": float(request.form.get("smv") or 0.0),
#                 "wip": int(request.form.get("wip") or 0),
#                 "over_time": int(request.form.get("over_time") or 0),
#                 "incentive": int(request.form.get("incentive") or 0),
#                 "idle_time": float(request.form.get("idle_time") or 0.0),
#                 "idle_men": int(request.form.get("idle_men") or 0),
#                 "no_of_style_change": int(request.form.get("no_of_style_change") or 0),
#                 "no_of_workers": int(request.form.get("no_of_workers") or 0),
#             }

#             pred = predict_productivity(raw)
#             label = categorize_productivity(pred)
#             prediction_notes = f"Category: {label}"
#             # Render output directly so we can see results (no redirect)
#             return render_template("output.html", prediction_score=pred, prediction_notes=prediction_notes, result_id=None)

#         except Exception as e:
#             # Print full traceback to terminal
#             traceback.print_exc()
#             # Show error message in the form page so you can see it in the browser
#             err_msg = f"Prediction failed: {str(e)}"
#             return render_template("predict.html", error=err_msg)
#     # GET
#     return render_template("predict.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # required fields that MUST be provided by the user
        required_fields = [
            "date", "quarter", "department", "day",
            "team", "targeted_productivity", "smv", "wip",
            "over_time", "incentive", "idle_time", "idle_men",
            "no_of_style_change", "no_of_workers"
        ]

        # collect raw form values (strings) so we can redisplay them when there's an error
        form_values = {k: request.form.get(k, "") for k in request.form.keys()}

        # find missing or blank required fields
        missing = [f for f in required_fields if not request.form.get(f)]

        if missing:
            err = f"Please fill all required fields: {', '.join(missing)}"
            # render the form again with the user's entered values and the error
            return render_template("predict.html", error=err, form_values=form_values)

        # at this point required fields exist — convert to the right types
        try:
            raw = {
                "date": request.form["date"],
                "quarter": request.form["quarter"],
                "department": request.form["department"],
                "day": request.form["day"],
                "team": int(request.form["team"]),
                "targeted_productivity": float(request.form["targeted_productivity"]),
                "smv": float(request.form["smv"]),
                "wip": int(request.form["wip"]),
                "over_time": int(request.form["over_time"]),
                "incentive": int(request.form["incentive"]),
                "idle_time": float(request.form["idle_time"]),
                "idle_men": int(request.form["idle_men"]),
                "no_of_style_change": int(request.form["no_of_style_change"]),
                "no_of_workers": int(request.form["no_of_workers"]),
            }
        except ValueError as e:
            # bad conversion (user typed letters into numeric field)
            err = f"Invalid numeric input: {e}"
            return render_template("predict.html", error=err, form_values=form_values)

        # compute prediction
        try:
            pred = predict_productivity(raw)
            label = categorize_productivity(pred)
            prediction_notes = f"Category: {label}"
            return render_template("output.html", prediction_score=pred, prediction_notes=prediction_notes, result_id=None)
        except Exception as e:
            # prediction error (model failure etc.)
            return render_template("predict.html", error=f"Prediction error: {e}", form_values=form_values)

    # GET
    return render_template("predict.html")

@app.route("/output")
def output():
    prediction_score = request.args.get("prediction_score")
    prediction_notes = request.args.get("prediction_notes", "")
    result_id = request.args.get("result_id")
    try:
        prediction_score = float(prediction_score) if prediction_score is not None else None
    except Exception:
        prediction_score = None
    return render_template("output.html", prediction_score=prediction_score, prediction_notes=prediction_notes, result_id=result_id)

if __name__ == "__main__":
    print("Starting Flask app (debug mode) — visit http://127.0.0.1:5000/health to confirm server is up")
    app.run(debug=True)
