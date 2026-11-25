from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="Kenya Household Financial Health Tier API",
    description="Serves the Unbiased K-Means Financial Health Clustering Model to the Mobile App",
    version="1.0.0",
)

# Allow mobile apps / web emulators to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained bundle once when the server starts
bundle = joblib.load("financial_health_app_bundle.joblib")


class HouseholdPayload(BaseModel):
    manage_day2day: int = Field(..., ge=0, le=1, description="1 = Manages daily expenses, 0 = Struggles")
    R1A: int = Field(..., ge=0, le=1, description="1 = No liquidity shock in past 12M, 0 = Faced shock")
    nfhi_12: int = Field(..., ge=0, le=1, description="1 = Manages non-food expenses, 0 = Unable")
    R3A: int = Field(..., ge=0, le=1, description="1 = Pursuing a major financial goal, 0 = No")
    mobile_money_access: int = Field(..., ge=0, le=1, description="1 = Has M-Pesa/Airtel Money, 0 = No")
    Mobile_bank_usage: int = Field(..., ge=0, le=1, description="1 = Uses M-Shwari/KCB M-Pesa/Mobile Bank, 0 = No")
    Savings_formal: int = Field(..., ge=0, le=1, description="1 = Saves in Bank/SACCO/Mobile, 0 = No")
    Savings_informal: int = Field(..., ge=0, le=1, description="1 = Saves in Chama/Informal group, 0 = No")
    Loan_formal: int = Field(0, ge=0, le=1, description="1 = Has formal loan, 0 = No")
    Loan_informal: int = Field(0, ge=0, le=1, description="1 = Has informal/Chama/shopkeeper loan, 0 = No")
    defaulted: int = Field(0, ge=0, le=1, description="1 = Defaulted/late on loan in past 12M, 0 = No")
    NHIF: int = Field(..., ge=0, le=1, description="1 = Has NHIF/SHIF coverage, 0 = No")
    A08: int = Field(..., ge=0, le=1, description="1 = Urban residence, 0 = Rural residence")
    NHM: int = Field(..., ge=1, le=25, description="Household size (number of members)")
    A14i: int = Field(..., ge=16, le=105, description="Respondent age in years")
    Education: str = Field("Secondary", description="Highest education completed")
    B3B: str = Field("Agriculture", description="Primary livelihood source")


@app.get("/metadata")
def get_app_dropdown_options():
    """Returns valid dropdown options for Education and Primary Livelihood (B3B) for the mobile UI."""
    return {
        "optimal_k": bundle["optimal_k"],
        "discovered_tiers": bundle["tier_names"],
        "dropdown_options": {
            col: info["valid_options"] for col, info in bundle["category_maps"].items()
        },
    }


@app.post("/predict-tier")
def predict_tier(payload: HouseholdPayload):
    """Accepts mobile app JSON payload and returns the predicted tier, score, and nudges."""
    user_dict = payload.model_dump()
    row = {}
    for col in bundle["app_features"]:
        val = user_dict.get(col, None)
        if col in bundle["category_maps"]:
            cmap = bundle["category_maps"][col]["mapping"]
            default_val = bundle["category_maps"][col]["default"]
            row[col] = cmap.get(str(val).strip(), default_val)
        else:
            row[col] = float(val) if val is not None else float(bundle["feature_medians"][col])

    x_df = pd.DataFrame([row])[bundle["app_features"]]
    x_scaled = bundle["scaler"].transform(x_df)
    x_pca = bundle["pca"].transform(x_scaled)
    x_pca[:, 0] = x_pca[:, 0] * bundle["pc1_sign"]

    raw_cluster = int(bundle["kmeans"].predict(x_pca)[0])
    tier_id = int(bundle["cluster_remap"][raw_cluster])

    dists = np.linalg.norm(bundle["ordered_pca_centroids"] - x_pca[0], axis=1)
    inv_dists = np.exp(-dists)
    probs = inv_dists / inv_dists.sum()

    p02, p98 = bundle["pc1_bounds"]
    pc1_norm = np.clip((x_pca[0, 0] - p02) / (p98 - p02), 0.0, 1.0)

    target_tier_id = min(tier_id + 1, bundle["optimal_k"] - 1)
    target_centroid = bundle["tier_feature_means"][target_tier_id]

    nudges = []
    if row.get("defaulted", 0) == 1.0:
        nudges.append({
            "feature": "Loan Default / Late Repayment",
            "priority_gap": 99.0,
            "action": bundle["nudge_messages"]["defaulted"],
        })

    for col in bundle["binary_advisory_cols"]:
        if col in ["defaulted", "Loan_formal", "Loan_informal"]:
            continue
        if row.get(col, 0.0) == 0.0 and col in bundle["nudge_messages"]:
            nudges.append({
                "feature": bundle["questionnaire_labels"].get(col, col),
                "priority_gap": round(float(target_centroid.get(col, 0.0)), 1),
                "action": bundle["nudge_messages"][col],
            })

    nudges = sorted(nudges, key=lambda x: x["priority_gap"], reverse=True)[:3]

    return {
        "tier_number": tier_id + 1,
        "total_tiers": bundle["optimal_k"],
        "tier_name": bundle["tier_names"][tier_id],
        "financial_health_score": round(float(pc1_norm * 100), 1),
        "confidence_pct": round(float(probs[tier_id] * 100), 1),
        "next_tier_target": bundle["tier_names"][target_tier_id],
        "nudges": nudges,
    }