import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080E1A;
    color: #E4EAF4;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1627 0%, #0A1220 100%);
    border-right: 1px solid #1C2D4A;
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00E5C3 0%, #0099FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
}
.hero-sub {
    color: #6B85A8;
    font-size: 0.95rem;
    margin-top: 6px;
    font-weight: 300;
    letter-spacing: 0.03em;
}
.hero-divider {
    height: 2px;
    background: linear-gradient(90deg, #00E5C3, #0099FF, transparent);
    border: none;
    margin: 18px 0 28px 0;
    border-radius: 2px;
}
.metric-row { display: flex; gap: 14px; margin-bottom: 28px; }
.metric-card {
    flex: 1;
    background: linear-gradient(135deg, #0E1C30 0%, #111D2E 100%);
    border: 1px solid #1C2D4A;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #00E5C3;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: #5A7399;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}
.result-card {
    border-radius: 16px;
    padding: 32px 28px;
    text-align: center;
    margin-top: 8px;
    border: 1px solid;
}
.result-card.churn  { background: linear-gradient(135deg,#1A0E15,#200D18); border-color:#FF4560; box-shadow:0 0 40px rgba(255,69,96,.12); }
.result-card.safe   { background: linear-gradient(135deg,#0A1A14,#0C1E18); border-color:#00E5A0; box-shadow:0 0 40px rgba(0,229,160,.10); }
.result-card.medium { background: linear-gradient(135deg,#1A1508,#1E190A); border-color:#FFB800; box-shadow:0 0 40px rgba(255,184,0,.10); }
.result-label { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; margin:0; }
.result-label.churn  { color:#FF4560; }
.result-label.safe   { color:#00E5A0; }
.result-label.medium { color:#FFB800; }
.result-prob { font-size:3.4rem; font-family:'Syne',sans-serif; font-weight:800; margin:10px 0 4px; }
.result-caption { font-size:0.82rem; color:#6B85A8; letter-spacing:.06em; text-transform:uppercase; }
.result-message { margin-top:14px; font-size:.88rem; padding:10px 16px; border-radius:8px; font-weight:500; }
.result-message.churn  { background:rgba(255,69,96,.12);  color:#FF8099; }
.result-message.safe   { background:rgba(0,229,160,.10);  color:#00E5A0; }
.result-message.medium { background:rgba(255,184,0,.10);  color:#FFD166; }
.insight-card {
    background: #0E1C30;
    border: 1px solid #1C2D4A;
    border-radius: 12px;
    padding: 20px;
    margin-top: 16px;
}
.insight-card h4 {
    font-family:'Syne',sans-serif;
    font-size:.9rem;
    font-weight:700;
    color:#00E5C3;
    text-transform:uppercase;
    letter-spacing:.08em;
    margin:0 0 14px 0;
}
.insight-row {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:6px 0;
    border-bottom:1px solid #131F30;
    font-size:.83rem;
}
.insight-row:last-child { border-bottom:none; }
.insight-feat { color:#A8BDD4; }
.insight-val  { color:#00E5C3; font-weight:600; font-family:'Syne',sans-serif; }
.section-label {
    font-family:'Syne',sans-serif;
    font-size:.7rem;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:.12em;
    color:#00E5C3;
    margin:18px 0 8px 0;
}
div[data-testid="stButton"] > button {
    width:100%;
    background:linear-gradient(135deg,#00E5C3 0%,#0099FF 100%);
    color:#080E1A;
    font-family:'Syne',sans-serif;
    font-weight:700;
    font-size:1rem;
    padding:14px;
    border:none;
    border-radius:10px;
    cursor:pointer;
    letter-spacing:.04em;
    margin-top:10px;
    transition:opacity .2s;
}
div[data-testid="stButton"] > button:hover { opacity:.88; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    font-size:.8rem; color:#7A96B8; font-weight:500;
}
.batch-info {
    background: #0E1C30;
    border: 1px solid #1C2D4A;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.batch-info p { font-size:.85rem; color:#7A96B8; margin:0; line-height:1.7; }
.batch-info code {
    background:#131F30;
    color:#00E5C3;
    padding:1px 6px;
    border-radius:4px;
    font-size:.8rem;
}
.footer {
    text-align:center;
    color:#2A3D57;
    font-size:.75rem;
    margin-top:40px;
    padding-top:16px;
    border-top:1px solid #131F30;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("churn_model_pipeline.pkl")

try:
    pipeline = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

TRAINING_COLS = [
    "age", "gender", "region_category", "membership_category",
    "joined_through_referral", "preferred_offer_types", "medium_of_operation",
    "internet_option", "days_since_last_login", "avg_time_spent",
    "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet",
    "used_special_discount", "offer_application_preference",
    "past_complaint", "complaint_status", "feedback",
]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 18px 0;'>
        <p style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#00E5C3;margin:0;'>
            Customer Profile
        </p>
        <p style='font-size:.75rem;color:#4A6080;margin:4px 0 0 0;'>
            For single prediction — Tab 1
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Demographics</p>', unsafe_allow_html=True)
    age = st.slider("Age", min_value=10, max_value=90, value=35, step=1)
    gender = st.selectbox("Gender", ["F", "M"])
    region_category = st.selectbox("Region", ["City", "Town", "Village"])

    st.markdown('<p class="section-label">Membership</p>', unsafe_allow_html=True)
    membership_category = st.selectbox(
        "Membership Tier",
        ["No Membership", "Basic Membership", "Silver Membership",
         "Gold Membership", "Premium Membership", "Platinum Membership"]
    )
    joined_through_referral = st.selectbox("Joined via Referral?", ["Yes", "No"])
    preferred_offer_types = st.selectbox(
        "Preferred Offer Type",
        ["Gift Vouchers/Coupons", "Credit/Debit Card Offers", "Without Offers"]
    )

    st.markdown('<p class="section-label">Engagement</p>', unsafe_allow_html=True)
    medium_of_operation = st.selectbox("Device Used", ["Desktop", "Smartphone", "Both"])
    internet_option = st.selectbox("Internet Option", ["Wi-Fi", "Mobile_Data", "Fiber_Optic"])
    days_since_last_login = st.slider("Days Since Last Login", 0, 30, 5)
    avg_frequency_login_days = st.slider("Avg Login Frequency (days)", 0.0, 30.0, 10.0, step=0.5)
    avg_time_spent = st.slider("Avg Time Spent (mins)", 0.0, 3000.0, 300.0, step=10.0)

    st.markdown('<p class="section-label">Financial</p>', unsafe_allow_html=True)
    avg_transaction_value = st.number_input(
        "Avg Transaction Value (₦)", min_value=0.0, max_value=100000.0, value=25000.0, step=500.0
    )
    points_in_wallet = st.number_input(
        "Points in Wallet", min_value=0.0, max_value=2000.0, value=500.0, step=10.0
    )
    used_special_discount = st.selectbox("Used Special Discount?", ["Yes", "No"])
    offer_application_preference = st.selectbox("Applies for Offers?", ["Yes", "No"])

    st.markdown('<p class="section-label">Complaints & Feedback</p>', unsafe_allow_html=True)
    past_complaint = st.selectbox("Has Past Complaint?", ["Yes", "No"])
    complaint_status = st.selectbox(
        "Complaint Status",
        ["Not Applicable", "Solved", "Solved in Follow-up", "Unsolved"]
    )
    feedback = st.selectbox(
        "Customer Feedback",
        ["Products always in Stock", "Quality Customer Care", "User Friendly Website",
         "No reason specified", "Poor Website", "Poor Customer Support",
         "Poor Product Quality", "Too many ads"]
    )

    predict_btn = st.button("🔮  Predict Churn Risk")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">Customer Churn Intelligence</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">XGBoost · F1 94.01% · ROC-AUC 97.60% · 37,000 customer records</p>',
    unsafe_allow_html=True
)
st.markdown('<hr class="hero-divider">', unsafe_allow_html=True)

st.markdown("""
<div class="metric-row">
    <div class="metric-card"><div class="val">94.01%</div><div class="lbl">F1 Score</div></div>
    <div class="metric-card"><div class="val">97.60%</div><div class="lbl">ROC-AUC</div></div>
    <div class="metric-card"><div class="val">93.47%</div><div class="lbl">Accuracy</div></div>
    <div class="metric-card"><div class="val">36,992</div><div class="lbl">Training Records</div></div>
    <div class="metric-card"><div class="val">18</div><div class="lbl">Features</div></div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️  `churn_model_pipeline.pkl` not found. Make sure it's in the same folder as `app.py`.")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮  Single Prediction", "📂  Batch CSV Prediction"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        if predict_btn:
            input_data = pd.DataFrame([{
                "age": age, "gender": gender,
                "region_category": region_category,
                "membership_category": membership_category,
                "joined_through_referral": joined_through_referral,
                "preferred_offer_types": preferred_offer_types,
                "medium_of_operation": medium_of_operation,
                "internet_option": internet_option,
                "days_since_last_login": days_since_last_login,
                "avg_time_spent": avg_time_spent,
                "avg_transaction_value": avg_transaction_value,
                "avg_frequency_login_days": avg_frequency_login_days,
                "points_in_wallet": points_in_wallet,
                "used_special_discount": used_special_discount,
                "offer_application_preference": offer_application_preference,
                "past_complaint": past_complaint,
                "complaint_status": complaint_status,
                "feedback": feedback,
            }])

            churn_pred = pipeline.predict(input_data)[0]
            churn_prob = pipeline.predict_proba(input_data)[0][1]
            prob_pct   = churn_prob * 100

            if prob_pct >= 70:
                risk_class, icon, label, message = "churn", "🚨", "HIGH CHURN RISK", "Immediate retention intervention recommended."
            elif prob_pct >= 40:
                risk_class, icon, label, message = "medium", "⚡", "MEDIUM CHURN RISK", "Monitor closely and consider proactive engagement."
            else:
                risk_class, icon, label, message = "safe", "✅", "LOW CHURN RISK", "Customer appears stable. Continue current strategy."

            prob_color = '#FF4560' if risk_class == 'churn' else '#FFB800' if risk_class == 'medium' else '#00E5A0'

            st.markdown(f"""
            <div class="result-card {risk_class}">
                <p style="font-size:2.2rem;margin:0;">{icon}</p>
                <p class="result-label {risk_class}">{label}</p>
                <p class="result-prob" style="color:{prob_color}">{prob_pct:.1f}%</p>
                <p class="result-caption">Churn Probability</p>
                <p class="result-message {risk_class}">{message}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:20px;'>", unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:.78rem;color:#4A6080;text-transform:uppercase;"
                "letter-spacing:.08em;margin-bottom:6px;'>Churn Probability Gauge</p>",
                unsafe_allow_html=True
            )
            st.progress(int(prob_pct))
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='border:1px dashed #1C2D4A;border-radius:16px;padding:60px 30px;
                        text-align:center;margin-top:8px;'>
                <p style='font-size:2.5rem;margin:0;'>🔮</p>
                <p style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                          color:#2A4060;margin:12px 0 6px;'>No prediction yet</p>
                <p style='font-size:.82rem;color:#2A4060;'>
                    Fill in the customer profile on the left<br>and click Predict Churn Risk
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="insight-card">
            <h4>Top Churn Drivers (SHAP)</h4>
            <div class="insight-row"><span class="insight-feat">🏆 Points in Wallet</span><span class="insight-val">0.4607</span></div>
            <div class="insight-row"><span class="insight-feat">💳 Membership Category</span><span class="insight-val">0.3552</span></div>
            <div class="insight-row"><span class="insight-feat">💬 Customer Feedback</span><span class="insight-val">0.0401</span></div>
            <div class="insight-row"><span class="insight-feat">💰 Avg Transaction Value</span><span class="insight-val">0.0301</span></div>
            <div class="insight-row"><span class="insight-feat">📱 Medium of Operation</span><span class="insight-val">0.0091</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card" style="margin-top:14px;">
            <h4>Business Recommendations</h4>
            <div class="insight-row" style="display:block;padding:8px 0;">
                <p style="margin:0;font-size:.82rem;">🎯 <strong style="color:#E4EAF4;">Wallet Campaign</strong><br>
                <span style="color:#5A7399;">Target customers below wallet-points threshold</span></p>
            </div>
            <div class="insight-row" style="display:block;padding:8px 0;">
                <p style="margin:0;font-size:.82rem;">⬆️ <strong style="color:#E4EAF4;">Membership Upgrade Offers</strong><br>
                <span style="color:#5A7399;">Push Basic & No Membership customers to higher tiers</span></p>
            </div>
            <div class="insight-row" style="display:block;padding:8px 0;">
                <p style="margin:0;font-size:.82rem;">🛠️ <strong style="color:#E4EAF4;">Prioritise Complaint Resolution</strong><br>
                <span style="color:#5A7399;">Negative feedback customers are at immediate risk</span></p>
            </div>
            <div class="insight-row" style="display:block;padding:8px 0;border-bottom:none;">
                <p style="margin:0;font-size:.82rem;">📡 <strong style="color:#E4EAF4;">Real-Time Scoring</strong><br>
                <span style="color:#5A7399;">Flag high-risk customers daily using this pipeline</span></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if predict_btn:
            st.markdown("""
            <div class="insight-card" style="margin-top:14px;">
                <h4>Submitted Profile</h4>
            """, unsafe_allow_html=True)
            summary_items = {
                "Age": age, "Gender": gender, "Region": region_category,
                "Membership": membership_category,
                "Points in Wallet": f"{points_in_wallet:,.0f}",
                "Avg Transaction": f"₦{avg_transaction_value:,.0f}",
                "Past Complaint": past_complaint, "Feedback": feedback,
            }
            rows_html = ""
            for k, v in summary_items.items():
                rows_html += f"""
                <div class="insight-row">
                    <span class="insight-feat">{k}</span>
                    <span style="color:#E4EAF4;font-size:.82rem;">{v}</span>
                </div>"""
            st.markdown(rows_html + "</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV Prediction
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="batch-info">
        <p>
            Upload a CSV file containing multiple customers. The file must include these columns:<br><br>
            <code>age</code> · <code>gender</code> · <code>region_category</code> ·
            <code>membership_category</code> · <code>joined_through_referral</code> ·
            <code>preferred_offer_types</code> · <code>medium_of_operation</code> ·
            <code>internet_option</code> · <code>days_since_last_login</code> ·
            <code>avg_time_spent</code> · <code>avg_transaction_value</code> ·
            <code>avg_frequency_login_days</code> · <code>points_in_wallet</code> ·
            <code>used_special_discount</code> · <code>offer_application_preference</code> ·
            <code>past_complaint</code> · <code>complaint_status</code> · <code>feedback</code><br><br>
            Extra columns like <code>customer_id</code> are allowed and will be preserved in the output.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            missing_cols = [c for c in TRAINING_COLS if c not in df.columns]
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                X_batch = df[TRAINING_COLS].copy()
                probs   = pipeline.predict_proba(X_batch)[:, 1]
                preds   = pipeline.predict(X_batch)

                def risk_label(p):
                    if p >= 0.70:   return "🚨 High Risk"
                    elif p >= 0.40: return "⚡ Medium Risk"
                    else:           return "✅ Low Risk"

                results_df = df.copy()
                results_df["Churn Probability (%)"] = (probs * 100).round(2)
                results_df["Prediction"]            = ["Churn" if p == 1 else "No Churn" for p in preds]
                results_df["Risk Level"]            = [risk_label(p) for p in probs]

                total    = len(results_df)
                n_churn  = int((preds == 1).sum())
                n_safe   = int((preds == 0).sum())
                avg_prob = probs.mean() * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Customers",       total)
                c2.metric("Predicted to Churn",    n_churn)
                c3.metric("Predicted to Stay",     n_safe)
                c4.metric("Avg Churn Probability", f"{avg_prob:.1f}%")

                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                st.dataframe(results_df, use_container_width=True, hide_index=True)

                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️  Download Results as CSV",
                    data=csv_out,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        st.markdown("""
        <div style='border:1px dashed #1C2D4A;border-radius:16px;padding:60px 30px;
                    text-align:center;margin-top:8px;'>
            <p style='font-size:2.5rem;margin:0;'>📂</p>
            <p style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                      color:#2A4060;margin:12px 0 6px;'>No file uploaded yet</p>
            <p style='font-size:.82rem;color:#2A4060;'>
                Upload a CSV file above to run predictions<br>on multiple customers at once
            </p>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <strong style="color:#00E5C3;">Pius Victor</strong> ·
    Customer Churn Prediction · XGBoost Pipeline ·
    <a href="https://github.com/profpius" style="color:#0099FF;text-decoration:none;">GitHub</a> ·
    <a href="https://linkedin.com/in/victor-pius-4061a9332" style="color:#0099FF;text-decoration:none;">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
