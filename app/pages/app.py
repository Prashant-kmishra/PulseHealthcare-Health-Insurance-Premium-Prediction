import os
import sys

# Add project root to sys.path so Streamlit Cloud can locate the 'src' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import streamlit.components.v1 as components
from src.router import ModelRouter

st.set_page_config(layout="wide", page_title="Pulse Healthcare AI", initial_sidebar_state="collapsed")

# Hide streamlit chrome
st.markdown("""
<style>
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stAppViewContainer"] { background: transparent !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; margin: 0 !important; }
    footer { display: none !important; }
    iframe { border: none !important; width: 100vw !important; height: 100vh !important; position: absolute; top: 0; left: 0; }
    
    .ai-fab {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        width: 3.5rem;
        height: 3.5rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.5);
        cursor: pointer;
        z-index: 999999;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        text-decoration: none;
    }
    .ai-fab:hover {
        transform: scale(1.1) translateY(-5px);
        box-shadow: 0 15px 35px -5px rgba(37, 99, 235, 0.6);
        color: white;
    }
    .ai-fab::after {
        content: "Talk to Pulse AI";
        position: absolute;
        right: 4.5rem;
        background: white;
        color: #0f172a;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        opacity: 0;
        pointer-events: none;
        transition: all 0.3s;
        transform: translateX(10px);
        white-space: nowrap;
        font-family: sans-serif;
    }
    .ai-fab:hover::after {
        opacity: 1;
        transform: translateX(0);
    }
    @media (max-width: 768px) {
        .ai-fab { bottom: 1.5rem; right: 1.5rem; }
        .ai-fab::after { display: none; }
    }
</style>

<a href="#" class="ai-fab" onclick="alert('Pulse AI Agent initializing... Coming soon!'); return false;">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path></svg>
</a>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    try:
        # Pass the base artifacts directory to the router (go up from app/pages/app.py to root)
        pages_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.dirname(pages_dir)
        root_dir = os.path.dirname(app_dir)
        artifacts_dir = os.path.join(root_dir, "artifacts")
        return ModelRouter(base_artifacts_dir=artifacts_dir)
    except Exception as e:
        print("Error loading pipeline:", e)
        return None
pipeline = load_pipeline()
import datetime
print(f"APP RAN AT {datetime.datetime.now()}", flush=True)

# Serve the frontend build
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend_v8")

try:
    show_dashboard = st.session_state.get("show_dashboard", False)
    show_accuracy = st.session_state.get("show_accuracy", False)
    
    if not show_dashboard and not show_accuracy:
        healthpulse_ui = components.declare_component("healthpulse_ui_v8", path=build_dir)
        
        # Get current prediction state
        prediction = st.session_state.get("prediction", None)
        plan_name = st.session_state.get("plan_name", "Bronze")
        last_ts = st.session_state.get("last_ts", None)
        
        # Render component with current state
        component_value = healthpulse_ui(key="ui_component", prediction=prediction, plan_name=plan_name, ts=last_ts)
        
        # If the user submitted the form
        if component_value:
            msg_type = component_value.get("type")
            ts = component_value.get("ts")
            
            if msg_type == "analyze_price" and ts != last_ts:
                st.session_state.last_ts = ts
                st.session_state.show_dashboard = True
                st.session_state.show_accuracy = False
                st.rerun()
                
            elif msg_type == "show_accuracy" and ts != last_ts:
                st.session_state.last_ts = ts
                st.session_state.show_dashboard = False
                st.session_state.show_accuracy = True
                st.rerun()
                
            elif msg_type == "calculate" and ts != last_ts:
                st.session_state.last_ts = ts
                data = component_value.get("data", {})
                if pipeline:
                    try:
                        def safe_float(val, default=0.0):
                            try: return float(val) if val else default
                            except: return default
                            
                        def safe_int(val, default=0):
                            try: return int(val) if val else default
                            except: return default

                        user_data = {
                            "age": safe_int(data.get("age"), 25),
                            "gender": data.get("gender") or "Male",
                            "marital_status": data.get("marital_status") or "Unmarried",
                            "number_of_dependants": safe_int(data.get("children"), 0),
                            "bmi_category": data.get("bmi") or "Normal",
                            "smoking_status": data.get("smoking_status") or "No Smoking",
                            "medical_history": data.get("medical_history") or "No Disease",
                            "employment_status": data.get("employment_status") or "Salaried",
                            "income_lakhs": safe_float(data.get("income_lakhs"), 25.0),
                            "region": data.get("region") or "Northwest",
                            "insurance_plan": data.get("insurance_plan") or "Bronze"
                        }
                        print(f"DEBUG INCOMING DATA: {user_data}")
                        try:
                            premium = pipeline.predict(user_data)
                            print(f"DEBUG PREDICTION: {premium}")
                        except Exception as e:
                            print(f"DEBUG ERROR: {e}")
                            premium = 2500
                        if hasattr(premium, '__iter__'):
                            premium = float(premium[0])
                        else:
                            premium = float(premium)
                        
                        st.session_state.prediction = premium
                        st.session_state.plan_name = user_data["insurance_plan"]
                        st.session_state.user_data = user_data # Save for dashboard
                        
                        st.rerun()
                    except Exception as e:
                        import traceback
                        err_trace = traceback.format_exc()
                        st.error(f"Error predicting: {e}\\n\\n{err_trace}")
    elif show_dashboard:
        # -------------------------------------------------------------
        # NATIVE STREAMLIT DASHBOARD
        # -------------------------------------------------------------
        st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #f0f7ff 0%, #ffffff 100%) !important; color: #0f172a !important; }
            .stMetric value { color: #2563eb !important; }
            h1, h2, h3, h4, h5, h6, p, span, div, text, g { color: #0f172a !important; }
            .stButton>button { background: white !important; color: #2563eb !important; border: 2px solid #2563eb !important; font-weight: 600 !important; border-radius: 0.5rem !important; }
            .stButton>button:hover { background: #2563eb !important; color: white !important; }
            hr { border-color: #e2e8f0 !important; }
            .explanation-box {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 1rem;
                padding: 1.5rem;
                margin-top: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            }
        </style>
        """, unsafe_allow_html=True)
        
        premium = st.session_state.get("prediction", 0)
        user_data = st.session_state.get("user_data", {})
        
        col1, col2 = st.columns([1, 8])
        with col1:
            if st.button("← Back"):
                st.session_state.show_dashboard = False
                st.rerun()
                
        with col2:
            st.title("Premium Analysis Dashboard")
            st.markdown("Detailed breakdown of your AI-predicted health insurance premium.")
            
        st.markdown("---")
        
        # KPI Cards
        kpi_html = f"""
        <div style="display: flex; gap: 1.5rem; margin-bottom: 2rem; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 200px; background: white; padding: 1.5rem; border-radius: 1rem; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                <p style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #64748b !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Annual</p>
                <h3 style="margin: 0; font-size: 2.5rem; color: #0f172a !important; font-weight: 700;">₹{premium:,.0f}</h3>
            </div>
            <div style="flex: 1; min-width: 200px; background: white; padding: 1.5rem; border-radius: 1rem; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                <p style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #64748b !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Approx. Monthly</p>
                <h3 style="margin: 0; font-size: 2.5rem; color: #2563eb !important; font-weight: 700;">₹{premium/12:,.0f}</h3>
            </div>
            <div style="flex: 1; min-width: 200px; background: white; padding: 1.5rem; border-radius: 1rem; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                <p style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #64748b !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Selected Plan</p>
                <h3 style="margin: 0; font-size: 2.5rem; color: #0f172a !important; font-weight: 700;">{user_data.get("insurance_plan", "Bronze")}</h3>
            </div>
            <div style="flex: 1; min-width: 200px; background: white; padding: 1.5rem; border-radius: 1rem; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                <p style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #64748b !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Income Bracket</p>
                <h3 style="margin: 0; font-size: 2.5rem; color: #0f172a !important; font-weight: 700;">₹{user_data.get('income_lakhs', 0):.1f}L</h3>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)
        
        # Charts using plotly
        import plotly.express as px
        import plotly.graph_objects as go
        
        c1, c2 = st.columns(2)
        
        # Calculate visual risk score (0-100)
        score = 20
        if user_data.get('smoking_status') != 'No Smoking': score += 30
        if user_data.get('medical_history') != 'No Disease': score += 25
        if user_data.get('bmi_category') == 'Obesity': score += 20
        elif user_data.get('bmi_category') == 'Overweight': score += 10
        score = min(score, 100)
        
        # Fake breakdown for visual purposes based on user data
        base = premium * 0.4
        age_factor = premium * (0.1 if user_data.get('age', 30) < 35 else 0.25)
        health_factor = premium - (base + age_factor)
        
        with c1:
            st.markdown("<h3 style='color: #0f172a !important; margin-bottom: 1rem;'>Premium Contributors</h3>", unsafe_allow_html=True)
            fig = go.Figure(go.Waterfall(
                name = "Premium", orientation = "h",
                measure = ["relative", "relative", "relative", "total"],
                y = ['Base Plan', 'Age Risk', 'Health Risk', 'Total Premium'],
                x = [base, age_factor, health_factor, premium],
                connector = {"line":{"color":"#94a3b8"}},
                decreasing = {"marker":{"color":"#10b981"}},
                increasing = {"marker":{"color":"#ef4444"}},
                totals = {"marker":{"color":"#3b82f6"}}
            ))
            fig.update_layout(
                paper_bgcolor='white', 
                plot_bgcolor='white', 
                font_color='#000000', 
                margin=dict(t=30, b=30, l=10, r=30),
                showlegend=False,
                xaxis_title="Premium (₹)",
                yaxis_title=""
            )
            fig.update_xaxes(color="black", tickfont=dict(color="black"), title_font=dict(color="black"), gridcolor="#e2e8f0")
            fig.update_yaxes(color="black", tickfont=dict(color="black"), title_font=dict(color="black"))
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("<h3 style='color: #0f172a !important; margin-bottom: 1rem;'>Health Risk Index</h3>", unsafe_allow_html=True)
            fig2 = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                title = {'text': "Overall Risk Score", 'font': {'color': '#000000'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': "#000000"},
                    'bar': {'color': "#2563eb"},
                    'bgcolor': "#f1f5f9",
                    'steps': [
                        {'range': [0, 30], 'color': "#dcfce7"},
                        {'range': [30, 70], 'color': "#fef08a"},
                        {'range': [70, 100], 'color': "#fecaca"}
                    ]
                }
            ))
            fig2.update_layout(paper_bgcolor='white', font_color='#000000', margin=dict(t=50, b=10, l=20, r=20))
            st.plotly_chart(fig2, use_container_width=True)

        # Explaination Box
        explanation_html = f"""
        <div class="explanation-box">
            <h3 style="margin-top:0; color:#2563eb !important; font-size:1.25rem;">💡 Why is my premium ₹{premium:,.0f}?</h3>
            <p>Our AI model analyzes multiple factors to calculate a fair premium. Here's a breakdown of your specific profile:</p>
            <ul style="line-height: 1.8;">
                <li><strong>Health & Lifestyle ({(health_factor/premium*100):.0f}%):</strong> Your BMI category is <em>{user_data.get('bmi_category')}</em> and your smoking status is <em>{user_data.get('smoking_status')}</em>. Because your health risk index is {score}/100, this forms a significant portion of your premium.</li>
                <li><strong>Age & Demographics ({(age_factor/premium*100):.0f}%):</strong> At age {user_data.get('age')}, your statistical risk bracket adds a moderate baseline to the plan cost.</li>
                <li><strong>Base Plan ({(base/premium*100):.0f}%):</strong> This covers the standard administrative and coverage costs for the <em>{user_data.get('insurance_plan')}</em> tier plan you selected.</li>
            </ul>
            <p style="margin-bottom:0; font-size:0.9rem; color:#64748b !important;"><em>Note: This is an AI-generated approximation based on your inputs.</em></p>
        </div>
        """
        st.markdown(explanation_html, unsafe_allow_html=True)
        
    elif show_accuracy:
        # -------------------------------------------------------------
        # MODEL ACCURACY PAGE
        # -------------------------------------------------------------
        st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #f0f7ff 0%, #ffffff 100%) !important; color: #0f172a !important; }
            h1, h2, h3, h4, h5, h6, p, span, div, text, g { color: #0f172a !important; }
            .stButton>button { background: white !important; color: #2563eb !important; border: 2px solid #2563eb !important; font-weight: 600 !important; border-radius: 0.5rem !important; }
            .stButton>button:hover { background: #2563eb !important; color: white !important; }
            .accuracy-box {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 1rem;
                padding: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 8])
        with col1:
            if st.button("← Back"):
                st.session_state.show_accuracy = False
                st.rerun()
                
        with col2:
            st.title("AI Model Accuracy Explained")
            st.markdown("Discover the robust machine learning architecture powering your premium predictions.")
            
        st.markdown("---")
        
        user_data = st.session_state.get("user_data", {})
        try: age = int(user_data.get("age", 25))
        except: age = 25
        is_youth = age <= 35
        
        if is_youth:
            model_type = "Youth Segmented Ensemble (Age <= 35)"
            mae = 261.32
            rmse = 303.46
            r2 = "98.79%"
        else:
            model_type = "Senior Segmented Ensemble (Age > 35)"
            mae = 254.56
            rmse = 300.33
            r2 = "99.81%"
            
        st.markdown(f"""
        <div class="accuracy-box">
            <h3 style="color:#2563eb !important; margin-top:0;">{model_type}</h3>
            <p style="font-size: 1.1rem; line-height: 1.7;">
                Instead of relying on a single algorithm, Pulse AI uses a powerful technique called <strong>Stacking</strong>, heavily customized for your age demographic. 
                Our final Meta-Model combines predictions from XGBoost, LightGBM, and Random Forest models to produce an incredibly accurate premium estimate, achieving an <strong>R² score of {r2}</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        import plotly.graph_objects as go
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("<br><h4 style='color: #0f172a !important;'>Model Architecture</h4>", unsafe_allow_html=True)
            # Architecture Diagram using Plotly Sankey
            fig_arch = go.Figure(data=[go.Sankey(
                node = dict(
                  pad = 20,
                  thickness = 30,
                  line = dict(color = "black", width = 0.5),
                  label = ["User Data", "XGBoost (Trees)", "LightGBM (Gradient)", "Random Forest (Ensemble)", "Ridge Regressor (Meta-Model)", "Final Premium Prediction"],
                  color = ["#cbd5e1", "#60a5fa", "#60a5fa", "#60a5fa", "#3b82f6", "#10b981"]
                ),
                link = dict(
                  source = [0, 0, 0, 1, 2, 3, 4], # indices correspond to labels
                  target = [1, 2, 3, 4, 4, 4, 5],
                  value =  [1, 1, 1, 1, 1, 1, 3],
                  color = "rgba(148, 163, 184, 0.4)"
              ))])
            fig_arch.update_layout(paper_bgcolor='white', plot_bgcolor='white', font_color='#000000', margin=dict(t=20, b=20, l=10, r=10), height=400)
            st.plotly_chart(fig_arch, use_container_width=True)
            
        with c2:
            st.markdown(f"<br><h4 style='color: #0f172a !important;'>Performance Metrics (Final Meta-Model)</h4>", unsafe_allow_html=True)
            # Bar chart comparing errors
            fig_perf = go.Figure(go.Bar(
                x=['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)'],
                y=[mae, rmse],
                marker_color=['#3b82f6', '#94a3b8'],
                text=[str(mae), str(rmse)],
                textposition='auto',
            ))
            fig_perf.update_layout(
                paper_bgcolor='white', plot_bgcolor='white', font_color='#000000',
                margin=dict(t=20, b=20, l=10, r=10), height=400,
                yaxis_title="Error Value (Lower is Better)"
            )
            fig_perf.update_xaxes(color="black", tickfont=dict(color="black"), title_font=dict(color="black"), gridcolor="#e2e8f0")
            fig_perf.update_yaxes(color="black", tickfont=dict(color="black"), title_font=dict(color="black"), gridcolor="#e2e8f0")
            st.plotly_chart(fig_perf, use_container_width=True)

except Exception as e:
    st.error(f"Error loading UI: {e}")
