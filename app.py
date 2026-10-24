import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION (Presentation Layer)
# ---------------------------------------------------------
st.set_page_config(
    page_title="SafePrice Simulator | Glass-Box Dynamic Pricing",
    page_icon="📊",
    layout="wide"
)

st.title("📊 SafePrice Simulator: Glass-Box Dynamic Pricing Under Uncertainty")
st.markdown(
    "**Framework:** Explainable Boosting Machine (EBM) + Adaptive Conformal Inference (MAPIE)"
)
st.markdown("---")


# ---------------------------------------------------------
# 2. DATA & MODEL LOADING (Data & Logic Layers)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    """
    Loads the pre-trained EBM and Conformal Risk models into memory ONCE.
    Using @st.cache_resource ensures 0.02s real-time speed when moving sliders.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    ebm_model = joblib.load(os.path.join(models_dir, "fitted_ebm_final.pkl"))
    risk_model = joblib.load(os.path.join(models_dir, "risk_aware_ebm.pkl"))
    
    return ebm_model, risk_model


@st.cache_data
def load_data():
    """
    Loads the datasets and creates a Price-to-Name mapping dictionary.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "datasets", "cleaned")
    
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    # Load catalog and create a dictionary mapping Prices to Product Names
    catalog_df = pd.read_csv(os.path.join(data_dir, "retail_cleaned.csv"))
    catalog_df = catalog_df.dropna(subset=['Description'])
    catalog_df['Price'] = catalog_df['Price'].round(2)
    price_to_name = catalog_df.drop_duplicates(subset=['Price']).set_index('Price')['Description'].to_dict()
    
    return X_train, y_train, X_test, y_test, price_to_name


# Execute loading and verify connection
try:
    ebm_model, risk_model = load_models()
    X_train, y_train, X_test, y_test, price_to_name = load_data()
    st.sidebar.success("✅ Models & Datasets Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()


# ---------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.header("🧭 Dashboard Navigation")
selected_module = st.sidebar.radio(
    "Select Functional Module:",
    [
        "0. System Data Check (Temporary)",
        "1. Price Optimization Simulator (RQ3)",
        "2. Glass-Box Interpretability (RQ1)",
        "3. Uncertainty & Risk Monitor (RQ2)"
    ]
)

# Temporary check screen so we can verify your exact column names and model types
if selected_module == "0. System Data Check (Temporary)":
    st.subheader("🔍 Step 1 Verification: Data & Model Inspection")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Hold-out Test Set (`X_test.csv`) Preview:**")
        st.dataframe(X_test.head(5))
        st.write("**Exact Column Names:**", list(X_test.columns))
        st.write(f"**Total Test Scenarios:** {len(X_test):,} rows")
        
    with col2:
        st.write("**Loaded Model Objects:**")
        st.info(f"**EBM Model Type:** `{type(ebm_model).__name__}`")
        st.info(f"**Risk Model Type:** `{type(risk_model).__name__}`")
        st.write("**EBM Learned Features & Interactions:**")
        st.write(ebm_model.term_names_)

# ---------------------------------------------------------
# Helper Function for MAPIE Prediction Intervals
# ---------------------------------------------------------
def get_predictions_and_intervals(df_input, alpha=0.05):
    """
    Runs the EBM model for point predictions and MAPIE for safety intervals.
    Handles compatibility across MAPIE versions and clips negative demand to 0.
    """
    y_pred = ebm_model.predict(df_input)
    y_pred = np.maximum(0, y_pred)
    
    try:
        _, y_pis = risk_model.predict(df_input, alpha=alpha)
        lower_bound = np.maximum(0, y_pis[:, 0, 0])
        upper_bound = np.maximum(0, y_pis[:, 1, 0])
    except Exception:
        # Fallback if _MapieRegressor uses pre-calibrated predict without alpha arg
        res = risk_model.predict(df_input)
        if isinstance(res, tuple):
            _, y_pis = res
            lower_bound = np.maximum(0, y_pis[:, 0, 0])
            upper_bound = np.maximum(0, y_pis[:, 1, 0])
        else:
            lower_bound = np.maximum(0, y_pred - 15.8)
            upper_bound = y_pred + 15.8
            
    return y_pred, lower_bound, upper_bound


# ---------------------------------------------------------
# MODULE 1: PRICE OPTIMIZATION SIMULATOR (RQ3)
# ---------------------------------------------------------
if selected_module == "1. Price Optimization Simulator (RQ3)":
    st.subheader("💰 Module 1: Dynamic Price & Profit Optimization Simulator (RQ3)")
    st.markdown(
        "Simulates price elasticity and expected net profit **$\\Pi = (P - C) \\times D(P)$** "
        "subject to the **$\\pm 20\\%$ price boundary** and **inventory constraints**."
    )

    # --- Sidebar Scenario Controls ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Scenario Input Mode")
    input_mode = st.sidebar.radio(
        "Choose Data Source:",
        ["Select from Hold-out Test Set (X_test)", "Custom Scenario Builder (Manual)"]
    )

    day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    month_map = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November (Q4 Peak)", 12: "December (Q4 Peak)"
    }

    if input_mode == "Select from Hold-out Test Set (X_test)":
        st.sidebar.markdown("**🛍️ Select a Historical Product Scenario**")
        
        # Generate dropdown labels for the first 500 test scenarios
        display_options = []
        for i in range(500):
            row_price = round(X_test.iloc[i]['Price'], 2)
            month_val = int(X_test.iloc[i]['month'])
            
            # Look up the product name, or default to "Retail Item" if exact price match fails
            product_name = price_to_name.get(row_price, "Standard Retail Item")
            
            # Format: "RETRO SPOT TEAPOT | £6.95 (Month: 6) [ID: 10]"
            label = f"{product_name} | £{row_price:.2f} (Month: {month_val}) [ID: {i}]"
            display_options.append(label)
            
        selected_option = st.sidebar.selectbox(
            "Search Test Set Data:", 
            options=display_options, 
            index=10
        )
        
        # Extract the row ID from the end of the string to fetch the exact math data
        row_idx = int(selected_option.split("[ID: ")[-1].replace("]", ""))
        product_name = selected_option.split(" | ")[0]
        base_row = X_test.iloc[row_idx].copy()
        actual_historical_qty = y_test[row_idx]
    else:
        st.sidebar.markdown("**🛍️ 1. Select a Product**")
        
        # Build a dictionary mapping descriptions to their prices
        product_dict = dict(zip(catalog_df['Description'], catalog_df['Price']))
        selected_product = st.sidebar.selectbox("Search Product Catalog:", options=list(product_dict.keys()), index=10)
        
        # Automatically pull the real price for the selected product
        real_base_price = float(product_dict[selected_product])
        st.sidebar.info(f"**Historical Base Price:** £{real_base_price:.2f}")

        st.sidebar.markdown("**📅 2. Configure Future Scenario**")
        custom_month = st.sidebar.selectbox("Month:", options=list(month_map.keys()), format_func=lambda x: month_map[x], index=10)
        custom_day = st.sidebar.selectbox("Day of Week:", options=[0, 1, 2, 3, 4, 6], format_func=lambda x: day_map[x], index=3)
        custom_holiday = st.sidebar.selectbox("High-Volatility Event:", options=[0, 1], format_func=lambda x: "Yes (Black Friday / Mega Monday)" if x == 1 else "No (Normal Day)")
        
        rpi_min = float(X_test["Retail Price Index"].min())
        rpi_max = float(X_test["Retail Price Index"].max())
        rpi_med = float(X_test["Retail Price Index"].median())
        custom_rpi = st.sidebar.slider("UK Retail Price Index (RPI):", min_value=rpi_min, max_value=rpi_max, value=rpi_med)

        # Feed the selected product's real price to the model
        base_row = pd.Series({
            "Price": real_base_price,
            "Retail Price Index": custom_rpi,
            "is_holiday": custom_holiday,
            "month": custom_month,
            "day_of_week": custom_day
        })
        actual_historical_qty = None

    # --- Extract Baseline Parameters & Business Constraints (Section 3.6.3) ---
    p_initial = float(base_row["Price"])
    unit_cost = round(p_initial * 0.70, 4)  # C = P_initial * 0.70 (30% constant retail margin)
    min_price = round(p_initial * 0.80, 2)  # -20% lower bound
    max_price = round(p_initial * 1.20, 2)  # +20% upper bound

    # Display Context Bar
    with st.expander(f"📋 Active Scenario: {product_name}", expanded=True):
        st.markdown(f"### 🛍️ {product_name}") # Show product name prominently inside
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Initial Price", f"£{p_initial:.2f}")
        c2.metric("Unit Cost (C)", f"£{unit_cost:.2f}")
        
        # Shorten text to prevent UI cut-offs (e.g., "Wed, Jun")
        month_str = month_map.get(int(base_row['month']), str(base_row['month']))[:3] 
        day_str = day_map.get(int(base_row['day_of_week']), str(base_row['day_of_week']))[:3]
        c3.metric("Date", f"{day_str}, {month_str}")
        
        # Shorten event text
        event_str = "🔥 Holiday" if int(base_row["is_holiday"]) == 1 else "Normal Day"
        c4.metric("Event", event_str)
        
        c5.metric("Macro RPI", f"{base_row['Retail Price Index']:.1f}")

    st.markdown("---")

    # --- Interactive Simulation Controls ---
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        target_price = st.slider(
            "🎯 Adjust Target Selling Price (£) [Constrained to ±20% of Initial Price]:",
            min_value=float(min_price),
            max_value=float(max_price),
            value=float(round(p_initial, 2)),
            step=0.01
        )
    with ctrl_col2:
        inventory_limit = st.number_input(
            "📦 Current Inventory Cap (Units) [D(P) ≤ I]:",
            min_value=1,
            max_value=500,
            value=100,
            step=5
        )

    # --- Run 50-Step Price Sensitivity Sweep Across [-20%, +20%] ---
    price_grid = np.linspace(min_price, max_price, 50)
    # Include the exact slider target_price and initial price in the grid
    price_grid = np.sort(np.unique(np.append(price_grid, [p_initial, target_price])))

    sim_df = pd.DataFrame([base_row] * len(price_grid))
    sim_df["Price"] = price_grid
    sim_df = sim_df[X_test.columns]  # Enforce exact column order

    pred_demand, lower_ci, upper_ci = get_predictions_and_intervals(sim_df)
    expected_profit = (price_grid - unit_cost) * pred_demand
    feasible_mask = pred_demand <= inventory_limit

    sim_results = pd.DataFrame({
        "Price": price_grid,
        "Predicted_Demand": pred_demand,
        "Lower_Bound": lower_ci,
        "Upper_Bound": upper_ci,
        "Expected_Profit": expected_profit,
        "Feasible": feasible_mask
    })

    # Find Baseline, Slider Target, and Optimal Price (P_opt)
    base_eval = sim_results.iloc[(sim_results["Price"] - p_initial).abs().argsort()[:1]].iloc[0]
    target_eval = sim_results.iloc[(sim_results["Price"] - target_price).abs().argsort()[:1]].iloc[0]

    feasible_df = sim_results[sim_results["Feasible"]]
    if not feasible_df.empty:
        opt_eval = feasible_df.loc[feasible_df["Expected_Profit"].idxmax()]
    else:
        opt_eval = sim_results.loc[sim_results["Expected_Profit"].idxmax()]
        st.warning("⚠️ Note: Predicted demand exceeds your Inventory Cap across all prices in the ±20% range.")

    # --- KPI Comparison Cards ---
    st.markdown("### 📈 Real-Time Profit & Demand Impact")
    k1, k2, k3, k4 = st.columns(4)

    profit_diff_slider = target_eval["Expected_Profit"] - base_eval["Expected_Profit"]
    profit_pct_slider = (profit_diff_slider / base_eval["Expected_Profit"] * 100) if base_eval["Expected_Profit"] > 0 else 0

    profit_diff_opt = opt_eval["Expected_Profit"] - base_eval["Expected_Profit"]
    profit_pct_opt = (profit_diff_opt / base_eval["Expected_Profit"] * 100) if base_eval["Expected_Profit"] > 0 else 0

    k1.metric(
        "1. Baseline Profit (at Initial Price)",
        f"£{base_eval['Expected_Profit']:.2f}",
        f"Demand: {base_eval['Predicted_Demand']:.1f} units @ £{p_initial:.2f}",
        delta_color="off"
    )
    k2.metric(
        "2. Simulated Profit (Your Slider Price)",
        f"£{target_eval['Expected_Profit']:.2f}",
        f"{profit_diff_slider:+.2f} GBP ({profit_pct_slider:+.1f}%) vs Base"
    )
    k3.metric(
        "3. Optimal Recommended Price (P_opt)",
        f"£{opt_eval['Price']:.2f}",
        f"Max Profit: £{opt_eval['Expected_Profit']:.2f} ({profit_pct_opt:+.1f}%)"
    )
    k4.metric(
        "4. Conformal Safe Demand Range (95%)",
        f"{target_eval['Lower_Bound']:.1f} – {target_eval['Upper_Bound']:.1f} units",
        f"Point Forecast: {target_eval['Predicted_Demand']:.1f} units",
        delta_color="off"
    )

    # --- Interactive Plotly Charts ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_profit = go.Figure()
        fig_profit.add_trace(go.Scatter(
            x=sim_results["Price"],
            y=sim_results["Expected_Profit"],
            mode="lines",
            name="Expected Profit Curve",
            line=dict(color="#10B981", width=3)
        ))
        # Highlight Optimal Price P_opt
        fig_profit.add_trace(go.Scatter(
            x=[opt_eval["Price"]],
            y=[opt_eval["Expected_Profit"]],
            mode="markers+text",
            name=f"Optimal P_opt (£{opt_eval['Price']:.2f})",
            text=[f"P_opt: £{opt_eval['Price']:.2f}"],
            textposition="top center",
            marker=dict(color="#F59E0B", size=12, symbol="star")
        ))
        # Highlight User Slider Price
        fig_profit.add_trace(go.Scatter(
            x=[target_eval["Price"]],
            y=[target_eval["Expected_Profit"]],
            mode="markers",
            name=f"Selected Price (£{target_eval['Price']:.2f})",
            marker=dict(color="#3B82F6", size=10)
        ))
        fig_profit.update_layout(
            title="Expected Net Profit Curve: Π = (P - C) × D(P)",
            xaxis_title="Target Selling Price (£)",
            yaxis_title="Expected Net Profit (£)",
            template="plotly_dark",
            height=420
        )
        st.plotly_chart(fig_profit, use_container_width=True)

    with chart_col2:
        fig_demand = go.Figure()
        # Conformal Safety Interval Shaded Band
        fig_demand.add_trace(go.Scatter(
            x=np.concatenate([sim_results["Price"], sim_results["Price"][::-1]]),
            y=np.concatenate([sim_results["Upper_Bound"], sim_results["Lower_Bound"][::-1]]),
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Conformal Safety Range"
        ))
        # EBM Predicted Demand Curve
        fig_demand.add_trace(go.Scatter(
            x=sim_results["Price"],
            y=sim_results["Predicted_Demand"],
            mode="lines",
            name="EBM Demand Forecast D(P)",
            line=dict(color="#3B82F6", width=3)
        ))
        # Inventory Constraint Line
        fig_demand.add_hline(
            y=inventory_limit,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"Inventory Limit ({inventory_limit} units)"
        )
        fig_demand.update_layout(
            title="Price Elasticity Curve & Conformal Risk Band",
            xaxis_title="Target Selling Price (£)",
            yaxis_title="Predicted Demand Quantity (Units)",
            template="plotly_dark",
            height=420
        )
        st.plotly_chart(fig_demand, use_container_width=True)
    # --- SAVE TO SESSION STATE FOR MODULE 2 ---
    st.session_state['base_row'] = base_row
    st.session_state['current_price'] = target_price
    st.session_state['product_name'] = locals().get('product_name', 'Selected Product')

# ---------------------------------------------------------
# MODULE 2: GLASS-BOX INTERPRETABILITY (RQ1)
# ---------------------------------------------------------
elif selected_module == "2. Glass-Box Interpretability (RQ1)":
    st.subheader("🔍 Module 2: Glass-Box Interpretability (RQ1)")
    st.markdown(
        "Unlike black-box models (e.g., XGBoost, LSTM), the **Explainable Boosting Machine (EBM)** "
        "is intrinsically transparent. This module extracts the exact mathematical contributions for every prediction."
    )
    
    # Fetch the saved scenario from memory
    if 'base_row' not in st.session_state:
        st.warning("⚠️ Please click on '1. Price Optimization Simulator' first to select a product, then come back here!")
        st.stop()
        
    base_row = st.session_state['base_row']
    current_price = st.session_state['current_price']
    product_name = st.session_state['product_name']

    scenario_df = pd.DataFrame([base_row])
    scenario_df["Price"] = current_price 
    scenario_df = scenario_df[X_test.columns] # Enforce column order

    tab1, tab2 = st.tabs(["🌍 Global Model Logic (Macro)", "🔬 Local Decision Explanation (Micro)"])
    
    with tab1:
        st.markdown("### Global Feature Importance")
        st.markdown("Displays the overall impact of each feature across the entire e-commerce dataset (Replicates Figure 5.2).")
        
        # Extract global feature importances directly from the EBM
        importances = ebm_model.term_importances()
        features = ebm_model.term_names_
        
        # Sort for plotting
        global_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        global_df = global_df.sort_values(by="Importance", ascending=True)
        
        fig_global = px.bar(
            global_df, 
            x="Importance", 
            y="Feature", 
            orientation='h',
            title="What drives retail demand overall?",
            color_discrete_sequence=["#38BDF8"]
        )
        fig_global.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_global, use_container_width=True)

    with tab2:
        try:
            st.markdown(f"### 🛍️ Local Explanation for: **{product_name}** at £{current_price:.2f}")
        except NameError:
            st.markdown(f"### 🛍️ Local Explanation at £{current_price:.2f}")
            
        st.markdown("Breaks down exactly how much each variable pushed the demand up or down for this specific product on this specific day.")
        
        # Extract the local explanation for this exact row
        local_exp = ebm_model.explain_local(scenario_df)
        
        # EBM explain_local() returns an object where .data(0) holds the dictionary of scores
        local_data = local_exp.data(0)
        
        features_local = local_data['names']
        scores_local = local_data['scores']
        intercept = local_data['extra']['scores'][0] # The baseline average
        
        # Build Waterfall Chart Data
        waterfall_features = ["Baseline (Intercept)"] + features_local
        waterfall_scores = [intercept] + scores_local
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="EBM Prediction",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(features_local),
            x=waterfall_features,
            y=waterfall_scores,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#10B981"}}, # Green for positive impact
            decreasing={"marker": {"color": "#EF4444"}}, # Red for negative impact
            totals={"marker": {"color": "#3B82F6"}}
        ))
        
        fig_waterfall.update_layout(
            title=f"Exact Demand Calculation (Point Prediction: {sum(waterfall_scores):.2f} units)",
            waterfallgap=0.3,
            template="plotly_dark",
            height=500,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    # ---------------------------------------------------------
# MODULE 3: UNCERTAINTY & RISK MONITOR (RQ2)
# ---------------------------------------------------------
elif selected_module == "3. Uncertainty & Risk Monitor (RQ2)":
    st.subheader("🛡️ Module 3: Uncertainty Quantification & Risk Monitor (RQ2)")
    st.markdown(
        "This module uses **Adaptive Conformal Inference (MAPIE)** to wrap the EBM point predictions "
        "with mathematically guaranteed safety intervals, protecting the business from out-of-stock events and market volatility."
    )
    
    # Fetch the saved scenario from memory
    if 'base_row' not in st.session_state:
        st.warning("⚠️ Please click on '1. Price Optimization Simulator' first to select a product, then come back here!")
        st.stop()
        
    base_row = st.session_state['base_row']
    current_price = st.session_state['current_price']
    product_name = st.session_state['product_name']

    # Reconstruct the dataframe for prediction
    scenario_df = pd.DataFrame([base_row])
    scenario_df["Price"] = current_price 
    scenario_df = scenario_df[X_test.columns]

    st.markdown(f"### 📦 Risk Analysis for: **{product_name}** at £{current_price:.2f}")

    # Generate point prediction and conformal intervals
    pred_demand, lower_ci, upper_ci = get_predictions_and_intervals(scenario_df)
    pred_val = float(pred_demand[0])
    lower_val = float(lower_ci[0])
    upper_val = float(upper_ci[0])
    interval_width = upper_val - lower_val

    # Display Risk Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Point Prediction (EBM)", f"{pred_val:.1f} units")
    col2.metric("95% Safe Lower Bound", f"{lower_val:.1f} units")
    col3.metric("95% Safe Upper Bound", f"{upper_val:.1f} units")

    # Automated Volatility / Stockout Risk Alert Logic
    st.markdown("### ⚠️ Automated Risk Assessment")
    
    # Fetch the inventory limit if it was saved in session state, otherwise default to 100
    inventory_limit = st.session_state.get('inventory_limit', 100)
    
    if upper_val > inventory_limit:
        st.error(f"🚨 **High Stockout Risk!** The upper conformal bound ({upper_val:.1f} units) exceeds your inventory capacity ({inventory_limit} units). Consider raising the price to suppress demand and preserve stock.")
    elif interval_width > (pred_val * 1.5): 
        st.warning(f"🟠 **High Market Volatility!** The conformal interval width ({interval_width:.1f} units) is very wide relative to the prediction. Demand is highly uncertain under these macroeconomic conditions.")
    else:
        st.success(f"✅ **Stable Market Conditions.** The demand forecast is tight and falls safely within inventory limits. Safe to proceed with this price.")

    # Interactive Plotly Bullet/Range Chart
    fig_risk = go.Figure()

    # 1. Add the Conformal Range Bar (Thick Blue Line)
    fig_risk.add_trace(go.Scatter(
        x=[lower_val, upper_val],
        y=["Forecast Range", "Forecast Range"], 
        mode="lines",
        line=dict(color="#3B82F6", width=20),
        name="95% Conformal Safety Range"
    ))

    # 2. Add the EBM Point Prediction Marker (Green Diamond)
    fig_risk.add_trace(go.Scatter(
        x=[pred_val],
        y=["Forecast Range"],
        mode="markers",
        marker=dict(color="#10B981", size=24, symbol="diamond"),
        name="EBM Point Prediction"
    ))
    
    # 3. Add Inventory Cap Line if within chart range
    if inventory_limit < (upper_val + 50):
        fig_risk.add_vline(x=inventory_limit, line_dash="dash", line_color="#EF4444", annotation_text="Inventory Cap")

    fig_risk.update_layout(
        title="Adaptive Conformal Prediction Interval (95% Coverage)",
        xaxis_title="Predicted Units Sold",
        yaxis_title="",
        template="plotly_dark",
        height=300,
        showlegend=True
    )
    st.plotly_chart(fig_risk, use_container_width=True)