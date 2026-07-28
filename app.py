import streamlit as st
import pandas as pd
import pydeck as pdk
import joblib

st.set_page_config(
    page_title="AI National Supply Chain Risk Intelligence Platform", 
    layout="wide"
)

# 1. CACHE THE DATA
@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_SupplyChain_Dataset.csv")

# 2. CACHE THE MODEL
@st.cache_resource
def load_model():
    return joblib.load("supply_chain_risk_model.pkl")

# 3. CACHE THE CSV CONVERSION
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

# Load resources once
df = load_data()
model = load_model()

st.sidebar.title("🚚 AI Supply Chain Platform")

page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Dashboard", "🤖 Prediction", "📈 Analytics"]
)

# Move filters to apply globally but only process if needed
st.sidebar.header("🔍 Filters")
selected_state = st.sidebar.selectbox(
    "Select State",
    ["All"] + sorted(df["Order State"].dropna().unique().tolist())
)

if selected_state != "All":
    filtered_df = df[df["Order State"] == selected_state]
else:
    filtered_df = df


if page == "🏠 Home":
    st.title("🚚 AI National Supply Chain Risk Intelligence Platform")
    st.markdown("""
    ### 🎯 Project Objective
    This AI platform predicts supply chain risks, analyzes shipment performance,
    and helps businesses reduce delivery delays using Machine Learning.

    ### 👨‍💻 Features
    ✅ AI Risk Prediction
    ✅ Interactive Dashboard
    ✅ Supply Chain Analytics
    ✅ Business Insights
    """)


elif page == "📊 Dashboard":
    # Moved metrics inside the Dashboard page so they don't load on Home/Analytics
    st.write("### Supply Chain Risk Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    high_risk = len(df[df["Late_delivery_risk"] == 1])

    st.metric("🚨 High Risk Shipments", high_risk)

    col1.metric("📦 Total Orders", len(df))
    col2.metric("💰 Total Sales", f"{df['Sales per customer'].sum():,.0f}")
    col3.metric("📈 Average Benefit", f"{df['Benefit per order'].mean():.2f}")
    col4.metric("🚚 Average Shipping Days", f"{df['Days for shipment (scheduled)'].mean():.1f}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
    
    # Generate CSV strictly when on this page, using the cached function
    csv = convert_df(df)
    st.download_button(
        "📥 Download CSV Report",
        csv,
        "Supply_Chain_Report.csv",
        "text/csv"
    )


elif page == "🤖 Prediction":
    st.subheader("🔮 Supply Chain Risk Prediction")

    benefit = st.number_input("Benefit per Order")
    sales = st.number_input("Sales per Customer")
    quantity = st.number_input("Order Item Quantity")
    price = st.number_input("Order Item Product Price")
    days = st.number_input("Days for Shipment (Scheduled)")

    if st.button("Predict Risk"):
        prediction = model.predict([[benefit, sales, quantity, price, days]])
        
        if prediction[0] == 1:
            st.error("🚨 High Risk Shipment")

            st.subheader("🤖 AI Recommendation")
            st.write("🚚 Switch to faster shipping mode")
            st.write("🏭 Evaluate supplier performance")
            st.write("📦 Increase safety stock level")
            st.write("📊 Monitor high-risk regions")
            
            st.subheader("🧠 Why is this shipment High Risk?")
            if days > 5:
                st.write("📅 Scheduled shipping time is high.")
            if quantity > 10:
                st.write("📦 Large order quantity increases delivery risk.")
            if sales > 500:
                st.write("💰 High sales volume requires faster logistics.")
            if benefit < 100:
                st.write("📉 Low benefit per order may reduce supply chain efficiency.")
        
        else:
            st.success("✅ Low Risk Shipment")

            st.subheader("🤖 AI Recommendation")
            st.write("✅ Supply chain operations are stable")
            st.write("📈 Continue current strategy")
            st.write("🔍 Maintain supplier monitoring")
            
            st.subheader("🧠 Why is this shipment Low Risk?")
            st.write("✅ Delivery schedule is within normal limits.")
            st.write("✅ Supply chain conditions appear stable.")
            st.write("✅ No major operational risk detected.")


elif page == "📈 Analytics":
    st.subheader("📊 Sales Distribution")
    st.bar_chart(filtered_df["Sales per customer"].head(20))

    st.subheader("📈 Benefit Distribution")
    st.line_chart(filtered_df["Benefit per order"].head(20))

    st.subheader("📊 Risk Distribution")
    risk_count = df["Late_delivery_risk"].value_counts()
    
    fig = risk_count.plot.pie(autopct="%1.1f%%", figsize=(5,5)).get_figure()
    st.pyplot(fig)
    
    st.subheader("🗺️ State-wise Risk Analysis")
    state_risk = filtered_df.groupby("Order State")["Benefit per order"].mean()
    st.bar_chart(state_risk)
    
    st.subheader("🌍 Interactive Risk Map")
    map_data = df[["Latitude", "Longitude"]].dropna().head(500)

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=map_data["Latitude"].mean(),
                longitude=map_data["Longitude"].mean(),
                zoom=2,
                pitch=40,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position='[Longitude, Latitude]',
                    get_radius=30000,
                    get_fill_color='[255, 0, 0, 160]',
                    pickable=True,
                ),
            ],
        )
    )
    
    st.subheader("🌍 Live Risk Intelligence")
    alerts = [
        "⚠ Heavy rainfall may affect deliveries in South India.",
        "🚢 Port congestion reported in Singapore.",
        "⛽ Fuel price increase may impact transportation cost.",
        "🚛 Highway traffic delays expected in North Region."
    ]

    for alert in alerts:
        st.warning(alert)