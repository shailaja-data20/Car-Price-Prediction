import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import time


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "car_data" not in st.session_state:
    st.session_state.car_data = None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("car_price_prediction_model.pkl")


try:
    model = load_model()

except Exception as e:

    st.error("❌ Unable to load the ML model.")

    st.write("Please make sure this file is in the same folder:")

    st.code("car_price_prediction_model.pkl")

    st.error(str(e))

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🚗 Car Price Prediction")

st.subheader("✨ Know the real value before you drive!")

st.write(
    "Enter your car details below and let Machine Learning "
    "estimate the selling price."
)

st.divider()


# ============================================================
# FEATURE HIGHLIGHTS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.info(
        "🤖\n\n"
        "**ML Powered**\n\n"
        "Machine Learning based prediction"
    )

with col2:

    st.info(
        "⚡\n\n"
        "**Fast & Easy**\n\n"
        "Get prediction within seconds"
    )

with col3:

    st.info(
        "📊\n\n"
        "**Data Driven**\n\n"
        "Prediction from car features"
    )

with col4:

    st.info(
        "💡\n\n"
        "**Smart Choice**\n\n"
        "Make better buying decisions"
    )


st.write("")


# ============================================================
# MAIN LAYOUT
# ============================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT SIDE - INPUT FORM
# ============================================================

with left_column:

    st.header("📝 Enter Car Details")

    st.caption(
        "Provide the information about the car."
    )

    st.write("")


    with st.form("car_prediction_form"):

        # ----------------------------------------------------
        # ROW 1
        # ----------------------------------------------------

        c1, c2 = st.columns(2)

        with c1:

            brand = st.text_input(
                "🚘 Brand",
                value="Toyota",
                placeholder="Example: Toyota"
            )

        with c2:

            year = st.number_input(
                "📅 Manufacturing Year",
                min_value=1990,
                max_value=2026,
                value=2022,
                step=1
            )


        # ----------------------------------------------------
        # ROW 2
        # ----------------------------------------------------

        c1, c2 = st.columns(2)

        with c1:

            km_driven = st.number_input(
                "🛣️ Kilometers Driven",
                min_value=0,
                max_value=1000000,
                value=30000,
                step=1000
            )

        with c2:

            fuel_type = st.selectbox(
                "⛽ Fuel Type",
                [
                    "Petrol",
                    "Diesel",
                    "CNG",
                    "Electric"
                ]
            )


        # ----------------------------------------------------
        # ROW 3
        # ----------------------------------------------------

        c1, c2 = st.columns(2)

        with c1:

            transmission = st.selectbox(
                "⚙️ Transmission",
                [
                    "Manual",
                    "Automatic"
                ]
            )

        with c2:

            owner = st.selectbox(
                "👤 Owner",
                [
                    "First",
                    "Second",
                    "Third"
                ]
            )


        # ----------------------------------------------------
        # ROW 4
        # ----------------------------------------------------

        c1, c2 = st.columns(2)

        with c1:

            location = st.text_input(
                "📍 Location",
                value="Hyderabad",
                placeholder="Example: Hyderabad"
            )

        with c2:

            engine_cc = st.number_input(
                "🔧 Engine CC",
                min_value=500.0,
                max_value=6000.0,
                value=1500.0,
                step=50.0
            )


        # ----------------------------------------------------
        # ROW 5
        # ----------------------------------------------------

        c1, c2 = st.columns(2)

        with c1:

            mileage = st.number_input(
                "⛽ Mileage (km/l)",
                min_value=5.0,
                max_value=60.0,
                value=18.0,
                step=0.5
            )

        with c2:

            seats = st.number_input(
                "💺 Seats",
                min_value=2,
                max_value=10,
                value=5,
                step=1
            )


        # ----------------------------------------------------
        # ROW 6
        # ----------------------------------------------------

        insurance = st.selectbox(
            "🛡️ Insurance",
            [
                "Yes",
                "No"
            ]
        )


        st.write("")


        # ----------------------------------------------------
        # SUBMIT
        # ----------------------------------------------------

        predict_button = st.form_submit_button(
            "✨ Predict Car Price",
            use_container_width=True
        )


# ============================================================
# RIGHT SIDE - RESULT
# ============================================================

with right_column:

    st.header("🎯 Prediction Result")

    st.caption(
        "Your estimated selling price will appear here."
    )

    st.write("")


    # ========================================================
    # BEFORE PREDICTION
    # ========================================================

    if not predict_button and st.session_state.prediction is None:

        st.info(
            "🚗 **Ready to Predict?**\n\n"
            "Enter the car details on the left and click "
            "**✨ Predict Car Price**."
        )

        st.write("")

        st.markdown("### 🔮 How it works")

        st.write(
            "1️⃣ Enter car information\n\n"
            "2️⃣ Click Predict Car Price\n\n"
            "3️⃣ Machine Learning processes the data\n\n"
            "4️⃣ Get the estimated selling price 🎉"
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        try:

            # ------------------------------------------------
            # IMPORTANT:
            # Your training notebook uses 2026
            # ------------------------------------------------

            current_year = 2026


            # ------------------------------------------------
            # FEATURE ENGINEERING
            # ------------------------------------------------

            car_age = max(
                current_year - int(year),
                1
            )

            km_per_year = (
                km_driven / car_age
            )


            # ------------------------------------------------
            # CREATE INPUT DATAFRAME
            # ------------------------------------------------

            input_data = pd.DataFrame({

                "Brand": [brand],

                "Year": [year],

                "Km_Driven": [km_driven],

                "Fuel_Type": [fuel_type],

                "Transmission": [transmission],

                "Owner": [owner],

                "Location": [location],

                "Engine_CC": [engine_cc],

                "Mileage": [mileage],

                "Seats": [seats],

                "Insurance": [insurance],

                "Car_Age": [car_age],

                "Km_per_Year": [km_per_year]

            })


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            prediction = model.predict(
                input_data
            )

            predicted_price = float(
                prediction[0]
            )


            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            st.session_state.prediction = predicted_price

            st.session_state.car_data = input_data


            # ------------------------------------------------
            # CELEBRATION 🎉
            # ------------------------------------------------

            st.success(
                "🎉 Prediction completed successfully!"
            )

            st.balloons()

            time.sleep(0.5)


            # ------------------------------------------------
            # PRICE
            # ------------------------------------------------

            st.metric(
                label="💰 Estimated Selling Price",
                value=f"₹ {predicted_price:.2f} Lakh"
            )


            st.write("")


            # ------------------------------------------------
            # RESULT MESSAGE
            # ------------------------------------------------

            if predicted_price >= 10:

                st.success(
                    "🌟 Excellent! The predicted market value "
                    "of this car is strong."
                )

            elif predicted_price >= 5:

                st.success(
                    "👍 Good! The model predicts a healthy "
                    "selling value for this car."
                )

            else:

                st.info(
                    "💡 The predicted selling price is relatively "
                    "lower. Actual price can vary depending on "
                    "vehicle condition and market demand."
                )


            # ------------------------------------------------
            # CAR SUMMARY
            # ------------------------------------------------

            st.write("")

            st.subheader("🚘 Your Car Summary")


            summary1, summary2, summary3 = st.columns(3)


            with summary1:

                st.metric(
                    "📅 Car Age",
                    f"{car_age} Years"
                )


            with summary2:

                st.metric(
                    "🛣️ Km / Year",
                    f"{km_per_year:,.0f}"
                )


            with summary3:

                st.metric(
                    "⚙️ Engine",
                    f"{engine_cc:.0f} CC"
                )


            # ------------------------------------------------
            # DETAILS
            # ------------------------------------------------

            st.write("")

            st.subheader(
                "📋 Details Used for Prediction"
            )


            display_data = pd.DataFrame({

                "Feature": [

                    "Brand",
                    "Manufacturing Year",
                    "Kilometers Driven",
                    "Fuel Type",
                    "Transmission",
                    "Owner",
                    "Location",
                    "Engine CC",
                    "Mileage",
                    "Seats",
                    "Insurance",
                    "Car Age",
                    "Km per Year"

                ],

                "Value": [

                    brand,
                    year,
                    f"{km_driven:,}",
                    fuel_type,
                    transmission,
                    owner,
                    location,
                    f"{engine_cc:.0f} CC",
                    f"{mileage:.1f} km/l",
                    seats,
                    insurance,
                    f"{car_age} years",
                    f"{km_per_year:,.0f}"

                ]

            })


            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # DOWNLOAD RESULT
            # ------------------------------------------------

            result_data = input_data.copy()

            result_data[
                "Predicted_Selling_Price_Lakh"
            ] = predicted_price


            csv_data = result_data.to_csv(
                index=False
            )


            st.download_button(

                label="📥 Download Prediction",

                data=csv_data,

                file_name="car_price_prediction.csv",

                mime="text/csv",

                use_container_width=True

            )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.write(
                "Please check the error below:"
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚗 Car Price Prediction | "
    "Built with Python + Machine Learning + Streamlit | "
    "✨ Drive Smart • Decide Smarter"
)
