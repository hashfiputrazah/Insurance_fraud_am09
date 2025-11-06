import streamlit as st
import pandas as pd
import numpy as np
import pickle
import dill
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚗",
    layout="wide"
)

# Load model and explainer
@st.cache_resource
def load_model_and_explainer():
    with open('final_model_xgb_tuned_FIX_BGT_20251106_0942.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('line_explainer.dill', 'rb') as f:
        explainer = dill.load(f)
    return model, explainer

try:
    model, line_explainer = load_model_and_explainer()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Title
st.title("🚗 Insurance Fraud Detection System")
st.markdown("---")

# Create input form in the center
if model_loaded:
    with st.form("prediction_form"):
        st.subheader("📋 Enter Claim Information")
        
        # Group 1: Personal Information
        st.markdown("### 👤 Personal Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Sex = st.selectbox("Sex", ['Female', 'Male'])
        with col2:
            MaritalStatus = st.selectbox("Marital Status", ['Single', 'Married', 'Widow', 'Divorced'])
        with col3:
            AgeOfPolicyHolder = st.selectbox("Age of Policy Holder", 
                ['16 to 17', '18 to 20', '21 to 25', '26 to 30', '31 to 35', 
                 '36 to 40', '41 to 50', '51 to 65', 'over 65'])
        
        st.markdown("---")
        
        # Group 2: Accident Details
        st.markdown("### 🚨 Accident Details")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            Month = st.selectbox("Accident Month", 
                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        with col2:
            DayOfWeek = st.selectbox("Accident Day", 
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        with col3:
            AccidentArea = st.selectbox("Accident Area", ['Urban', 'Rural'])
        with col4:
            Fault = st.selectbox("Fault", ['Policy Holder', 'Third Party'])
        
        col1, col2 = st.columns(2)
        with col1:
            PoliceReportFiled = st.selectbox("Police Report Filed", ['No', 'Yes'])
        with col2:
            WitnessPresent = st.selectbox("Witness Present", ['No', 'Yes'])
        
        st.markdown("---")
        
        # Group 3: Vehicle Information
        st.markdown("### 🚙 Vehicle Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            Make = st.selectbox("Make", 
                ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac', 'Accura', 
                 'Dodge', 'Mercury', 'Jaguar', 'Nisson', 'VW', 'Saab', 'Saturn', 
                 'Porche', 'BMW', 'Mecedes', 'Ferrari', 'Lexus'])
        with col2:
            VehicleCategory = st.selectbox("Vehicle Category", ['Sport', 'Utility', 'Sedan'])
        with col3:
            VehiclePrice = st.selectbox("Vehicle Price", 
                ['less than 20000', '20000 to 29000', '30000 to 39000', 
                 '40000 to 59000', '60000 to 69000', 'more than 69000'])
        with col4:
            AgeOfVehicle = st.selectbox("Age of Vehicle", 
                ['new', '2 years', '3 years', '4 years', '5 years', 
                 '6 years', '7 years', 'more than 7'])
        
        st.markdown("---")
        
        # Group 4: Policy Information
        st.markdown("### 📄 Policy Information")
        col1, col2 = st.columns(2)
        
        with col1:
            PolicyType = st.selectbox("Policy Type", 
                ['Sport - Liability', 'Sport - Collision', 'Sedan - Liability', 
                 'Utility - All Perils', 'Sedan - All Perils', 'Sedan - Collision', 
                 'Utility - Collision', 'Utility - Liability', 'Sport - All Perils'])
        with col2:
            BasePolicy = st.selectbox("Base Policy", ['Liability', 'Collision', 'All Perils'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            Days_Policy_Accident = st.selectbox("Days Policy to Accident", 
                ['none', '1 to 7', '8 to 15', '15 to 30', 'more than 30'])
        with col2:
            PastNumberOfClaims = st.selectbox("Past Number of Claims", 
                ['none', '1', '2 to 4', 'more than 4'])
        with col3:
            AgentType = st.selectbox("Agent Type", ['External', 'Internal'])
        with col4:
            NumberOfCars = st.selectbox("Number of Cars", 
                ['1 vehicle', '2 vehicles', '3 to 4', '5 to 8', 'more than 8'])
        
        st.markdown("---")
        
        # Group 5: Claim Information
        st.markdown("### 📋 Claim Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            MonthClaimed = st.selectbox("Month Claimed", 
                ['0', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        with col2:
            DayOfWeekClaimed = st.selectbox("Day of Week Claimed", 
                ['0', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        with col3:
            Days_Policy_Claim = st.selectbox("Days Policy to Claim", 
                ['none', '8 to 15', '15 to 30', 'more than 30'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            NumberOfSuppliments = st.selectbox("Number of Supplements", 
                ['none', '1 to 2', '3 to 5', 'more than 5'])
        with col2:
            AddressChange_Claim = st.selectbox("Address Change before Claim", 
                ['no change', 'under 6 months', '1 year', '2 to 3 years', '4 to 8 years'])
        
        st.markdown("---")
        
        # Group 6: Numerical Features
        st.markdown("### 🔢 Numerical Features")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            WeekOfMonth = st.number_input("Week of Month (Accident)", min_value=1, max_value=5, value=3)
        with col2:
            WeekOfMonthClaimed = st.number_input("Week of Month (Claimed)", min_value=1, max_value=5, value=3)
        with col3:
            Age = st.number_input("Age", min_value=0, max_value=80, value=40)
        with col4:
            Year = st.number_input("Year", min_value=1994, max_value=1996, value=1995)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            PolicyNumber = st.number_input("Policy Number", min_value=1, max_value=15420, value=7710)
        with col2:
            RepNumber = st.number_input("Rep Number", min_value=1, max_value=16, value=8)
        with col3:
            Deductible = st.number_input("Deductible", min_value=300, max_value=700, value=400)
        with col4:
            DriverRating = st.number_input("Driver Rating", min_value=1, max_value=4, value=2)
        
        st.markdown("---")
        
        # Submit button
        submit_button = st.form_submit_button("🔍 Predict Fraud", use_container_width=True, type="primary")
    
    # Process prediction when button is clicked
    if submit_button:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Month': [Month],
            'WeekOfMonth': [WeekOfMonth],
            'DayOfWeek': [DayOfWeek],
            'Make': [Make],
            'AccidentArea': [AccidentArea],
            'DayOfWeekClaimed': [DayOfWeekClaimed],
            'MonthClaimed': [MonthClaimed],
            'WeekOfMonthClaimed': [WeekOfMonthClaimed],
            'Sex': [Sex],
            'MaritalStatus': [MaritalStatus],
            'Age': [Age],
            'Fault': [Fault],
            'PolicyType': [PolicyType],
            'VehicleCategory': [VehicleCategory],
            'VehiclePrice': [VehiclePrice],
            'Days_Policy_Accident': [Days_Policy_Accident],
            'Days_Policy_Claim': [Days_Policy_Claim],
            'PastNumberOfClaims': [PastNumberOfClaims],
            'AgeOfVehicle': [AgeOfVehicle],
            'AgeOfPolicyHolder': [AgeOfPolicyHolder],
            'PoliceReportFiled': [PoliceReportFiled],
            'WitnessPresent': [WitnessPresent],
            'AgentType': [AgentType],
            'NumberOfSuppliments': [NumberOfSuppliments],
            'AddressChange_Claim': [AddressChange_Claim],
            'NumberOfCars': [NumberOfCars],
            'Year': [Year],
            'BasePolicy': [BasePolicy],
            'PolicyNumber': [PolicyNumber],
            'RepNumber': [RepNumber],
            'Deductible': [Deductible],
            'DriverRating': [DriverRating]
        })
        
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display prediction
            st.markdown("---")
            st.subheader("🎯 Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if prediction == 1:
                    st.error("⚠️ **FRAUD DETECTED**")
                    st.metric("Fraud Probability", f"{prediction_proba[1]:.2%}")
                else:
                    st.success("✅ **LEGITIMATE CLAIM**")
                    st.metric("Legitimate Probability", f"{prediction_proba[0]:.2%}")
                
                # Show probability distribution
                st.markdown("#### Probability Distribution")
                prob_df = pd.DataFrame({
                    'Class': ['Legitimate', 'Fraud'],
                    'Probability': [prediction_proba[0], prediction_proba[1]]
                })
                st.bar_chart(prob_df.set_index('Class'))
            
            # LIME Explanation
            st.markdown("---")
            st.subheader("📊 LIME Explanation")
            
            with st.spinner("Generating explanation..."):
                # Transform input using preprocessing step
                preprocessor = model.named_steps['preprocessing']
                transformed_data = preprocessor.transform(input_data)
                
                # Convert to numpy array for LIME (if it's a DataFrame)
                if isinstance(transformed_data, pd.DataFrame):
                    transformed_array = transformed_data.values
                    feature_names = transformed_data.columns.tolist()
                else:
                    transformed_array = transformed_data
                    feature_names = None
                
                # Generate LIME explanation
                # Get the model's predict_proba method
                model_predict = model.named_steps['model'].predict_proba
                
                # Create explanation
                explanation = line_explainer.explain_instance(
                    transformed_array[0],
                    model_predict,
                    num_features=10
                )
                
                # Display explanation
                fig = explanation.as_pyplot_figure()
                st.pyplot(fig)
                plt.close()
                
                # Additional explanation details
                with st.expander("📋 View Detailed Feature Contributions"):
                    exp_list = explanation.as_list()
                    exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Contribution'])
                    exp_df['Contribution'] = exp_df['Contribution'].round(4)
                    st.dataframe(exp_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.exception(e)

else:
    st.warning("⚠️ Model not loaded. Please ensure the model files are in the correct location.")
    st.info("Required files:\n- final_model_xgb_tuned_FIX_BGT_20251106_0942.pkl\n- line_explainer.dill")