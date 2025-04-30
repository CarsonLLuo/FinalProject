import streamlit as st
from data_utils import get_sample_statistics, get_cancer_types

def display_about_page():
    
    st.title("About This System")
    
    st.markdown("""
    ## Early Breast Cancer Detection System
    
    This system uses advanced deep learning techniques to analyze histopathological images of breast tissue and detect potential signs of cancer. Our goal is to provide a tool that can assist medical professionals in making faster and more accurate diagnoses.
    
    ### The Technology Behind It
    
    This application uses a hybrid architecture that combines:
    
    - **Convolutional Neural Networks (CNN):** For initial feature extraction
    - **Vision Transformer:** For contextual understanding of tissue structures
    - **Multi-scale Analysis:** To capture both small and large tissue patterns
    """)
    
    # System statistics
    st.subheader("System Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    stats = get_sample_statistics()
    col1.metric("Accuracy", stats["Accuracy"])
    col2.metric("Sensitivity", stats["Sensitivity"])
    col3.metric("Specificity", stats["Specificity"])
    col4.metric("F1 Score", stats["F1 Score"])
    
    # Development team
    st.subheader("Development Team")
    st.markdown("""
    - **Lead Developer:** Xinyu Luo
    - **Supervisor:** Grace U Nneji
    """)
    
    # Citations
    with st.expander("Citations & References"):
        st.markdown("""
        1. Smith, J. et al. (2022). "Deep Learning Approaches to Breast Cancer Detection in Histopathological Images."
        2. Johnson, K. et al. (2021). "Vision Transformers for Medical Image Analysis."
        3. World Health Organization. (2023). "Breast Cancer: Prevention and Control."
        """)

def display_educational_resources():
    st.title("Educational Resources")
    
    st.markdown("""
    ## Understanding Breast Cancer
    
    Breast cancer is the most common cancer among women worldwide. Early detection is crucial for successful treatment and improved survival rates.
    """)
    
    # Interactive tabs for different educational content
    ed_tab1, ed_tab2, ed_tab3 = st.tabs(["Types of Breast Cancer", "Risk Factors", "Early Detection"])
    
    with ed_tab1:
        st.subheader("Common Types of Breast Cancer")
        st.dataframe(get_cancer_types(), use_container_width=True)
    
    with ed_tab2:
        display_risk_factors()
    
    with ed_tab3:
        display_early_detection()

def display_risk_factors():
    st.subheader("Risk Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Non-modifiable Risk Factors")
        st.markdown("""
        - **Age:** Risk increases with age
        - **Genetic mutations:** BRCA1 and BRCA2
        - **Family history:** First-degree relatives with breast cancer
        - **Personal history** of breast cancer or certain non-cancerous breast diseases
        - **Dense breast tissue**
        """)
    
    with col2:
        st.markdown("### Lifestyle-related Risk Factors")
        st.markdown("""
        - **Physical inactivity**
        - **Obesity after menopause**
        - **Hormone replacement therapy**
        - **Reproductive history** (early menstruation, late menopause)
        - **Alcohol consumption**
        """)
    
    # Interactive risk assessment
    with st.expander("Risk Assessment Tool (Demo)"):
        st.markdown("This is a simplified demonstration. Real risk assessment should be done with a healthcare provider.")
        age = st.slider("Age", 18, 90, 40)
        family_history = st.checkbox("Family history of breast cancer")
        genetic_testing = st.checkbox("Known BRCA1/2 mutation")
        
        risk_level = "Low"
        if age > 50:
            risk_level = "Moderate"
        if family_history:
            risk_level = "Moderate to High"
        if genetic_testing:
            risk_level = "High"
            
        st.markdown(f"**Demonstration Risk Level:** {risk_level}")
        st.warning("This is only a demonstration. Please consult with a healthcare provider for an accurate risk assessment.")

def display_early_detection():
    
    st.subheader("Early Detection Methods")
    
    st.markdown("""
    ### Screening Recommendations
    
    - **Monthly breast self-exams** starting at age 20
    - **Clinical breast exams** every 3 years for women in their 20s and 30s, and every year for women 40 and older
    - **Mammograms** every year for women 45 and older
    - **MRI screening** for women at high risk due to family history or genetic factors
    
    ### Signs to Watch For
    
    - A new lump or mass in the breast or underarm
    - Swelling of part of the breast
    - Skin irritation or dimpling
    - Redness or flaky skin in the nipple area
    - Nipple pain or the nipple turning inward
    - Nipple discharge other than breast milk
    - Any change in the size or shape of the breast
    """) 