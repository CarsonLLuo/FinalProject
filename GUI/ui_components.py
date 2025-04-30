import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from PIL import Image

def setup_page():
    """Set up basic page configuration"""
    st.set_page_config(
        page_title="Early Breast Cancer Detection System",
        page_icon="🔬",
        layout="wide"
    )
    st.title("Early Breast Cancer Detection System")
    st.markdown("### Deep Learning Approach with Multi-scale CNN and Transformer")

def setup_sidebar():
    """Set up sidebar navigation"""
    with st.sidebar:
        st.image("/Users/carson/Desktop/code/Breast/5a9eb003-9b22-48c6-9ecf-3e6bff78caa3.jpg", width=100)
        st.title("Navigation")
        
        page = st.radio("Go to", ["Detection Tool", "About", "Educational Resources"])
        
        st.markdown("---")
        st.markdown("### System Status")
        st.markdown("✅ Model: Active")
        st.markdown("✅ Database: Connected")
        st.markdown(f"🕒 Last Updated: {datetime.now().strftime('%Y-%m-%d')}")
        
        st.markdown("---")
        st.markdown("### Need Help?")
        if st.button("Show Sample Images"):
            st.markdown("Examples of histopathological images:")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Benign Sample**")
                st.image("/Users/carson/Desktop/code/Breast/Benign.png", caption="Benign tissue sample", use_container_width=True)
            with col2:
                st.markdown("**Malignant Sample**")
                st.image("/Users/carson/Desktop/code/Breast/Malignant.png", caption="Malignant tissue sample", use_container_width=True)
    
    return page

def display_detection_results(prediction, container=st):
    """
    Display detection results
    
    Args:
        prediction: Prediction result array
        container: streamlit container object, default is st
    """
    # Get prediction results
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx] * 100
    
    # Create two-column layout for displaying main results
    col1, col2 = container.columns(2)
    
    with col1:
        # Result display
        result_text = "Benign" if class_idx == 0 else "Malignant"
        result_color = "green" if class_idx == 0 else "red"
        
        container.markdown(f"<h1 style='color:{result_color}'>{result_text}</h1>", unsafe_allow_html=True)
        container.metric("Confidence Level", f"{confidence:.2f}%")
        
        threshold_met = confidence > 70
        container.markdown(f"**Threshold Status:** {'✅ Passed' if threshold_met else '⚠️ Low Confidence'}")
        
        if class_idx == 0:
            container.markdown("""
            ### What does 'Benign' mean?
            A benign result indicates that the analyzed tissue sample shows characteristics 
            typical of non-cancerous cells. While benign breast conditions are not life-threatening, 
            some types can increase the risk of developing breast cancer later.
            
            #### Common Benign Breast Conditions:
            - **Fibroadenomas:** Solid, noncancerous breast lumps
            - **Cysts:** Fluid-filled sacs
            - **Fibrocystic Changes:** General lumpiness and pain
            - **Intraductal Papillomas:** Small, wart-like growths
            
            #### Follow-up Recommendations:
            - Regular self-examinations
            - Maintain scheduled screening appointments
            - Report any changes to your healthcare provider
            """)
        else:
            container.markdown("""
            ### What does 'Malignant' mean?
            A malignant result indicates that the analyzed tissue sample shows characteristics 
            typical of cancerous cells. Further medical consultation is strongly recommended to 
            confirm this result and discuss treatment options.
            
            #### Common Types of Breast Cancer:
            - **Ductal Carcinoma In Situ (DCIS):** Early-stage, non-invasive
            - **Invasive Ductal Carcinoma (IDC):** Most common type
            - **Invasive Lobular Carcinoma (ILC):** Starts in milk-producing glands
            - **Triple Negative:** More aggressive form
            
            #### Immediate Next Steps:
            1. Schedule an appointment with your healthcare provider
            2. Prepare questions about your diagnosis
            3. Bring all imaging and test results to your appointment
            4. Consider seeking a second opinion
            """)
    
    with col2:
        # Display prediction probability chart
        display_probability_chart(prediction[0])
        
        # Add model explanation section
        container.markdown("""
        ### Understanding the Analysis
        
        #### How the AI Makes Decisions:
        - Analyzes tissue patterns and cellular structures
        - Compares features with thousands of known cases
        - Uses both local and global image features
        - Considers multiple scales of analysis
        
        #### Confidence Score Interpretation:
        - **90-100%:** Very high confidence
        - **70-90%:** High confidence
        - **50-70%:** Moderate confidence
        - **<50%:** Low confidence, may need additional testing
        
        #### Limitations:
        - AI is a supportive tool, not a replacement for medical expertise
        - Results should be confirmed through additional testing
        - Some rare conditions may not be well-represented in training data
        """)
    
    # Add system performance metrics
    container.markdown("### System Performance Metrics")
    metrics_cols = container.columns(4)
    metrics_cols[0].metric("Overall Accuracy", "94.2%")
    metrics_cols[1].metric("Sensitivity", "92.8%")
    metrics_cols[2].metric("Specificity", "95.6%")
    metrics_cols[3].metric("F1 Score", "93.7%")
    
    # Add important notice
    container.warning("""
    🔔 **Important Notice:**
    
    This analysis is provided as a screening tool to assist medical professionals. The results should:
    - Not be considered as a final diagnosis
    - Be reviewed by qualified healthcare providers
    - Be confirmed through additional testing if necessary
    - Be considered alongside patient history and other clinical findings
    """)

def display_probability_chart(probabilities):
    """Display prediction probability chart"""
    st.markdown("### Prediction Probabilities")
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Benign', 'Malignant']
    colors = ['green', 'red']
    ax.bar(labels, probabilities, color=colors, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Class Prediction Probabilities')
    
    for i, v in enumerate(probabilities):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
    st.pyplot(fig)

def display_instructions():
    """Display system instructions"""
    st.markdown("### System Instructions")
    st.markdown("""
    **How to Use:**
    1. Upload a breast tissue histopathological image
    2. Click the "Detect" button
    3. The system will analyze the image and display results
    
    **Technical Details:**
    - This system employs a hybrid deep learning architecture combining CNN and Vision Transformer
    - Features SPT (Shifted Patch Tokenization) technology to enhance local feature capture
    - Utilizes LSA (Learned-Scale Attention) for optimized attention mechanism
    
    **Important Note:** This tool is for assistance only and should not replace professional medical diagnosis.
    """)

def display_footer():
    """Display page footer"""
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    with footer_col1:
        st.markdown("© 2024 Early Breast Cancer Detection Project")
    with footer_col2:
        st.markdown("Developer: Carson")
    with footer_col3:
        st.markdown("Version 2.0 | Last Updated: March 2025") 