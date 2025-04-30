import streamlit as st
from PIL import Image
import time

from model_utils import mock_prediction
from data_utils import preprocess_image
from ui_components import (
    setup_page,
    setup_sidebar,
    display_detection_results,
    display_instructions,
    display_footer
)
from pages import display_about_page, display_educational_resources

def main():
    # Set up basic page configuration
    setup_page()
    
    # Set up sidebar and get current page
    current_page = setup_sidebar()
    
    if current_page == "Detection Tool":
        # Create two tabs
        tab1, tab2 = st.tabs(["Image Upload", "Results Analysis"])
        
        with tab1:
            # Image upload tab
            st.markdown("### Upload Image")
            uploaded_file = st.file_uploader("Select a histopathological image", type=["jpg", "jpeg", "png", "bmp"])
            
            # Add demo option
            use_demo = st.checkbox("Use demo image instead")
            
            if use_demo:
                st.success("Demo mode activated. Click 'Detect' to analyze a sample image.")
                uploaded_file = "demo"  # Flag for demo mode
            
            
            
            # Display uploaded image
            if uploaded_file is not None or use_demo:
                if uploaded_file != "demo":
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", width=400)
                else:
                    st.image("/Users/carson/Desktop/code/Breast/Example.png", caption="Demo Image", width=400)
                    image = Image.open("/Users/carson/Desktop/code/Breast/Example.png")
                
                # Detection button
                detect_button = st.button("Detect", type="primary")
                if detect_button:
                    with st.spinner("Analyzing..."):
                        # Add a progress bar
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Preprocess image and make prediction
                        processed_image = preprocess_image(image)
                        prediction = mock_prediction(processed_image)
                        
                        # Store results
                        st.session_state.prediction_made = True
                        st.session_state.prediction = prediction
                        
                        # Switch to results tab
                        st.success("Analysis complete! View results in the Results Analysis tab.")
            else:
                display_instructions()
        
        with tab2:
            # Results analysis tab
            if 'prediction_made' in st.session_state and st.session_state.prediction_made:
                display_detection_results(st.session_state.prediction, st)
            else:
                st.info("No analysis has been performed yet. Please upload an image and click 'Detect' in the Image Upload tab.")
    
    elif current_page == "About":
        display_about_page()
    
    elif current_page == "Educational Resources":
        display_educational_resources()
    
    # Display footer
    display_footer()

if __name__ == "__main__":
    main() 