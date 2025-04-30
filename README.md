# 🧬 Early Cancer Detection Using Multi-Scale CNN and Transformer-Based Deep Learning Approaches

\> **Author:** Xinyu Luo  

\> **Institution:** Oxford Brookes University × Chengdu University of Technology  

\> **Supervisor:** Dr. Grace U. Nneji  

\> **Module Code:** CHC6096

## 📌 Overview

This repository presents an advanced hybrid deep learning framework for **early breast cancer detection** based on **histopathological image analysis**. The model combines **EfficientNetV2** for local feature extraction with **Vision Transformers (ViT)** enhanced by **Shifted Patch Tokenization (SPT)** and **Learned-Scale Attention (LSA)** mechanisms.

Two public histopathology datasets are used:
- **[BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis)**: Breast Cancer Histopathological Image dataset (7,909 samples across 4 magnification levels: 40x, 100x, 200x, 400x)
- **[BACH](https://iciar2018-challenge.grand-challenge.org/)**: BACH Grand Challenge dataset, for cross-dataset validation.

---

## 💡 Key Features

- ✅ **Hybrid CNN-ViT** architecture for combining local and global image understanding  
- ✅ **Shifted Patch Tokenization (SPT)**: enables spatial context enhancement for ViT  
- ✅ **Learned-Scale Attention (LSA)**: dynamic attention scaling for better interpretability  
- ✅ Custom data split and augmentation strategy for training robustness  
- ✅ ROC, confusion matrix, AUC, and other metric visualizations  
- ✅ Integrated **Streamlit GUI** for user-friendly model interaction  
- ✅ Modular TensorFlow 2.13 implementation with GPU memory control

---

## 🧠 Architecture Highlights

| Module                 | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| **EfficientNetV2-B0**  | Extracts multi-scale local features from histopathological images |
| **Vision Transformer** | Captures global context via multi-head self-attention        |
| **SPT**                | Improves ViT patch representation via local shifts           |
| **LSA**                | Learnable scaling in attention to improve focus on relevant regions |
| **Fusion Strategy**    | Concatenation of CNN and ViT branches followed by dense classifier head |

---

## 🖥️ GUI with Streamlit

To launch the **Streamlit-based GUI interface**:

```bash
cd gui
streamlit run main.py
```

The GUI supports:

- Image upload and preprocessing
- Real-time inference using trained hybrid model
- Visual display of prediction (Benign vs. Malignant)
- Confidence scores and Grad-CAM support (optional)

## **⚙️ Installation**

### **Requirements**

```
pip install -r requirements.txt
```

Ensure the following Python libraries are installed:

- tensorflow>=2.13
- scikit-learn
- opencv-python
- streamlit
- matplotlib, seaborn
- pandas, numpy

### **Clone Repository**

```
git clone https://github.com/CarsonLLuo/FinalProject.git
cd FinalProject
```

## **👩‍⚕️ Target Audience**

- Medical professionals (pathologists, oncologists)
- Medical AI researchers
- Developers building AI-assisted diagnostic tools
- Institutions seeking to deploy lightweight, accurate cancer screening models

## **🙏 Acknowledgment**

- Supervisor: **Dr. Grace U. Nneji**
- Module Leader: **Dr. Joojo Walker**
- Dataset providers: [BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis), [BACH Grand Challenge](https://iciar2018-challenge.grand-challenge.org/)
- Special thanks to teammates, dormmates, and collaborators in *一緒Impactですよ* for their unwavering support

## **📜 License**

This project is distributed for educational and research purposes. Please contact the author for any commercial inquiries.
