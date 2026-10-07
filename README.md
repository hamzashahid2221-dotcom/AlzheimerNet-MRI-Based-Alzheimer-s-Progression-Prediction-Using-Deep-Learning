  # 🧠 AI-Assisted Detection of Alzheimer’s Disease Stages

This repository presents my research project on developing an **AI-assisted diagnostic model** for detecting stages of **Alzheimer’s Disease** using **Deep Learning**.  
The model leverages a **fine-tuned ResNet50** architecture combined with an **Adaptive Categorical Focal Loss** to handle class imbalance and enhance generalization.

---

## 🎯 Project Overview

Early and accurate detection of Alzheimer’s Disease (AD) is critical for **timely intervention** and **management**.  
Manual diagnosis via MRI scans or clinical assessments can be **time-consuming** and **subject to human variability**.

To address this, I developed a **computer-aided diagnostic (CAD)** system that automatically classifies patients into:

- **Non Demented**  
- **Very Mild Dementia**  
- **Mild Dementia**  
- **Moderate Dementia**

This work lies at the intersection of **AI and medical imaging**, contributing to **neuroimaging analysis** for cognitive disorders.

---

## 🧩 Dataset

The project uses the **OASIS-1 dataset**, a publicly available MRI dataset widely used in Alzheimer’s research.

**Dataset Description:**

| Source | Description | Classes |
|--------|-------------|--------|
| [OASIS-1 Dataset](https://www.oasis-brains.org/) | T1-weighted MRI scans of subjects aged 18–96 | Non Demented, Very Mild Dementia, Mild Dementia, Moderate Dementia |

**Data Split:**
- 70% Training  
- 15% Validation  
- 15% Testing  

**Classes:**
- Non Demented  
- Very Mild Dementia  
- Mild Dementia  
- Moderate Dementia  

## 🧠 Model Architecture

**Base Model:** ResNet50 (pretrained on ImageNet)

- The base model is **frozen initially** to leverage pretrained features.  
- Later, the **last 20 layers are unfrozen** for **fine-tuning**, allowing the model to learn domain-specific neuroimaging patterns.  
- Input MRI scans are preprocessed and resized to match ResNet50 requirements.

---

## ⚙️ Adaptive Categorical Focal Loss

To handle **class imbalance**, particularly due to fewer samples of Moderate Dementia, the model employs an **adaptive categorical focal loss**:

\[
FL(p_t) = - \alpha_t (1 - p_t)^{\gamma_t} \log(p_t)
\]

Where:  
- \( p_t \): Predicted probability of the true class  
- \( \alpha_t \): Adaptive class weight (updates per epoch)  
- \( \gamma_t \): Focusing parameter emphasizing hard-to-classify examples  

This ensures high precision and recall across all dementia stages.
Key Innovations:
1. **Dynamic per-class adaptation:**  
   - Unlike standard focal loss, the **class weight** (\(\alpha_t\)) and **focusing parameter** (\(\gamma_t\)) are **adjusted during training** based on class-wise performance metrics (e.g., recall).  
   - This allows the model to focus more on harder-to-classify or underrepresented classes.

2. **Self-adjusting loss mechanism:**  
   - The adaptive loss forms a **feedback loop**, where model performance informs how the loss function emphasizes each class.  
   - This reduces manual hyperparameter tuning and promotes balanced learning.

3. **Stable updates:**  
   - Adjustments are applied carefully to prevent abrupt changes, ensuring **robust and stable training**.

> **Note:** The full implementation details of this adaptive loss function are part of ongoing research and are available upon request or will be published in a forthcoming manuscript.
---

## 🚀 Training Strategy

- **Optimizer:** Adam (LR = 0.001 → adaptive reduction)  
- **Batch Size:** 64  
- **EarlyStopping:** Patience = 4 (restore best weights)  
- **ModelCheckpoint:** Saves best model on validation loss  
- **Fine-Tuning Phase:** Unfreeze last 20 ResNet50 layers  

---

## 📊 Results

The model achieves **near-perfect classification** across all classes:

- **Macro Avg F1:** 1.00  
- **Weighted Avg F1:** 1.00  
- **Test Accuracy:** 100%  

The system demonstrates high reliability and potential for clinical applications.

---

## 🔬 Interpretability

Post-hoc interpretability can be added using:

- **Grad-CAM / Grad-CAM++** for visualizing discriminative regions in MRI scans  
- **Integrated Gradients** to understand voxel contributions  

This supports **clinician trust** and **explainable AI** in neuroimaging.

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Deep Learning Framework |
| NumPy, Pandas | Data Handling |
| Matplotlib, Seaborn | Visualization |
| scikit-learn | Evaluation Metrics |
| OASIS Dataset | MRI Dataset for Alzheimer’s Disease |

---

## 📁 Repository Structure

---

## 🎓 Research Significance

This project demonstrates how **transfer learning combined with adaptive loss functions** can accurately classify **Alzheimer’s Disease stages**.  
The model provides a **scalable tool** for **early detection**, addressing challenges in neurodegenerative disease diagnosis.

---

## 👨‍💻 Author

**Hamza Shahid**  
Bachelor of Biomedical Engineering (with Distinction)  
University of Engineering & Technology (UET), Lahore  

🔍 Research Interests:  
AI in Healthcare • Neuroimaging Analysis • Deep Learning for Diagnostics  


---

## 📜 License

This repository is released under the **MIT License** — freely available for academic and research purposes.

---

## 🌍 Acknowledgment

Thanks to the **OASIS dataset contributors** and the open-source AI community for enabling reproducible neuroimaging research.



