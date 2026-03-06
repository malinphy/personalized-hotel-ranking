# 🏨 Expedia Hotel Search Ranking System (ESMM)

## 📌 Project Overview
This project implements a state-of-the-art **Search & Recommendation Pipeline** for the Expedia Hotel Search dataset. The system features an **ESMM (Entire Space Multi-Task Model)**, an industry-standard architecture for joint optimization of Click-Through Rate (CTR) and Conversion Rate (CVR).

The goal is to personalize search results by ranking hotels based on their relevance to the user's query, effectively handling selection bias in conversion modeling.

---

## 🏗️ Architecture: Entire Space Multi-Task Model (ESMM)

The system is built on the **ESMM** framework, which addresses two major issues in traditional CVR modeling: **Sample Selection Bias (SSB)** and **Data Sparsity (DS)**.

### 1. 🔍 Two-Tower Shared Bottom
*   **Encoders**: User and Item attributes are encoded through shared embedding layers.
*   **Shared Representation**: Both CTR and CVR tasks share the same underlying feature representation, allowing the model to learn from the much larger "Entire Space" of impressions.

### 2. ⚡ Multi-Task Learning (MTL)
*   **CTR Tower**: Predicts the probability of a click (`pCTR`).
*   **CVR Tower**: Predicts the probability of a conversion given a click (`pCVR`).
*   **CTCVR Estimation**: The model is trained to optimize for `pCTCVR = pCTR * pCVR`, which can be supervised using the entire impression space.

---

## 📂 Directory Structure

```plaintext
ADTECH/
├── data/                   # Data storage (gitignored)
│   ├── encoders/           # LabelEncoder artifacts
│   └── expedia/            # Raw datasets
├── models/
│   └── esmm.py             # ESMM Model Architecture (PyTorch)
├── src/
│   ├── data/               # Data Handling
│   │   ├── dataset.py      # PyTorch Dataset for Expedia
│   │   ├── encoders.py     # LabelEncoding management
│   │   ├── preprocess.py   # Raw data to encoded CSV/Parquet
│   │   ├── split_data.py   # Query-based (srch_id) Train/Test/Val split
│   │   └── balance_data.py # Utility for handling label imbalance
│   ├── training/           # Model Training
│   │   ├── trainer.py      # ESMM Training & Evaluation scripts
│   │   └── weights/        # Saved model weights (.pth) and configs
│   └── utils/              # Utilities
│       ├── config.py       # Hyperparameters & Constants
│       └── metrics.py      # Evaluation metrics (NDCG)
├── data_preprocess/        # Preprocessing artifacts
├── requirements.txt        # Full Python dependencies
├── inference.py            # Model Inference & Ranking script
├── eda.ipynb               # Preliminary Data Analysis notebook
└── README.md               # You are here
```

---

## 🚀 Getting Started

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
1. **Preprocessing**: Encode categorical variables and clean data.
   ```bash
   python -m src.data.preprocess
   ```
2. **Data Splitting**: Split the encoded data into train, test, and validation sets (grouped by `srch_id`).
   ```bash
   python -m src.data.split_data
   ```
3. **Data Balancing**: Balance the training set to handle sparse conversion labels. A balanced sampling script was used to ensure an equal distribution of labels (clicks vs. non-clicks, bookings vs. non-bookings) for more stable model convergence.
   ```bash
   python -m src.data.balance_data
   ```

### 3. Training & Evaluation
Train the ESMM model and evaluate using NDCG@38 and AUC:
   ```bash
   python -m src.training.trainer
   ```

### 4. Inference & Ranking
You can run inference on a search query from the test set to see the model's ranking in action:
   ```bash
   # Runs inference on a random search ID from test_split.csv
   python inference.py
   ```

> [!TIP]
> **Checkpoints**: The system automatically saves model weights (`.pth`) and the training configuration to `src/training/weights/` after every epoch. This allows you to resume training or deploy specific model versions.

---

## 📊 Evaluation Metrics

*   **NDCG@38**: Normalized Discounted Cumulative Gain at rank 38 (Expedia page size). Primary ranking quality metric.
---

## 💡 Key Design Decisions
1.  **Entire Space Modeling**: By training on all impressions (not just clicks), we eliminate the bias typically found in conversion models.
2.  **Shared Embeddings**: Shared architecture handles the sparsity of conversion data by leveraging click data.
3.  **Grouped Data Splitting**: Data is split based on `srch_id` (search queries) rather than individual rows. This ensures that all hotels seen by a specific user in a single search stay within the same set (Train or Test), preventing data leakage and ensuring a realistic evaluation of ranking quality.
4.  **Modular Design**: Clean separation of model architecture, data pipeline, and training logic for better maintainability.
5.  **Balanced Sampling Strategy**: To handle extreme label imbalance in the Expedia dataset (where clicks and bookings are sparse), a dedicated sampling script was used to create a balanced training set. This ensures the model learns effectively from both positive interactions (clicks/bookings) and negative impressions by providing an equal distribution of labels during the initial training phase.

---

## 📚 References & Resources

### 🔗 Dataset
*   **Expedia Personalized Sort Competition**: [Kaggle Dataset Link](https://www.kaggle.com/c/expedia-personalized-sort/data)

### 📄 Academic Paper (ESMM)
```bibtex
@article{ma2018esmm,
  title   = {Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate},
  author  = {Ma, Xiao and Zhao, Liqin and Huang, Guan and Wang, Zhi and Hu, Zelin and Zhu, Xiaoqiang and Gai, Kun},
  journal = {arXiv preprint arXiv:1804.07931},
  year    = {2018},
  doi     = {10.48550/arXiv.1804.07931}
}

```

