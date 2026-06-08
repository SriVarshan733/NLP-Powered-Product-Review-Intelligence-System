# NLP-Powered Product Review Intelligence System 🚀

An end-to-end, production-grade ML and analytics engineering pipeline that processes unstructured product reviews, extracts granular sentiment using a fine-tuned Transformer model, uncovers hidden thematic issues via topic modeling, and surfaces insights through a decoupled business intelligence layer.

---

## 🛠️ Architecture Overview

The system is designed with modern data stack patterns, emphasizing scalability, low latency, and distinct separation of concerns:

1. **Ingestion & Preprocessing:** Simulated raw streaming inputs are cleaned using an optimized `spaCy` NLP pipeline (stopword removal, lemmatization, tokenization) and staged into **Google BigQuery**.
2. **Analytics Engineering (dbt):** A modular **dbt (Data Build Tool)** layer transforms raw staged records into structured, production-ready dimension and fact tables inside BigQuery, handling data quality testing and lineage tracking.
3. **Deep Learning Pipeline:**
    - **Sentiment Analysis:** A custom fine-tuned **DistilBERT** model yielding a **91% F1-score**.
    - **Topic Modeling:** **BERTopic** combined with `SentenceTransformers` for advanced, unsupervised thematic clustering (automatically isolating 20+ product topics).
4. **Model Serving API:** A high-performance **FastAPI** application wrapped in a lightweight **Docker container** designed for seamless, auto-scaling deployment to **GCP Cloud Run**.
5. **Business Intelligence Engine:** A rich **Streamlit** executive dashboard enabling both ad-hoc live inference testing and structural warehouse analytics.

---

## 📂 Repository Structure

```text
nlp-review-intelligence/
│
├── data_pipeline/
│   └── preprocess.py          # Step 1 & 2: spaCy cleaning & BigQuery stream ingestion
│
├── models/
│   ├── train_bert.py          # Step 4: HuggingFace DistilBERT fine-tuning pipeline
│   └── topic_model.py         # Step 5: BERTopic thematic extraction
│
├── dbt_transform/             # Step 6: dbt transformation and data mart layer
│   ├── dbt_project.yml        # dbt project global configuration
│   └── models/
│       ├── schema.yml         # Data quality constraints & schema tests
│       ├── stg_reviews.sql    # Staging view preparation
│       └── fct_review_insights.sql # Core optimized analytical fact tables
│
├── api/                       # Step 7: FastAPI Serving Infrastructure
│   ├── Dockerfile             # Container configuration for serverless deployment
│   ├── main.py                # Asynchronous API with lazy-loading performance
│   └── requirements.txt       # Production API dependencies
│
└── app/                       # Step 8: Interactive Visualization layer
    ├── app.py                 # Streamlit UI implementation
    └── requirements.txt       # Frontend dependencies

