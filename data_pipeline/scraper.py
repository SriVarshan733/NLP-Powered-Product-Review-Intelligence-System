import pandas as pd
from google.cloud import bigquery
import json
import os

# 1. Simulate Raw Scraping Logic (e.g., pulling reviews from an App Store or Amazon API)
def scrape_raw_reviews():
    """
    Simulates scraping raw, messy reviews from an external product page.
    """
    print("🕷️ Scraping target product review pages...")
    raw_scraped_data = [
        {"review_id": "1", "product_id": "P100", "review_text": "The battery life on this phone is absolutely amazing! Highly recommend.", "rating": 5},
        {"review_id": "2", "product_id": "P100", "review_text": "Terrible customer service. Screen cracked within two days.", "rating": 1},
        {"review_id": "3", "product_id": "P101", "review_text": "Decent sound quality, but the headphones hurt my ears after an hour.", "rating": 3},
        {"review_id": "4", "product_id": "P102", "review_text": "App crashes constantly. Terrible UI update.", "rating": 1},
        {"review_id": "5", "product_id": "P100", "review_text": "Great value for money. Fast shipping.", "rating": 4},
    ]
    return pd.DataFrame(raw_scraped_data)

# 2. Upload Unprocessed Data directly to your Data Lake Landing Zone
def stream_to_landing_zone(df, dataset_id, table_id):
    """
    Streams raw data straight into BigQuery raw schema landing zone without modifications.
    """
    try:
        bq_client = bigquery.Client()
        table_ref = bq_client.dataset(dataset_id).table(table_id)
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND", # Append data as it is continuously scraped
            source_format=bigquery.SourceFormat.CSV,
            autodetect=True,
        )
        
        print(f"📥 Pushing {len(df)} raw records into BigQuery landing table...")
        job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result() # Wait for the load job to complete
        print(f"✅ Ingestion successful to {dataset_id}.{table_id}")
    except Exception as e:
        print(f"⚠️ GCP Credentials missing. Running in local fallback mode. Logs:\n{e}")
        # Local development fallback
        os.makedirs("data_local", exist_ok=True)
        df.to_csv("data_local/raw_reviews.csv", index=False)
        print("💾 Saved raw records locally to 'data_local/raw_reviews.csv'")

if __name__ == "__main__":
    DATASET = "nlp_review_intelligence"
    RAW_TABLE = "raw_reviews"
    
    # Run the scraping process
    df_raw = scrape_raw_reviews()
    
    # Ingest directly into the warehouse landing zone
    stream_to_landing_zone(df_raw, DATASET, RAW_TABLE)