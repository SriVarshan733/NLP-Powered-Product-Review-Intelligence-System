# Inside data_pipeline/preprocess.py
if __name__ == "__main__":
    import os
    
    print("🧹 Starting text preprocessing step...")
    
    # Load from the scraped staging file or read from BigQuery raw table
    if os.path.exists("data_local/raw_reviews.csv"):
        df = pd.read_csv("data_local/raw_reviews.csv")
        print("📖 Loaded raw records from local cache.")
    else:
        # Fallback generated data if run context is isolated
        df = generate_mock_data()
        
    print("✨ Running spaCy tokenizer, lemmatizer, and stopword filters...")
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    
    # Save the cleaned layer back to a preprocessed storage state
    os.makedirs("data_local", exist_ok=True)
    df.to_csv("data_local/cleaned_reviews.csv", index=False)
    print("✅ Preprocessing step complete. Saved to 'data_local/cleaned_reviews.csv'.")