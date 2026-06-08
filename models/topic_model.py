from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def run_topic_modeling(documents):
    # Step 3: Use SentenceTransformers for generating state-of-the-art embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize and fit BERTopic
    topic_model = BERTopic(embedding_model=embedding_model, nr_topics="auto")
    topics, probabilities = topic_model.fit_transform(documents)
    
    # Extract Topic Information
    topic_info = topic_model.get_topic_info()
    return topic_model, topic_info

if __name__ == "__main__":
    docs = [
        "The battery life is exceptional and charges quickly.",
        "Battery drain issue noticed after latest firmware patch.",
        "Customer service refused to refund my broken screen.",
        "UI is crisp, very modern application interface.",
        "Support team was responsive and solved my issue within minutes."
    ]
    model, info = run_topic_modeling(docs)
    print(info)