{{ config(materialized='table') }}

with staging as (
    select * from {{ ref('stg_reviews') }}
)

select
    review_id,
    product_id,
    raw_review_text,
    user_rating,
    -- Simple analytics logic placeholders for Looker dashboard consumption
    case 
        when user_rating >= 4 then 'Positive'
        when user_rating = 3 then 'Neutral'
        else 'Negative'
    end as manual_sentiment_bracket,
    ingested_at
from staging