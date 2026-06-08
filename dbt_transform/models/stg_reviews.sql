with source as (
    select * from {{ source('raw_data', 'raw_reviews') }}
)

select
    cast(review_id as string) as review_id,
    cast(product_id as string) as product_id,
    cast(review_text as string) as raw_review_text,
    cast(cleaned_text as string) as cleaned_review_text,
    cast(rating as int64) as user_rating,
    current_timestamp() as ingested_at
from source