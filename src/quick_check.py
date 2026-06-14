import os
import duckdb

base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curated_dir = os.path.join(base_dir, "data", "processed")

# use ** wildcard to read all partitions at once
pattern = os.path.join(curated_dir, "facts_*", "**", "*.parquet").replace("\\", "/")

conn = duckdb.connect()

df = conn.execute(f"""
    SELECT make, year, body, body_sentiment_label
    FROM read_parquet('{pattern}', hive_partitioning=true)
    where make='FORD' and year='2015' and body_sentiment_label='positive'
    ORDER BY make, year, body_sentiment_label
""").df()

print(df)