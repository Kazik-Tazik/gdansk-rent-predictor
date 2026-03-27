import re
import pandas as pd

APARTMENTS_CSV = "otodom_gdansk_rent_dirty.csv"
IMAGE_SCORES_CSV = "listing_image_scores.csv"
OUTPUT_CSV = "otodom_gdansk_rent_with_images.csv"

# Which image features to keep in the merged file
IMAGE_FEATURE_COLUMNS = [
    "listing_id",
    "has_image_score",
    "total_images_seen",
    "scored_images",
    "relevant_image_count",
    "overview_image_count",
    "usable_image_count",
    "avg_confidence_all",
    "avg_confidence_usable",
    "image_score_mean",
    "image_score_weighted",
    "image_score_top2_weighted",
    "image_score_max",
    "best_overall_quality",
    "best_condition",
    "best_modernness",
    "best_furnishing_quality",
    "kitchen_quality_weighted",
    "bathroom_quality_weighted",
]

def extract_listing_id_from_url(url):
    if pd.isna(url):
        return None
    url = str(url)
    m = re.search(r"-ID([A-Za-z0-9]+)", url)
    if m:
        return m.group(1)
    return None

# Load files
apartments = pd.read_csv(APARTMENTS_CSV)
image_scores = pd.read_csv(IMAGE_SCORES_CSV)

# Extract join key from apartment URLs
apartments["listing_id"] = apartments["url"].apply(extract_listing_id_from_url)

# Keep only needed image columns that actually exist
image_cols = [c for c in IMAGE_FEATURE_COLUMNS if c in image_scores.columns]
image_scores = image_scores[image_cols].copy()

# Drop duplicate listing IDs in image scores just in case
image_scores = image_scores.drop_duplicates(subset=["listing_id"], keep="first")

# Merge
merged = apartments.merge(
    image_scores,
    on="listing_id",
    how="left",
    validate="many_to_one"
)

# Fill helper flag for rows with no matched image score
if "has_image_score" in merged.columns:
    merged["has_image_score"] = merged["has_image_score"].fillna(0).astype(int)
else:
    merged["has_image_score"] = merged["image_score_weighted"].notna().astype(int)

# Save
merged.to_csv(OUTPUT_CSV, index=False)

# Diagnostics
matched = int(merged["has_image_score"].sum())
total = len(merged)
unmatched = total - matched

print("Done.")
print(f"Apartment rows: {total}")
print(f"Matched image scores: {matched}")
print(f"Unmatched apartment rows: {unmatched}")
print(f"Saved to: {OUTPUT_CSV}")

print("\nExample matches:")
cols_to_show = [c for c in ["url", "listing_id", "image_score_weighted", "image_score_top2_weighted", "best_overall_quality"] if c in merged.columns]
print(merged[cols_to_show].head(10).to_string(index=False))
