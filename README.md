
# Gdańsk Apartment Rent Predictor (with VLM Photo Grading)

## Project Overview
This project aims to predict apartment rental prices in Gdańsk, Poland. While most real estate pricing models rely purely on tabular data (area, number of rooms, location), this project takes it a step further by **quantifying the visual quality of the apartments**. 

Using a locally hosted Vision-Language Model (VLM), I analyzed and scored the photos for each scraped listing to evaluate aspects like modernness, condition, and furnishing quality. These visual scores were then merged with the tabular data to train a machine learning model, resulting in a dataset of ~800 fully processed listings.

## Data Pipeline & Methodology

1. **Data Scraping & Cleaning:** * Scraped rental listing data from Otodom for Gdańsk.
   * Cleaned tabular features including price, area, room count, year built, and various amenities (balcony, elevator, parking, etc.).
2. **Visual Feature Extraction (The Cool Part):**
   * Used **Ollama** running the `openbmb/minicpm-v4` model locally to process listing images.
   * Evaluated photos for specific features: `condition`, `modernness`, `brightness`, and `furnishing_quality`.
   * Aggregated these scores per listing to create new features like `best_overall_quality`, `best_condition`, and `image_score_weighted`.
3. **Merging & Processing:** * Combined the tabular data and the VLM-generated image scores using custom Python scripts (`merger.py`).
4. **Model Training:** * Trained a predictive machine learning model to estimate the `total_price_pln` based on both traditional features and the new visual grading metrics.

## Model Performance

* **R-squared ($R^2$):** 0.4649
* **Mean Absolute Percentage Error (MAPE):** 12.14%
* **Mean Absolute Error (MAE):** 485.23 PLN

### Actual vs. Predicted Prices
This plot shows how closely the model's predictions align with the actual rental prices in the dataset.
![Model Results](https://github.com/Kazik-Tazik/gdansk-rent-predictor/blob/main/plots/model_results.png)

## Interpretability (SHAP Analysis)

To understand what drives apartment prices in Gdańsk, I used SHAP (SHapley Additive exPlanations) to interpret the model's predictions. 

### Feature Importance
Unsurprisingly, `area_m2` is the dominant factor in determining price. However, our custom VLM-generated features (like `image_score_top2_weighted` and `best_overall_quality`) also play a role in the model's decision-making process!

<div align="center">
  <img src="https://github.com/Kazik-Tazik/gdansk-rent-predictor/blob/main/plots/shap_importance.png" width="450" alt="SHAP Feature Importance">
</div>

### SHAP Summary Detail
This detailed beeswarm plot shows *how* each feature impacts the price. For example, larger areas (red dots on the top row) push the predicted price higher, while older buildings (`year_built`) might push it lower depending on the context.

<div align="center">
  <img src="https://github.com/Kazik-Tazik/gdansk-rent-predictor/blob/main/plots/shap_detail.png" width="450" alt="SHAP Summary Detail">
</div>

## Repository Structure

* `dataset_clean_and_visualization.ipynb` - Notebook for exploratory data analysis and cleaning the raw scraped data.
* `photo_grader.py` - Script using the `openbmb/minicpm-v4` LLM to evaluate and score apartment photos.
* `merger.py` - Joins the tabular data with the generated image scores based on listing IDs.
* `model_training.ipynb` - The ML pipeline for training the model and evaluating its performance.
* `otodom_gdansk_rent_clean_new_images.csv` - The final, clean dataset used for training.
