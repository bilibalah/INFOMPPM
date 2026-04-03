# INFOMPPM Recommender System Prototype
This project is a recommender system prototype developed for a public service media use case.  
It combines content-based and collaborative filtering recommendation with fairness, diversity, autonomy, transparency features.

## Project Structure
- `app.py` / `main.py`: 
- `pages/`: Streamlit pages for genres, transparency, and recommendations
- `data/`: datasets used in the prototype
- `ExposureFairness_calculation.ipynb`: fairness calculation
- `syntheticdata.ipynb`: synthetic data generation

## How to Run
1. Run `syntheticdata.ipynb` to generate the synthetic data.
2. Run `ExposureFairness_calculation.ipynb` to compute the inclusion scores.