# INFOMPPM Recommender System Prototype
This project is a recommender system prototype developed for a public service media use case.  
It combines content-based and collaborative filtering recommendation with fairness, diversity, autonomy, transparency features.

## Project Structure
- `pages/`: Streamlit pages for genre browsing, transparency, and recommendations.
- `data/`: Datasets used in the prototype.
- `main.py`: Entry point of the Streamlit application.
- `recommendations.py`: Functions for generating content-based and collaborative recommendation scores.
- `ExposureFairness_calculation.ipynb`: Notebook for fairness metric calculation.
- `syntheticdata.ipynb`: Notebook for generating the synthetic dataset.
- `data_prep.py`: Script for preprocessing the recommendation data.


## How to Run
1. Run `syntheticdata.ipynb` to generate the synthetic dataset.
2. Run `ExposureFairness_calculation.ipynb` to compute the inclusion scores.
3. Run `data_prep.py` to preprocess the recommendation data.
4. Run `recommendations.py` to generate the baseline relevance scores.
5. Navigate to the app folder.
6. Run `streamlit run main.py` in Terminal to launch the application.
