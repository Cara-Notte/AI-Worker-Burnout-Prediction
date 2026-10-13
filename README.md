# AI Worker Burnout Prediction

An exploratory machine learning project investigating how workplace characteristics and AI-related variables can be used to predict worker burnout scores.

The project compares three regression models and examines their prediction errors, generalization performance, and feature importance.

## Project Objective

Predict the continuous `burnout_score` target and compare the performance of:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

This project focuses on predictive modeling. It does not establish whether AI adoption causes burnout.

## Dataset

This project uses the AI Worker Burnout and Attrition Risk Dataset, originally obtained from Kaggle.

Original source: https://www.kaggle.com/datasets/nudratabbas/ai-worker-burnout-and-attrition-risk-dataset
Attributed creator/uploader: Nudrat Abbas (nudratabbas)
Filename: ai_worker_burnout_attrition_2026.csv
Dataset size: 1,500 records and 21 columns
Prediction target: burnout_score
License: Not verified
Data provenance: Not verified; whether the records are observed or synthetically generated remains unknown.

The original Kaggle listing was unavailable when checked on October 14, 2026. Creator attribution was recovered from a secondary article referencing the same dataset.
Because the data collection or generation process could not be verified, this project is presented as an exploratory machine learning analysis. Its findings should not be generalized to real-world workers without further validation.

## Workflow

1. Inspect data types, missing values, duplicates, and category consistency.
2. Explore burnout distributions and relationships between variables.
3. Prepare numerical and categorical features.
4. Compare three regression models using an 80/20 train/test split.
5. Evaluate performance using MAE, RMSE, and R².
6. Perform five-fold cross-validation.
7. Inspect actual-versus-predicted plots, residuals, and tree-based feature importance.

## Preliminary Results

The following results are taken from the saved outputs of the original notebook. They have not been independently reproduced and should be re-evaluated after the preprocessing workflow is corrected.

| Model | Test MAE | Test RMSE | Test R² | Mean CV R² |
|---|---:|---:|---:|---:|
| Linear Regression | 5.50 | 6.91 | 0.629 | 0.615 |
| Random Forest | 5.67 | 7.15 | 0.603 | 0.592 |
| Decision Tree | 7.17 | 9.06 | 0.363 | 0.410 |

Lower MAE and RMSE indicate smaller prediction errors. Higher R² indicates better performance relative to predicting the mean target value.

### Observations

- Linear Regression achieved the best recorded test performance among the three models.
- The Decision Tree showed a substantial training-to-test performance gap: training R² was approximately 0.842, compared with test R² of 0.363.
- Random Forest improved on the Decision Tree but did not outperform Linear Regression.
- Greater model complexity did not produce better test performance in this experiment.

## Repository Structure

- `notebooks/burnout_analysis.ipynb`: analysis and modeling notebook
- `requirements.txt`: Python dependencies
- `data/README.md`: dataset source and download instructions
- `reports/figures/`: selected visualizations

## Running the Project

1. Install the dependencies:

   `pip install -r requirements.txt`

2. Download the dataset from the source listed above.
3. Place the CSV in `data/`.
4. Open `notebooks/burnout_analysis.ipynb`.
5. Ensure the dataset path points to `../data/ai_worker_burnout_attrition_2026.csv`.
6. Restart the kernel and run all cells in order.

The original uploaded notebook uses Google Colab's file-upload interface. Local execution requires updating the loading step and correcting the cell order.

## Limitations and Planned Improvements

- Correct the execution order so the notebook runs from a fresh kernel.
- Fit preprocessing only on training data and within each cross-validation fold.
- Replace nominal-category label encoding with one-hot encoding.
- Review outlier filtering, particularly filtering based on the prediction target.
- Verify whether `attrition_risk` introduces target leakage.
- Add a mean-prediction baseline using `DummyRegressor`.
- Verify dataset provenance and document its implications for generalization.
- Re-run the corrected experiment and update the reported metrics.

Feature importance describes how a fitted model uses predictors; it does not establish causal effects.

The current analysis does not demonstrate performance on workers outside this dataset.

## Author

Dylan Wilson  
Informatics undergraduate, Universitas Multimedia Nusantara
