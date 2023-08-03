1. Load in input/output
    - DataFrames in pickle
    - Dimension time x parcel(grid)
2. Manually align the space and time
3. Manually data separation
4. Convert to numpy and flat to 1D (mixing time and space per parcel)
5. Attaching input and output attributes together:
    - `X`: input
    - `X_H`: input for testing
    - `Y`: output, only using the CR
    - `Y_H`: output for testing
6. Converting Y to 10 logarithm 
7. Rebuild a DataFrame merging from X and Y, export to pickle 
8. Setup Exhaustive Parameter Searching (`GridSearchCV`)
    - estimator: a pipeline with preprosessor + estimator
    - param_grid: parameter search grid
    - cv: cross validation search grid
    - by default 1 job
9. Call `fit` (expensive step, takes ~27mins)
10. Export model as pickle
11. Testing and evaluation on test dataset: `X_H` and `Y_H`. Caculate follwing coefficients:
    mean_squared_error, mean_absolute_error, coefficient of determination (`r2_score`), Pearson Correlation (`pearsonr`), Spearman Rank-order Correlation (`spearmanr`)
12. Compute correlation between estimation and observations, using test datasets: `X_H` and `Y_H`

Question:
1. Why only using the CR not VH, VV
2. Why only use 1 yr?
