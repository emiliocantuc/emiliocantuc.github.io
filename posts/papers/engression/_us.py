# %% [markdown]
# ### How to use this Utility script
# 
# To use this script to perform a local evaluation in the ["Probabilistic forecasting I: Temperature"](https://www.kaggle.com/competitions/probabilistic-forecasting-i-temperature) competition attach this notebook to your competition notebook by, when in the editor mode, going to 
# the top right hand side **Notebook Input** section, then **+Add Input** and search for "`[Utility Script] CRPS score`" and click on **⊕**.
# 
# Then, within a notebook code cell:
# 
# ```python
# import utility_script_crps_score as us
# help(us.crps)
# 
# us.crps(submission_df, solution_df)
# ```
# 
# where the `submission_df` contains the (21) predictions columns, and the `solution_df` contains a ground truth `Temperature` column.
# 
# For more on Kaggle utility scripts see the topic ["*Feature Launch: Import scripts into notebook kernels*"](https://www.kaggle.com/discussions/product-feedback/91185).
# 
# There is also the function `coverage_report` that provides information regarding the coverage of ones calculations.

# %%
def crps(submission, solution):
    """
    This routine returns the mean continuous ranked probability score (CRPS).
    Each individual CRPS score is numerically integrated using 23 points.
    The extremal points (100% coverage) are competition fixed at -30 and 60.
    The "submission" dataframe: the last 21 columns should be the predictions
    The "solution" dataframe must contain a "Temperature" column (the "ground truth")
    
    Author: Carl McBride Ellis
    Version: 1.0.0
    Date: 2024-03-30
    """
        
    # A list of the requested quantile values, along with added 100% coverage endpoints 
    # (these values are all competition fixed)
    # the 0.5 quantile is the "zero coverage" forecast i.e. the median point prediction
    quantiles = [0.00, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 1.00]
    submission_tmp = submission.copy()
    # inset the y_true values to the submission_tmp dataframe to the LHS
    submission_tmp.insert(0, "Temperature", solution["Temperature"].values)
    
    CRPS = 0
    for index, row in submission_tmp.iterrows():
        x_values = row[-(len(quantiles)-2):] # column name agnostic
        y_true = row["Temperature"] # the ground truth value
        
        x_values = [float(i) for i in x_values] # make sure all x values are floats
        # add extremal 100% quantile x-values so as to be sure to bracket all possible y_true values
        # note: any changing of these values will change the score
        x_values.append(-30.0)
        x_values.append( 60.0)
        x_values.sort() # sort x values into ascending order (no quantile crossing)

        # split predictions to the left and right of the true value
        # get items below the true value (y_true)
        LHS_keys = [i for i,x in enumerate(x_values) if x < y_true]
        # get items above the true value (y_true)
        RHS_keys = [i for i,x in enumerate(x_values) if x >= y_true]

        # quantiles and predictions below the true value (y_true)
        LHS_values = [x_values[i] for i in LHS_keys]
        LHS_quantiles = [quantiles[i] for i in LHS_keys]

        # quantiles and predictions above the true value (y_true)
        RHS_values = [x_values[i] for i in RHS_keys]
        RHS_quantiles = [quantiles[i] for i in RHS_keys]

        # also calculate quantile at y (q_at_y_true)
        x1, y1 = LHS_values[-1], LHS_quantiles[-1]
        x2, y2 = RHS_values[0], RHS_quantiles[0]
        q_at_y_true = ((y2-y1)*(y_true-x1)/(x2-x1))+y1

        # add y_true and q_at_y_true to RHS of LHS list
        LHS_values.append(y_true)
        LHS_quantiles.append(q_at_y_true)

        # add y_true and q_at_y_true to LHS of RHS list
        RHS_values.insert(0, y_true)
        RHS_quantiles.insert(0, q_at_y_true)

        # integrate the LHS as a sum of trapezium for CDF**2
        LHS_integral = 0
        for i in range(len(LHS_values)-1):
            LHS_integral += (0.5 * (LHS_values[i+1]-LHS_values[i]) * (LHS_quantiles[i]**2 + LHS_quantiles[i+1]**2) )

        # integrate the RHS as a sum of trapezium for (1-CDF)**2
        RHS_integral = 0
        for i in range(len(RHS_values)-1):
            RHS_integral += (0.5 * (RHS_values[i+1]-RHS_values[i]) * ((1-RHS_quantiles[i])**2 +(1-RHS_quantiles[i+1])**2 ) )

        CRPS += (LHS_integral + RHS_integral)

    del submission_tmp
    # calculate the mean CRPS
    CRPS = CRPS/len(submission)
    return CRPS


def coverage_report(submission, solution):
    """
    Version: 1.0.1
    """
    y_true = solution["Temperature"].values
    # this does not take the "zero coverage" prediction into account
    # which is assumed to be located in submission.csv column -11
    coverages = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    N = len(coverages)
    # ANSI color codes
    BOLD_RED = '\033[1;31m'
    BOLD_GREEN = '\033[1;32m'
    END_COLOR = '\033[0m'
    
    def mean_coverage(y_pred_low,y_true,y_pred_up):
        return ( (y_pred_low <= y_true) & (y_pred_up >= y_true) ).mean()
    
    for i, coverage in enumerate(coverages):
        lower_col, upper_col = (2*N+1-i), (i+1)
        actual_coverage = mean_coverage(submission.iloc[:,-lower_col], y_true, submission.iloc[:,-upper_col])
        actual_coverage = round(actual_coverage*100,2)
        if actual_coverage >= coverages[i]:
            print(BOLD_GREEN, "Ideal: {}% Actual: {}% [PASS]".format(coverage, actual_coverage), END_COLOR)
        else:
            print(BOLD_RED, "Ideal: {}% Actual: {}% [FAIL]".format(coverage, actual_coverage), END_COLOR)

# %% [markdown]
# #### Reading related to the CRPS metric
# * [James E. Matheson, Robert L. Winkler "*Scoring Rules for Continuous Probability Distributions*", Management Science **22** pages 1087-1096 (1976)](https://doi.org/10.1287/mnsc.22.10.1087)
# * [Tilmann Gneiting and Adrian E Raftery "*Strictly Proper Scoring Rules, Prediction, and Estimation*", Journal of the American Statistical Association, **102**, pp. 359-378 (2007)](https://doi.org/10.1198/016214506000001437) (Section 4.2)
# * [Michaël Zamo, Philippe Naveau "*Estimation of the Continuous Ranked Probability Score with Limited Information and Applications to Ensemble Weather Forecasts*", Math Geosci **50** pages 209-234 (2018)](https://doi.org/10.1007/s11004-017-9709-7)
# * [Johannes Bracher, Evan L. Ray, Tilmann Gneiting, Nicholas G. Reich "*Evaluating epidemic forecasts in an interval format*",  PLOS Computational Biology **17** e1008618 (2021)](https://doi.org/10.1371/journal.pcbi.1008618) 


