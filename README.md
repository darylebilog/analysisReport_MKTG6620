# Sales and Brand Analysis Report

## Project Overview
<p align="justify">
As a new data scientist, the Brand Manager and Sales Manager expressed their interest in improving the performance of the orange juice category. Based on the historical data, Minute Maid (MM) yields higher margins compared to Citrus Hill (CH), thus, both managers aim to increase the sales of this brand. There are two sections of problems that we need to solve for this project. Below are the detailed problem statements for each section:
Brand Manager’s Questions: The Brand Manager is interested in understanding the significant variables that affect the probability of a customer buying MM. They specifically want to know the effectiveness of the variables indicated on the dataset provided. In addition, the Brand Manager seeks for recommendations to increase the sale of MM.
Sales Manager’s Questions: On the other hand, the Sales Manager wants to build a predictive model that can inform them about the probability of customers buying MM.

The confidence for each of the recommendations and predictive model should be stated by the end of the project to inform both the Brand Manager and Sales Manager how confident the solutions will be.
Overall, the goal of the project is to address the specific needs of both the brand manager and the sales manager using appropriate statistical analyses and provide actionable recommendations based on the findings.
</p>

## Methods Overview
<p align="justify">
In addressing the challenge of overfitting, I standardized variables using the recipes package's preprocess function and implemented a train/test split with rsample's initial_split function. Cross-validation was employed during hyperparameter tuning for the boosted tree model. The dataset, OJ, was explored, featuring factors like pricing, discounts, special offers, and customer loyalty, with key predictors including PriceCH, PriceMM, DiscCH, DiscMM, SpecialCH, SpecialMM, LoyalCH, and others. A cautious approach to variable selection was taken, excluding those with less informative value or causing multicollinearity issues to enhance model parsimony.
</p>

## Model Evaluation and Interpretation
<p align="justify">
Performance evaluation of logistic regression and boosted tree models relied on accuracy and ROC-AUC metrics, demonstrating comparable outcomes. The choice between models hinged on stakeholder preferences for interpretability versus predictive accuracy. Partial dependence plots (PDPs) with Gradient-Boosted Trees showcased distinct variable influence patterns compared to logistic regression, emphasizing the importance of model choice. The overall analysis encompassed exploratory data exploration, multicollinearity assessment, logistic regression, boosted tree model building, and hyperparameter tuning with cross-validation. Assumptions centered on linearity in logistic regression, absence of multicollinearity, and appropriate hyperparameter choices for the boosted tree model, with interpretations grounded in variable importance, coefficient estimates, and PDPs.
</p>

## Analysis Methods
<p align="justify">
I utilized logistic regression and boosted tree models to analyze the data. Logistic regression helps identify significant predictors for the outcome, while boosted trees offer a more complex, non-linear approach. I chose these methods for their interpretability and predictive power.

## Results and Conclusion

### Logistic Regression Results
•	Identified significant predictors: “PriceCH,” “PriceMM,” “DiscMM,” “LoyalCH,” and “PctDiscMM.”

•	Achieved an accuracy of 82.5% on the test set.

### Boosted Tree Results
•	Tuned hyperparameters: trees = 150, tree depth = 3, learning rate = 0.1.

•	Achieved an accuracy of 19.53% on the test set.

### Model Comparison
•	Logistic and boosted tree models differed significantly in predictive performance.

•	Logistic regression outperformed the boosted tree in accuracy.

### XAI Method - Partial Dependence Plots (PDP)
•	Examined PDP with boosted trees.

•	Compared the influence of predictor variables on outcomes between logistic regression and boosted trees.
