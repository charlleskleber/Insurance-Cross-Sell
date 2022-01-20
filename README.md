# InsuranceCS
## 
![is-it-better-to-cancel-an-unused-credit-card-or-keep-it-960x450](https://user-images.githubusercontent.com/68538809/142218390-a331bfc7-deb6-44c2-99c0-a8a74ad0ac24.png)



# 1. Business Context.

The company is a health insurance provider. The product team is looking into introducing a new product to their clients: vehicle insurance.

Vehicle Insurance (also known as motor insurance) intends to provide financial protection against physical damage or bodily injury resulting from traffic collisions and against liability that could arise from incidents in a vehicle.
Around 380,000 customers were surveyed to find out if they had an interest in joining a new vehicle insurance product, the answers were stored in the customer's database. 

Outside of the 380,000 clients scope, the product team selected 127,000 customers that were not surveyed to be offered the new vehicle insurance product.

Phone calls will be used to offer the new product, which have an average price of 900€/annum. The sales team has a budget limit of 20,000 calls through the campaign period.

In this context, this project intends to provide the sales team with a solution that ranks the 127,000 by likelihood of purchasing the new insurance to decide which 20,000 potential customers should be contacted and by what order.

## 1.1 Project Deliverables.

In this business context, as a data scientist, I intend to develop a machine learning model that ranks customers by the likelihood of purchasing a new product.
 
With this model, the products team expects to prioritize clients that would be interested in the new product and so optimize sales campaigns by phone calling as much interested clients as possible.

A report with the following analysis is the deliverable goals:

1. Key findings on interested customers' most relevant attributes.

2. How many interested customers will the sales team be able to reach with 20,000 phone calls, how much does that represent of the overall interested custumers, and how much will revenue increase by using the model.

3. How many interested customers would the sales team be able to reach if the budget was increased to 40,000 calls, how much that represents of the overall interested customers, and how much will revenue increase by using the model.

# 2. The Solution

## Solution strategy

The following strategy was used to achieve the goal:

### Step 1. Data Description

The initial dataset has 381.109 rows and 12 columns. Follows the features description:

- **Id**: Unique ID for the customer   
- **Gender**: Gender of the customer 
- **Age**: Age of the customer  
- **Driving License**: 0, customer does not have DL; 1, customer already has DL  
- **Region Code**: Unique code for the region of the customer 
- **Previously Insured**: 1, customer already has vehicle insurance; 0, customer doesn't have vehicle insurance
- **Vehicle Age**: Age of the vehicle 
- **Vehicle Damage**: 1, customer got his/her vehicle damaged in the past; 0, customer didn't get his/her vehicle damaged in the past
- **Anual Premium**: The amount customer needs to pay a premium in the year 
- **Policy sales channel**: Anonymized Code for the channel of outreaching to the customer ie 
- **Vintage**: Number of Days, customer has been associated with the company 
- **Response**: 1, customer is interested; 0, customer is not interested.  

Numerical attributes statistic analysis:

 ![numericalfeatures](https://user-images.githubusercontent.com/68538809/144523197-f12e4d30-6fe9-4cd1-b995-14c54d0728fa.JPG)

Categorical attributes statistic analysis:

![categorical features](https://user-images.githubusercontent.com/68538809/144523226-d0f0728e-1a8c-43be-b7d0-a3fdb3879800.JPG)

### Step 2. Feature Engineering

On feature creation, 2 columns were modified.
'Vehicle Change' values were change from categorical to numerical.
'Vehicle Age' values were changes to a standardized descriptive format.

### Step 3. Exploratory Data Analysis (EDA)

On Exploratory Data Analysis, Univariate, Bivariate, and Multivariates study was performed to help understand the statistical properties of each attribute, correlations, and hypothesis testing.

### Step 4. Data Preparation

In this section, Standardization (StandardScaler Scaler), Rescaling (MinMax Scaler), and Encoding Transformations (Target Encoder) of the variables were carried out.

### Step 5. Feature Selection

To select the features to be used, two methods were used:

1. Application of Feature Importance using ExtraTreesClassifier method;
2. EDA insights.

**From Feature Importance:** 'vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage' and 'policy_sales_channel'

**From EDA:** 'driving_license'

### Step 6. Machine Learning Modelling

Machine learning models were trained and passed through Cross-Validation to evaluate the "real" performance. 

### Step 7. Hyperparameter Fine Tuning

Based on Machine Learning Modelling results, the best model was chosen and submitted to Hyperparameter Fine Tuning to optimize its performance.

### Step 8. Final Model

The production (test) dataset was run through Feature Engineering/Data Preparation/Feature Selection to be used on the optimized model. The output is a ranked list of customers by likelihood of buying the new product.

### Step 9. Performance Evaluation and Interpretation

One of the main steps is the performance evaluation and interpretation, this stage intends to translate the model performance into business value. In this case, correct classification of customers' intention will bring a higher volume of sales inside a limited investment budget.

# 3. Data Insights

# 4. Machine Learning Model Applied

The following Machine Learning models were tested and cross-validated:

- **Logistic Regression**
- **Naive Bayes**
- **Extra Trees**
- **Random Forest Regressor**
- **K-Nearest Neighbors Regressor (KNN)**
- **XGBoost Classifier**
- **LightGBM Classifier**
- **CatBoost Classifier**

# 5. Machine Learning Model Performance

To evaluate the performance of the models, 4 metrics were used:

**ROC AUC**

**Precision at k**

**Recall at k**

**F1_Score**

The indicator "Recall at K" was the one to assume the major importance as it represents the proportion of relevant items found within the total number of the defined items.

The following table discloses the cross-validated ("real") performance of all the models used.
**"LGBM Classifier"** holds the best Recall at k result, making it the one to be used.

![CVResults](https://user-images.githubusercontent.com/68538809/147599534-6757fe6e-e862-4f8f-9827-3d4f27f06d9b.JPG)

Performance plots show the model's ability to identify **all customers interested** in purchasing the product using approximately **50% of the sample** (Cumulative Gain Curve). It is also possible to see that the model is initially **more than 3 times better than a random selection** (Lift Curve). It is important to notice that even after reaching half the sample, the model remains to perform 2 times better than a random method. 

![PerformancePlots](https://user-images.githubusercontent.com/68538809/147599532-9a6bb7ed-e3c5-4083-b03e-a219f68a2de7.png)

# 6. Business Results

## **1.** Key findings on interested customers' most relevant attributes.

### Insight 1. Age

> The average age of interested clients is higher than non-interested clients. Both plots disclose well how younger clients are not as interested as older clients. 

![Insight1](https://user-images.githubusercontent.com/68538809/147419338-12614b60-af99-42fe-bfc9-30e659e43c0f.JPG)

### Insight 2. Vehicles Age

> Interest is greater on customers' with vehicles age over 2 years old.

![Insight2](https://user-images.githubusercontent.com/68538809/147420148-6b78b6fa-43c1-4d33-83db-b8b7da0a8f61.JPG)

### Insight 3. Previously Damaged Vehicle

> Interest is greater on customers' that had his/her vehicle damaged in the past.

![Insight3](https://user-images.githubusercontent.com/68538809/147420356-86dbb2ad-28ff-4484-b8d3-b2a80e2ee8ff.JPG)

### Insight 4. Annual Premium

> Customers with higher health insurance annual premiums are more interested in getting a vehicle insurance.

![Insight4](https://user-images.githubusercontent.com/68538809/147420570-b0968368-2b2d-4675-a919-76b825099d99.JPG)

## **2.** How many interest customers will the sales team be able to reach with 20,000 phone calls.

> With 20,000 calls, the sales team will be able to **reach 45.44%** of the interested customers. This is more 2,829 customers when compared to a random model, representing an increase of 2,546,100€ in revenue.

![Plot20000](https://user-images.githubusercontent.com/68538809/147598392-9bb55d9a-74f1-4fd1-b8ef-e568ef75a222.JPG)

## **3.** How many interested customers would the sales team be able to reach if the budget was increased to 40,000 calls.

With 40,000 calls, the sales team will be able to **reach 78.84%** of the interested customers. This is more 4,510 customers when compared to a random model, representing an increase of 4,059,000€ in revenue.

![Plot40000](https://user-images.githubusercontent.com/68538809/147598393-26a17311-d2f3-4787-800c-203f20e2bf63.JPG)

# 7. Conclusions

Learn to rank models is a special classification task. Businesswise, I have created a ranking list capable of identifying potential customers, this is a major help in terms of managing workflows and guiding the business towards revenue generation and costs reduction. 
The project has demonstrated that using a Machine Learning model is more effective than random sorting.

The sales team will need to call ≈60% of the customers to reach 100% of interested parties. With random sorting, this rate would only be achieved by calling all customers. This concludes that by using data science techniques, the company is able to save approximately 40% in customer contact costs.

# 8. Next Steps to Improve

1. Search for **new features** to improve the final result;

2. **Balance** the response variable to test the results;

3. Improve **fine-tuning** parameters;

4. Implement deployment strategy.