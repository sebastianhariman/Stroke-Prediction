Stroke Data Prediction
================
Sebastian Hariman
2022-07-13

\#Load Data & Libraries

``` r
library(MASS)
library(car)
```

    ## Loading required package: carData

``` r
library(ROCR)
library(rpart)
library(rpart.plot)
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following object is masked from 'package:car':
    ## 
    ##     recode

    ## The following object is masked from 'package:MASS':
    ## 
    ##     select

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(Hmisc)
```

    ## 
    ## Attaching package: 'Hmisc'

    ## The following objects are masked from 'package:dplyr':
    ## 
    ##     src, summarize

    ## The following objects are masked from 'package:base':
    ## 
    ##     format.pval, units

``` r
strokeData <- read.csv("/cloud/project/healthcare-dataset-stroke-data.csv")
```

\#EDA

``` r
dim(strokeData)
```

    ## [1] 5110   12

    The strokeData dataset has 5110 records and 12 attribute

``` r
str(strokeData)
```

    ## 'data.frame':    5110 obs. of  12 variables:
    ##  $ id               : int  9046 51676 31112 60182 1665 56669 53882 10434 27419 60491 ...
    ##  $ gender           : chr  "Male" "Female" "Male" "Female" ...
    ##  $ age              : num  67 61 80 49 79 81 74 69 59 78 ...
    ##  $ hypertension     : int  0 0 0 0 1 0 1 0 0 0 ...
    ##  $ heart_disease    : int  1 0 1 0 0 0 1 0 0 0 ...
    ##  $ ever_married     : chr  "Yes" "Yes" "Yes" "Yes" ...
    ##  $ work_type        : chr  "Private" "Self-employed" "Private" "Private" ...
    ##  $ Residence_type   : chr  "Urban" "Rural" "Rural" "Urban" ...
    ##  $ avg_glucose_level: num  229 202 106 171 174 ...
    ##  $ bmi              : chr  "36.6" "N/A" "32.5" "34.4" ...
    ##  $ smoking_status   : chr  "formerly smoked" "never smoked" "never smoked" "smokes" ...
    ##  $ stroke           : int  1 1 1 1 1 1 1 1 1 1 ...

    1. Most of the attributes in this dataset have a reasonable name
    2. The dataset has 4 integer attributes, 6 character attributes, and 2 numeric attributes.

``` r
sapply(strokeData, function(x) sum(is.na(x)))
```

    ##                id            gender               age      hypertension 
    ##                 0                 0                 0                 0 
    ##     heart_disease      ever_married         work_type    Residence_type 
    ##                 0                 0                 0                 0 
    ## avg_glucose_level               bmi    smoking_status            stroke 
    ##                 0                 0                 0                 0

    All attributes in the dataset doesn't have a missing value

``` r
sapply(strokeData, function(x) length(unique(x)))
```

    ##                id            gender               age      hypertension 
    ##              5110                 3               104                 2 
    ##     heart_disease      ever_married         work_type    Residence_type 
    ##                 2                 2                 5                 2 
    ## avg_glucose_level               bmi    smoking_status            stroke 
    ##              3979               419                 4                 2

    1. All values in id attributes are distinct
    2. As they are a numeric and integer attributes, the age, glucose level, and bmi has more than 100 unique values
    3. The gender, hypertension, heart_disease, ever_married, work_type, residence_type, smoking_status, stroke attributes has less than 6 unique values

``` r
#Turn attribute with under 6 unique value to Factor
under_6 <- sapply(strokeData, function(col) length(unique(col)) < 6)
strokeData[ , under_6] <- lapply(strokeData[ , under_6] , factor)

#Turn bmi from chr to num
strokeData$bmi <- as.numeric(strokeData$bmi)
```

    ## Warning: NAs introduced by coercion

``` r
str(strokeData)
```

    ## 'data.frame':    5110 obs. of  12 variables:
    ##  $ id               : int  9046 51676 31112 60182 1665 56669 53882 10434 27419 60491 ...
    ##  $ gender           : Factor w/ 3 levels "Female","Male",..: 2 1 2 1 1 2 2 1 1 1 ...
    ##  $ age              : num  67 61 80 49 79 81 74 69 59 78 ...
    ##  $ hypertension     : Factor w/ 2 levels "0","1": 1 1 1 1 2 1 2 1 1 1 ...
    ##  $ heart_disease    : Factor w/ 2 levels "0","1": 2 1 2 1 1 1 2 1 1 1 ...
    ##  $ ever_married     : Factor w/ 2 levels "No","Yes": 2 2 2 2 2 2 2 1 2 2 ...
    ##  $ work_type        : Factor w/ 5 levels "children","Govt_job",..: 4 5 4 4 5 4 4 4 4 4 ...
    ##  $ Residence_type   : Factor w/ 2 levels "Rural","Urban": 2 1 1 2 1 2 1 2 1 2 ...
    ##  $ avg_glucose_level: num  229 202 106 171 174 ...
    ##  $ bmi              : num  36.6 NA 32.5 34.4 24 29 27.4 22.8 NA 24.2 ...
    ##  $ smoking_status   : Factor w/ 4 levels "formerly smoked",..: 1 2 2 3 2 1 2 2 4 4 ...
    ##  $ stroke           : Factor w/ 2 levels "0","1": 2 2 2 2 2 2 2 2 2 2 ...

    1. The attributes with less than 6 unique values are turned into factor type attribute
    2. The bmi attribute turned into numeric from character

``` r
summary(strokeData)
```

    ##        id           gender          age        hypertension heart_disease
    ##  Min.   :   67   Female:2994   Min.   : 0.08   0:4612       0:4834       
    ##  1st Qu.:17741   Male  :2115   1st Qu.:25.00   1: 498       1: 276       
    ##  Median :36932   Other :   1   Median :45.00                             
    ##  Mean   :36518                 Mean   :43.23                             
    ##  3rd Qu.:54682                 3rd Qu.:61.00                             
    ##  Max.   :72940                 Max.   :82.00                             
    ##                                                                          
    ##  ever_married         work_type    Residence_type avg_glucose_level
    ##  No :1757     children     : 687   Rural:2514     Min.   : 55.12   
    ##  Yes:3353     Govt_job     : 657   Urban:2596     1st Qu.: 77.25   
    ##               Never_worked :  22                  Median : 91.89   
    ##               Private      :2925                  Mean   :106.15   
    ##               Self-employed: 819                  3rd Qu.:114.09   
    ##                                                   Max.   :271.74   
    ##                                                                    
    ##       bmi                smoking_status stroke  
    ##  Min.   :10.30   formerly smoked: 885   0:4861  
    ##  1st Qu.:23.50   never smoked   :1892   1: 249  
    ##  Median :28.10   smokes         : 789           
    ##  Mean   :28.89   Unknown        :1544           
    ##  3rd Qu.:33.10                                  
    ##  Max.   :97.60                                  
    ##  NA's   :201

    1. There are 1 "Other" level in a gender attribute. I think it's not suitable with the rest of the levels, it also only has 1 values which are not impactful, so i decided to drop "Other" level in the next step
    2. There are more female than male in this dataset
    3. The oldest people in dataset is 82 y.o and the youngest is 0.08 y.o which indicates the age range in the dataset is from baby to elderly
    4. The highest BMI could be found in the dataset is 97.6 which definitely very high as it shows a big gap from the Q3
    5. After turning BMI attribute from chr to num, we can found there are 201 missing values which is not shown before when in chr type
    6. There are 498 people from the total of 5110 people in dataset who has a hypertension
    7. There are more people with heart disease(276) than people with stroke(249) in this dataset

``` r
# Remove "Other" level in gender attribute
strokeData <- strokeData %>% filter(gender != 'Other') %>% droplevels()

levels(strokeData$gender)
```

    ## [1] "Female" "Male"

    Dropping "Other" levels in gender because it's not suitable with the rest of the levels, it also only has 1 value which are not impactful

\##Checking Anomalies

``` r
qqPlot(strokeData$bmi)
```

![](Stroke-Prediction_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

    ## [1] 2129 4209

``` r
ggplot(strokeData, aes(x = 1, y = bmi)) + 
  geom_boxplot() +
  coord_flip()
```

    ## Warning: Removed 201 rows containing non-finite values (`stat_boxplot()`).

![](Stroke-Prediction_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

    Even though from the summary we can see that the max value of bmi attribute is 97.6 and it looks like an anomaly in the dataset. But from the qqPlot and boxplot, we can see that there are pretty much bmi value that lies outside the 3rd Quartile, so we can say that the values outside the 3rd Quartile is not an anomalies.

\##Univariate Analysis

``` r
bmi_cat <- cut(strokeData$bmi, breaks=c(0, 18.5, 25, 30, 35, 40, Inf), labels=c("UnderWeight", "Normal", "OverWeight", "ObesityClassI", "ObesityClassII", "ObesityClassIII"))

levels(bmi_cat)
```

    ## [1] "UnderWeight"     "Normal"          "OverWeight"      "ObesityClassI"  
    ## [5] "ObesityClassII"  "ObesityClassIII"

``` r
glucose_cat <- cut(strokeData$avg_glucose_level, breaks=c(0, 100, 126, Inf), labels=c("Normal", "Prediabetes", "Diabetes"))
levels(glucose_cat)
```

    ## [1] "Normal"      "Prediabetes" "Diabetes"

    - Make a new variable(bmi_cat and glucose_cat) to contain BMI and glucose level in factor form for analysis
    - The classifications are based on WHO classifications on BMI and glucose level

``` r
#BMI
plot(bmi_cat, 
     xlab="BMI Classifications", 
     ylab="Record number in dataset", 
     main="BMI Classifications",
     cex.names=0.7,
     cex.axis=0.8,
     col="orange")
```

![](Stroke-Prediction_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

    - According to WHO BMI classifications, we can see that Over Weight class is the most frequent and Under Weight is the least frequent found in the dataset - There are also more than 400 people that have Obesity either Class I, Class II, and even Class III.

``` r
#Average Glucose Level
plot(glucose_cat, 
     xlab="Glucose Levels", 
     ylab="Record number in dataset", 
     main="Average Glucose Level",
     las=1,
     col="Red")
```

![](Stroke-Prediction_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

    - Most poeple in the dataset has a normal average glucose level according to WHO Classifications
    - There are pretty equal amount of people with prediabetes and diabetes type of glucose level

\##Bivariate Analysis

``` r
table(strokeData$smoking_status, strokeData$gender)
```

    ##                  
    ##                   Female Male
    ##   formerly smoked    477  407
    ##   never smoked      1229  663
    ##   smokes             452  337
    ##   Unknown            836  708

``` r
#barchart of gender by smoking status
ggplot(strokeData, aes(x = smoking_status, fill = gender)) + 
  geom_bar(position = "dodge")
```

![](Stroke-Prediction_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

    From the barplot, we can see that the dataset is dominated by female who never smoked(1229). We also can see that there are few male who smokes(337) in the dataset.

``` r
table(strokeData$gender, strokeData$stroke)
```

    ##         
    ##             0    1
    ##   Female 2853  141
    ##   Male   2007  108

``` r
#barchart of stroke by gender
ggplot(strokeData, aes(x = stroke, fill = gender)) + 
  geom_bar(position = "dodge") + scale_x_discrete(labels=c("0" = "Doesn't Have Stroke", "1" = "Have Stroke"))
```

![](Stroke-Prediction_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

    From the barplot, we can see that female who doesn't have stroke is the most frequently found and male who have stroke is the least frequent in the dataset.

\#Data Preparation

\##Cleaning

``` r
strokeData <- subset(strokeData, select = c(2,3,4,5,6,7,8,9,10,11,12))
```

    Ignore "id" column as it has various values(all are distinct) and not significant for modeling

``` r
sapply(strokeData, function(x) sum(is.na(x)))
```

    ##            gender               age      hypertension     heart_disease 
    ##                 0                 0                 0                 0 
    ##      ever_married         work_type    Residence_type avg_glucose_level 
    ##                 0                 0                 0                 0 
    ##               bmi    smoking_status            stroke 
    ##               201                 0                 0

    In the form of numeric attribute, we can found 201 missing values in bmi attribute.

``` r
#Drop missing value rows
strokeData$bmi[is.na(strokeData$bmi)] <- mean(strokeData$bmi, na.rm = T)
sapply(strokeData, function(x) sum(is.na(x)))
```

    ##            gender               age      hypertension     heart_disease 
    ##                 0                 0                 0                 0 
    ##      ever_married         work_type    Residence_type avg_glucose_level 
    ##                 0                 0                 0                 0 
    ##               bmi    smoking_status            stroke 
    ##                 0                 0                 0

    Reasons to drop missing value rows:
    - The age of people in dateset range from baby to elderly, i think we can't replace the bmi missing values with mean or median of the bmi because it will ruin the prediction
    - This dataset has 5110 records and the missing values are 201. The proportion of missing value from total of data is only 3.9%, so i think dropping missing values won't affect much

\##Correlations

``` r
temp <- strokeData[, c(2,3,4,8,9,11)]
rcorr(as.matrix(temp), type="pearson")
```

    ##                    age hypertension heart_disease avg_glucose_level  bmi stroke
    ## age               1.00         0.28          0.26              0.24 0.33   0.25
    ## hypertension      0.28         1.00          0.11              0.17 0.16   0.13
    ## heart_disease     0.26         0.11          1.00              0.16 0.04   0.13
    ## avg_glucose_level 0.24         0.17          0.16              1.00 0.17   0.13
    ## bmi               0.33         0.16          0.04              0.17 1.00   0.04
    ## stroke            0.25         0.13          0.13              0.13 0.04   1.00
    ## 
    ## n= 5109 
    ## 
    ## 
    ## P
    ##                   age    hypertension heart_disease avg_glucose_level bmi   
    ## age                      0.0000       0.0000        0.0000            0.0000
    ## hypertension      0.0000              0.0000        0.0000            0.0000
    ## heart_disease     0.0000 0.0000                     0.0000            0.0055
    ## avg_glucose_level 0.0000 0.0000       0.0000                          0.0000
    ## bmi               0.0000 0.0000       0.0055        0.0000                  
    ## stroke            0.0000 0.0000       0.0000        0.0000            0.0054
    ##                   stroke
    ## age               0.0000
    ## hypertension      0.0000
    ## heart_disease     0.0000
    ## avg_glucose_level 0.0000
    ## bmi               0.0054
    ## stroke

    1. The P-value between the target (stroke) and the independent variable (bmi) are > 0.05, it means the correlation between them are not really significant.

    2. The correlation coefficient between the target (stroke) and the independent variables are a weak correlations as all of them are under 0.4.

    3. From all of the correlation coefficient, the age variable give the highest correlation coefficient number among the other independent variables which means highest correlation to the stroke variable

    4. So, i think the most important variable that affects the stroke variable is age.

\#Predictive Modeling

\##Split Data into Training and Validation

``` r
set.seed(1)
val_idx = createDataPartition(strokeData$stroke, p=0.8, list=FALSE)
train_set = strokeData[val_idx,]
val_set = strokeData[-val_idx,]
```

\##Logistic Modeling

``` r
logisticModel <- glm(stroke ~., family =  binomial(link = "logit"), data = train_set)

summary(logisticModel)
```

    ## 
    ## Call:
    ## glm(formula = stroke ~ ., family = binomial(link = "logit"), 
    ##     data = train_set)
    ## 
    ## Coefficients:
    ##                              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                 -6.619891   0.805190  -8.222  < 2e-16 ***
    ## genderMale                   0.096310   0.157329   0.612  0.54043    
    ## age                          0.074001   0.006583  11.242  < 2e-16 ***
    ## hypertension1                0.371163   0.184865   2.008  0.04467 *  
    ## heart_disease1               0.293962   0.212531   1.383  0.16662    
    ## ever_marriedYes             -0.077502   0.264670  -0.293  0.76965    
    ## work_typeGovt_job           -1.159865   0.866577  -1.338  0.18075    
    ## work_typeNever_worked      -10.538046 362.527978  -0.029  0.97681    
    ## work_typePrivate            -1.174719   0.849786  -1.382  0.16686    
    ## work_typeSelf-employed      -1.504795   0.874131  -1.721  0.08516 .  
    ## Residence_typeUrban          0.019993   0.153828   0.130  0.89659    
    ## avg_glucose_level            0.003648   0.001349   2.704  0.00685 ** 
    ## bmi                          0.008391   0.012555   0.668  0.50391    
    ## smoking_statusnever smoked  -0.284153   0.195270  -1.455  0.14562    
    ## smoking_statussmokes         0.137925   0.234163   0.589  0.55585    
    ## smoking_statusUnknown       -0.166907   0.233159  -0.716  0.47408    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1597.0  on 4087  degrees of freedom
    ## Residual deviance: 1277.9  on 4072  degrees of freedom
    ## AIC: 1309.9
    ## 
    ## Number of Fisher Scoring iterations: 14

    from the logisticModel summary, we can see only few attributes that are significant by looking at the p value, which are "age", "hypertension", and "avg_glucose_level". Therefore i decided to exclude other attributes from the model. Then i fit the data again with "age", "hypertension", and "avg_glucose_level".

``` r
logisticModel1 <- glm(stroke ~ age + avg_glucose_level + hypertension, family =  binomial(link = "logit"), data = train_set)

summary(logisticModel1)
```

    ## 
    ## Call:
    ## glm(formula = stroke ~ age + avg_glucose_level + hypertension, 
    ##     family = binomial(link = "logit"), data = train_set)
    ## 
    ## Coefficients:
    ##                    Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)       -7.422553   0.389525 -19.055   <2e-16 ***
    ## age                0.068561   0.005556  12.340   <2e-16 ***
    ## avg_glucose_level  0.004208   0.001300   3.238   0.0012 ** 
    ## hypertension1      0.359721   0.182098   1.975   0.0482 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1597  on 4087  degrees of freedom
    ## Residual deviance: 1291  on 4084  degrees of freedom
    ## AIC: 1299
    ## 
    ## Number of Fisher Scoring iterations: 7

    - From the second logistic model (logisticModel1) that uses attribute age, average glucose level and hypertension has an improvement from the first model. The AIC value we got from second model is 1299 which is lower than AIC value in the first model (1309.9). The lower the AIC value means the better the model that uses the sama dataset.

    - So, i decided to choose the second modeling, which gives the equation:
      y = -7.422553 + 0.068561(age) + 0.004208(average glucose level) + 0.359721(hypertension)

\##Evaluation

``` r
predictionLogistic <- predict(logisticModel1, val_set, type = "response")
evaluation <- prediction(predictionLogistic, val_set$stroke)
prf <- performance(evaluation, measure = "tpr", x.measure = "fpr")
plot(prf)
```

![](Stroke-Prediction_files/figure-gfm/unnamed-chunk-26-1.png)<!-- -->

    - From the ROC Curve shown above, we can see the curve is closer to the True Positive Rate which is a good thing, because it means the test is more accurate

\##Check AUC Score

``` r
auc <- performance(evaluation, measure = "auc")
auc <- auc@y.values[[1]]
auc
```

    ## [1] 0.8590745

    - From the ROC Curve, we also can get the AUC or Area under Curve. Because the better the curve to the top left spot, so the bigger the AUC also means better. The logistic model has AUC 0.86 which is a high number and it means the better the performance of the logistic model at distinguishing positive and negative classes.

\#Decision Tree Model

``` r
DTmodel <- rpart(stroke ~., data = train_set, method = "class", cp = 0.001)
DTmodel
```

    ## n= 4088 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##    1) root 4088 200 0 (0.95107632 0.04892368)  
    ##      2) age< 56.5 2788  32 0 (0.98852224 0.01147776) *
    ##      3) age>=56.5 1300 168 0 (0.87076923 0.12923077)  
    ##        6) age< 73.5 850  83 0 (0.90235294 0.09764706)  
    ##         12) avg_glucose_level< 189.83 669  51 0 (0.92376682 0.07623318)  
    ##           24) smoking_status=never smoked 241   9 0 (0.96265560 0.03734440) *
    ##           25) smoking_status=formerly smoked,smokes,Unknown 428  42 0 (0.90186916 0.09813084)  
    ##             50) bmi>=28.89728 221  15 0 (0.93212670 0.06787330) *
    ##             51) bmi< 28.89728 207  27 0 (0.86956522 0.13043478)  
    ##              102) bmi< 26.85 110   6 0 (0.94545455 0.05454545) *
    ##              103) bmi>=26.85 97  21 0 (0.78350515 0.21649485)  
    ##                206) avg_glucose_level< 67.17 18   1 0 (0.94444444 0.05555556) *
    ##                207) avg_glucose_level>=67.17 79  20 0 (0.74683544 0.25316456)  
    ##                  414) avg_glucose_level>=82.34 58  10 0 (0.82758621 0.17241379) *
    ##                  415) avg_glucose_level< 82.34 21  10 0 (0.52380952 0.47619048)  
    ##                    830) Residence_type=Rural 8   2 0 (0.75000000 0.25000000) *
    ##                    831) Residence_type=Urban 13   5 1 (0.38461538 0.61538462) *
    ##         13) avg_glucose_level>=189.83 181  32 0 (0.82320442 0.17679558)  
    ##           26) heart_disease=0 143  20 0 (0.86013986 0.13986014)  
    ##             52) bmi< 44.05 132  16 0 (0.87878788 0.12121212)  
    ##              104) avg_glucose_level>=197.615 111  10 0 (0.90990991 0.09009009) *
    ##              105) avg_glucose_level< 197.615 21   6 0 (0.71428571 0.28571429)  
    ##                210) age< 69 14   2 0 (0.85714286 0.14285714) *
    ##                211) age>=69 7   3 1 (0.42857143 0.57142857) *
    ##             53) bmi>=44.05 11   4 0 (0.63636364 0.36363636) *
    ##           27) heart_disease=1 38  12 0 (0.68421053 0.31578947)  
    ##             54) gender=Female 13   1 0 (0.92307692 0.07692308) *
    ##             55) gender=Male 25  11 0 (0.56000000 0.44000000)  
    ##              110) bmi< 30.8 13   4 0 (0.69230769 0.30769231) *
    ##              111) bmi>=30.8 12   5 1 (0.41666667 0.58333333) *
    ##        7) age>=73.5 450  85 0 (0.81111111 0.18888889)  
    ##         14) bmi>=34.15 56   3 0 (0.94642857 0.05357143) *
    ##         15) bmi< 34.15 394  82 0 (0.79187817 0.20812183)  
    ##           30) bmi< 28.84728 238  40 0 (0.83193277 0.16806723)  
    ##             60) work_type=Govt_job,Self-employed 126  15 0 (0.88095238 0.11904762) *
    ##             61) work_type=Private 112  25 0 (0.77678571 0.22321429)  
    ##              122) avg_glucose_level< 109.015 81  14 0 (0.82716049 0.17283951) *
    ##              123) avg_glucose_level>=109.015 31  11 0 (0.64516129 0.35483871)  
    ##                246) avg_glucose_level>=131.835 24   6 0 (0.75000000 0.25000000) *
    ##                247) avg_glucose_level< 131.835 7   2 1 (0.28571429 0.71428571) *
    ##           31) bmi>=28.84728 156  42 0 (0.73076923 0.26923077)  
    ##             62) bmi>=28.89728 126  28 0 (0.77777778 0.22222222)  
    ##              124) smoking_status=Unknown 27   2 0 (0.92592593 0.07407407) *
    ##              125) smoking_status=formerly smoked,never smoked,smokes 99  26 0 (0.73737374 0.26262626)  
    ##                250) avg_glucose_level< 72.185 8   0 0 (1.00000000 0.00000000) *
    ##                251) avg_glucose_level>=72.185 91  26 0 (0.71428571 0.28571429)  
    ##                  502) bmi< 31.25 50  11 0 (0.78000000 0.22000000) *
    ##                  503) bmi>=31.25 41  15 0 (0.63414634 0.36585366)  
    ##                   1006) gender=Female 27   6 0 (0.77777778 0.22222222) *
    ##                   1007) gender=Male 14   5 1 (0.35714286 0.64285714) *
    ##             63) bmi< 28.89728 30  14 0 (0.53333333 0.46666667)  
    ##              126) smoking_status=never smoked,smokes,Unknown 22   7 0 (0.68181818 0.31818182) *
    ##              127) smoking_status=formerly smoked 8   1 1 (0.12500000 0.87500000) *

``` r
rpart.plot(DTmodel)
```

![](Stroke-Prediction_files/figure-gfm/unnamed-chunk-29-1.png)<!-- -->

    What can we interpret from the decision tree model:
    - People under 57 in dataset definitely don't have a stroke
    - Age is the most important attribute as it is in the top, followed by average glucose level and bmi

\##Variable Importance

``` r
DTmodel$variable.importance
```

    ##               age avg_glucose_level               bmi    smoking_status 
    ##       32.37997459       18.91330183       15.38384281        7.61662451 
    ##            gender     heart_disease         work_type      hypertension 
    ##        5.86205170        4.75817570        4.16054225        2.70178319 
    ##    Residence_type      ever_married 
    ##        1.50375575        0.02701991

    The most important variable is determined by the bigger the number from result above. The more important the variable means the more the variable needed by the model to make prediction accurate. So from the result above the most important variable is age. The second and third most important variable are average glucose level and bmi. From here we can see logistic model and decision tree gave slightly different important variables for the prediction.

\##Confusion Matrix

``` r
predictionDT <- predict(DTmodel, val_set, type = "class")
cf <- table(predictionDT, val_set$stroke)
```

\##Overall Classification

``` r
#overall correct classification
sum(diag(cf))
```

    ## [1] 966

``` r
#incorrect classification
1 - sum(diag(cf)) 
```

    ## [1] -965

    The overall correct classification from the confusion matrix from prediction and validation set is 966 and the overall incorrect classification is -965

\##Accuracy

``` r
#Accuracy per label
acc <- diag(cf) / rowSums(cf) * 100
acc
```

    ##       0       1 
    ## 95.2616 12.5000

    - The accuracy for predicting people who don't have a stroke is 95.3%
    - The accuracy for predicting people who have a stroke is 12.5%

\##Overall Accuracy

``` r
overallAcc <- sum(diag(cf)) / sum(cf) * 100 
overallAcc
```

    ## [1] 94.61312

    The overall accuracy of prediction using Decision Tree Model is 94.6% which is a very high accuracy
