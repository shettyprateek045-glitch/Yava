# EDA Summary Report — Streaming Service Churn Dataset

## Data Quality Summary

- Raw rows: 5010
- Duplicate rows removed: 10
- Columns with missing values: {'age': np.int64(234), 'payment_method': np.int64(216), 'monthly_charge': np.int64(228), 'avg_watch_hours_week': np.int64(221)}
- Outliers capped in monthly_charge (IQR method): 26
- Final clean rows: 5000

## Univariate Analysis

- Saved: 01_univariate_histograms.png
- Saved: 02_churn_class_balance.png

Summary statistics:

|       |     age |   tenure_months |   monthly_charge |   avg_watch_hours_week |   num_devices |   support_tickets |   days_since_last_login |
|:------|--------:|----------------:|-----------------:|-----------------------:|--------------:|------------------:|------------------------:|
| count | 5000    |         5000    |          5000    |                5000    |       5000    |           5000    |                 5000    |
| mean  |   34.82 |           23.49 |            13.53 |                   6.56 |          1.91 |              0.61 |                    9.42 |
| std   |   11.31 |           16.08 |             3.78 |                   4.45 |          0.93 |              0.77 |                    9.97 |
| min   |   13    |            0    |             4.99 |                   0.09 |          1    |              0    |                    0    |
| 25%   |   27    |           11    |            11    |                   3.35 |          1    |              0    |                    2    |
| 50%   |   35    |           20    |            13.54 |                   5.59 |          2    |              0    |                    6    |
| 75%   |   42    |           32    |            16    |                   8.74 |          3    |              1    |                   13    |
| max   |   79    |           72    |            23.5  |                  40    |          4    |              5    |                   97    |

## Bivariate Analysis

- Saved: 03_bivariate_boxplots.png
- Saved: 04_churn_by_plan_type.png
## Multivariate Analysis

- Saved: 05_correlation_heatmap.png
- Saved: 06_pairplot.png

Strongest correlations with `churned`:

|                       |   churned |
|:----------------------|----------:|
| days_since_last_login |     0.14  |
| tenure_months         |    -0.125 |
| avg_watch_hours_week  |    -0.078 |
| support_tickets       |     0.074 |
| num_devices           |     0.019 |
| age                   |     0.009 |
| monthly_charge        |    -0.001 |
