
## Input data

M5 is the main input data. After preprocessing the raw files and turn it into long format
Core features:
- sales history
- item_id
- store_id
- date



| Column         | Type     | Description                         |
| -------------- | -------- | ----------------------------------- |
| `id`           | string   | Unique item-store series identifier |
| `item_id`      | string   | Product identifier                  |
| `dept_id`      | string   | Department identifier               |
| `cat_id`       | string   | Category identifier                 |
| `store_id`     | string   | Store identifier                    |
| `state_id`     | string   | State identifier                    |
| `d`            | string   | M5 day identifier                   |
| `sales`        | numeric  | Target demand value                 |
| `date`         | datetime | Calendar date                       |
| `wm_yr_wk`     | integer  | Walmart week year                   |
| `weekday`      | string   | Weekday name                        |
| `wday`         | integer  | Weekday number                      |
| `month`        | integer  | Month number                        |
| `year`         | integer  | Calendar year                       |
| `event_name_1` | string   | Primary event name                  |
| `event_type_1` | string   | Primary event type                  |
| `event_name_2` | string   | Secondary event name                |
| `event_type_2` | string   | Secondary event type                |
| `snap_CA`      | integer  | SNAP flag for California            |
| `snap_TX`      | integer  | SNAP flag for Texas                 |
| `snap_WI`      | integer  | SNAP flag for Wisconsin             |
| `sell_price`   | numeric  | Item-store weekly price             |
| `price_missing_flag`   | integer  | Missing   price             |


**SNAP stands for the Supplemental Nutrition Assistance Program

SNAP is a US federal program that provides nutritional assistance to low-income individuals, usually via an Electronic Benefits Transfer (EBT) card.



Covariates:
- sell_price
- calendar features
- event indicators
- SNAP indicators


| Column             | Type    | Description                                        |
| ------------------ | ------- | -------------------------------------------------- |
| `day_of_week`      | integer | Day of week.Monday=`0` and Sunday=`6`              |
| `is_weekend`       | integer | Flag if weekend                                    |
| `time_idx`         | integer | Time step on each series, from `0` for each `id`   |
| `event_flag`       | integer | Flag of either `event_name_1` or `event_name_2`    |
| `snap_flag`        | integer | State-specific SNAP flag                           |
| `price_change_pct` | numeric | % change in `sell_price` row against previous time |
| `zero_demand_flag` | integer | Flag when `sales` = `0`                            |
| `log_sales`        | numeric | Log sales using `log(1 + sales)`                   |
| `lag_1`            | numeric | Sales from 1 day before in the same series         |
| `lag_7`            | numeric | Sales from 7 days before in the same series        |
| `lag_28`           | numeric | Sales from 28 days before in the same series       |
| `item_idx`         | integer | Ordered encoded integer ID for `item_id`           |
| `dept_idx`         | integer | Ordered encoded integer ID for `dept_id`           |
| `cat_idx`          | integer | Ordered encoded integer ID for `cat_id`            |
| `store_idx`        | integer | Ordered encoded integer ID for `store_id`          |
| `state_idx`        | integer | Ordered encoded integer ID for `state_id`          |
| `series_idx`       | integer | Ordered encoded integer ID for  `id`               |