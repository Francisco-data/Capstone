# Research question

 `Can a unified global forecasting model improve performance on sparse and cold-start retail demand compared with standard global forecasting baselines? `
 
The project evaluates if a single global architecture can address 3 forecasting challenges that are usually treated separately:
 * intermittent demand, 
 * cold-start products,
 * multi-buyer retail forecasting.

The cold-start component will first be implemented using metadata-based embeddings. 

A similarity-based fallback or ensemble approach may be explored if time permits.



## Sub-questions

1. **Intermittent demand** with many zero observations.
2. **Cold-start items** with little or no historical sales.
3. **Cold-start buyers or stores** with limited historical data.

### 1. Intermittent demand
 `Does a zero-inflated or a two-part output head improve performance on sparse demand seriescompared with a standard DLinear-style global model? `

### 2. Cold-start items
 `Can metadata-based embeddings improve forecasting for items with limited or no historical sales? `

### 3. Cold-start stores / buyers
 `Can a global model generalize to an unseen store using store and iteml hierarchy features? `

### 4. Unified performance
 `Does the combined model perform better than its individual components? `