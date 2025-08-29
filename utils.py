def weighted_median(data, weights):
    if len(data) == 0:
        raise ValueError("Forecast data is empty or does not exist.")
    
    weights = weights[:len(data)]
    total_weight = sum(weights)
    cumul_weight = 0
    half_weight = total_weight / 2
    
    for i in range(len(data)):
        cumul_weight += weights[i]
        if cumul_weight >= half_weight:
            return data[i]
