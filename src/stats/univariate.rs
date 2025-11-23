// src/stats/univariate.rs

/// Calculates the sum of elements.
pub fn sum(data: &[f64]) -> f64 {
    data.iter().sum()
}

/// Calculates the arithmetic mean.
pub fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() { return None; }
    Some(sum(data) / data.len() as f64)
}

/// Calculates the median.
pub fn median(data: &[f64]) -> Option<f64> {
    if data.is_empty() { return None; }
    let mut sorted = data.to_vec();
    // Handle NaNs by defaulting to Equal (safe-ish for stats)
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some((sorted[mid - 1] + sorted[mid]) / 2.0)
    } else {
        Some(sorted[mid])
    }
}

/// Sample variance (Welford's algorithm).
pub fn sample_variance(data: &[f64]) -> Option<f64> {
    if data.len() < 2 { return None; }
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (i, &x) in data.iter().enumerate() {
        let count = (i + 1) as f64;
        let delta = x - mean;
        mean += delta / count;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    Some(m2 / (data.len() as f64 - 1.0))
}