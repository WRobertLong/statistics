// src/linalg/mod.rs

// Simple sum function
// TODO: Implement Kahan summation algorithm
pub fn sum(data: &[f64]) -> f64 {
    let mut total = 0.0;
    for &x in data { // The '&' here dereferences the borrowed value so we get a raw f64
        total += x;
    }
    total
}



/// Calculates the arithmetic mean.
/// Returns None if the vector is empty.
pub fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    
    let sum: f64 = data.iter().sum();
    Some(sum / data.len() as f64)
}

/// Calculates the median.
/// Note: This requires sorting, so we must clone the data to avoid modifying the original.
pub fn median(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    // 1. Clone the data so we can sort it (f64 doesn't implement Ord, so we need a workaround)
    let mut sorted_data = data.to_vec();
    
    // Rust f64 includes NaN (Not a Number), which cannot be compared.
    // We use partial_cmp and unwrap_or to handle potential NaNs safely-ish.
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted_data.len();
    let mid = n / 2;

    if n % 2 == 0 {
        // Even number of elements: average the two middle ones
        Some((sorted_data[mid - 1] + sorted_data[mid]) / 2.0)
    } else {
        // Odd number of elements
        Some(sorted_data[mid])
    }
}
