///* ~src/linalg/mod.rs
/// Library for implementation of statistics routines in Rust

/// ## Design Philosophy
/// 
/// The matrix is stored in **row-major order** (all elements of the first row, followed 
/// by all elements of the second row, etc.).
/// 
/// This design was chosen over a `Vec<Vec<f64>>` (vector of vectors) approach for 
/// **performance and memory safety** reasons:
/// 
/// 1.  **Contiguous Memory:** A single `Vec<f64>` is guaranteed to be allocated as a 
///     **contiguous block of memory** on the heap. This maximizes **cache locality**, making 
///     sequential operations (like addition or iteration) significantly faster. On the other
///     hand, a vector of vectors, `Vec<Vec<f64>>` may be stored may be stored in scattered 
///     memory locations, forcing the CPU to perform pointer chasing (jumping to different 
///     heap addresses), which breaks cache efficiency.
/// 2.  **Predictable Indexing:** Accessing an element at $(r, c)$ uses the formula: 
///     `index = r * cols + c`.
/// 3.  **No Jagged Arrays:** It prevents the possibility of "jagged" or non-uniform rows, 
///     ensuring the matrix shape is always rectangular.
/// 
/// ## ⚠️ Safety Note
/// 
/// Although the dimensions (`rows` and `cols`) are redundant with `data.len()`, all three 
/// fields are stored explicitly for **robust error checking** during construction and 
/// clarity during indexing.
/// 
/// The data type is currently fixed at `f64` for maximum numerical precision in statistical 
/// calculations.
/// 
/// 
/// 
/// 
use std::{
    fmt,                // For Display trait
    ops::{Add, Index},  // For '+' operator and indexing []
};
use crate::error::StatsError;  // Custom error handling

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Creates a new Matrix from a flattened vector of data.
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Option<Self> {
        if data.len() != rows * cols {
            return None;
        }
        Some(Matrix { rows, cols, data })
    }

    /// Zeros matrix of size rows x cols
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    // <-- TRANSPOSE MUST BE HERE!
    /// Calculates the transpose of the matrix (M^T).
    pub fn transpose(&self) -> Self {
        // 1. Create the new matrix with dimensions swapped (cols x rows)
        let mut transposed = Matrix::zeros(self.cols, self.rows);

        // 2. Iterate over the original indices (r, c)
        for r in 0..self.rows {
            for c in 0..self.cols {
                // Move the value from (r, c) to (c, r)
                transposed.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        transposed
    }
}

// -----------------------------------------------------------
// 3. THE DISPLAY TRAIT (Impl for external traits go outside the main impl)
// -----------------------------------------------------------
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..self.rows {
            for c in 0..self.cols {
                let idx = r * self.cols + c;
                write!(f, "{:.4}\t", self.data[idx])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}


///Simple sum function
///TODO: Implement Kahan summation algorithm
///https://en.wikipedia.org/wiki/Kahan_summation_algorithm

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


/// Sample variance - using Welford's online algorithm
/// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
pub fn sample_variance(data: &[f64]) -> Option<f64> {
    if data.len() < 2 {
        return None;
    }

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

// -----------------------------------------------------------
// 4. INDEXING TRAIT (Allows matrix[r, c])
// -----------------------------------------------------------
impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (r, c) = index;
        if r >= self.rows || c >= self.cols {
            panic!("Index out of bounds: ({}, {}) for matrix size {}x{}", r, c, self.rows, self.cols);
        }
        &self.data[r * self.cols + c]
    }
}

// -----------------------------------------------------------
// 5. ADDITION TRAIT (Allows matrix_a + matrix_b)
// -----------------------------------------------------------
impl Add for Matrix {
    type Output = Result<Matrix, StatsError>;

    fn add(self, other: Self) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(StatsError::DimensionMismatch {
                expected: format!("{}x{}", self.rows, self.cols),
                actual: format!("{}x{}", other.rows, other.cols),
            });
        }

        let result_data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Matrix::new(result_data, self.rows, self.cols).unwrap())
    }
}

// -----------------------------------------------------------
// 6. MULTIPLICATION IMPLEMENTATION (Dot Product)
// -----------------------------------------------------------
impl Matrix {
    // We extend the Matrix impl here for organization
    pub fn multiply(&self, other: &Matrix) -> Result<Self, StatsError> {
        if self.cols != other.rows {
            return Err(StatsError::DimensionMismatch {
                expected: format!("A.cols ({}) == B.rows ({})", self.cols, self.rows),
                actual: format!("{}x{} and {}x{}", self.rows, self.cols, other.rows, other.cols),
            });
        }

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let mut result = Matrix::zeros(m, n);

        for r in 0..m {
            for c in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    // This uses the Index trait we just defined above!
                    sum += self[(r, p)] * other[(p, c)]; 
                }
                result.data[r * n + c] = sum;
            }
        }
        Ok(result)
    }
}