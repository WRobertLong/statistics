//! Library for implementation of linear algebra routines.
//!
//! ## Design Philosophy
//! 
//! The matrix is stored in **row-major order** (all elements of the first row, followed 
//! by all elements of the second row, etc.).
//! 
//! This design was chosen over a `Vec<Vec<f64>>` (vector of vectors) approach for 
//! **performance and memory safety** reasons:
//! 
//! 1.  **Contiguous Memory:** A single `Vec<f64>` is guaranteed to be allocated as a 
//!     **contiguous block of memory** on the heap. This maximizes **cache locality**, making 
//!     sequential operations (like addition or iteration) significantly faster. On the other
//!     hand, a vector of vectors `Vec<Vec<f64>>` may be stored in scattered 
//!     memory locations, forcing the CPU to perform pointer chasing (jumping to different 
//!     heap addresses), which breaks cache efficiency.
//! 2.  **Predictable Indexing:** Accessing an element at $(r, c)$ uses the formula: 
//!     `index = r * cols + c`.
//! 3.  **No Jagged Arrays:** It prevents the possibility of "jagged" or non-uniform rows, 
//!     ensuring the matrix shape is always rectangular.
//! 
//! ## ⚠️ Safety Note
//! 
//! Although the dimensions (`rows` and `cols`) are redundant with `data.len()`, all three 
//! fields are stored explicitly for **robust error checking** during construction and 
//! clarity during indexing.
//! 
//! The data type is currently fixed at `f64` for maximum numerical precision in statistical 
//! calculations.
//! 

// PURE MATH ONLY: Matrix definitions and operations

use std::{
    fmt,                
    ops::{Add, Index},  
};
use crate::error::StatsError;  

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

// ========================================================
// 1. CORE IMPLEMENTATION
// ========================================================
impl Matrix {
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Option<Self> {
        if data.len() != rows * cols {
            return None;
        }
        Some(Matrix { rows, cols, data })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut mat = Matrix::zeros(n, n);
        for i in 0..n {
            mat.data[i * n + i] = 1.0;
        }
        mat
    }

    pub fn transpose(&self) -> Self {
        let mut transposed = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                transposed.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        transposed
    }

    pub fn inverse(&self) -> Result<Self, StatsError> {
        if self.rows != self.cols {
            return Err(StatsError::NotSquare(self.rows, self.cols));
        }
        let n = self.rows;
        let mut result = Matrix::identity(n);
        let mut m = self.clone();

        for i in 0..n {
            let mut pivot_row = i;
            let mut max_val = m[(i, i)].abs();
            for r in (i + 1)..n {
                if m[(r, i)].abs() > max_val {
                    max_val = m[(r, i)].abs();
                    pivot_row = r;
                }
            }
            if max_val < 1e-10 {
                return Err(StatsError::SingularMatrix);
            }
            if pivot_row != i {
                m.swap_rows(i, pivot_row);
                result.swap_rows(i, pivot_row);
            }
            let pivot = m[(i, i)];
            for c in 0..n {
                m.data[i * n + c] /= pivot;
                result.data[i * n + c] /= pivot;
            }
            for r in 0..n {
                if r != i {
                    let factor = m[(r, i)];
                    for c in 0..n {
                        let val_m = m[(i, c)]; 
                        m.data[r * n + c] -= factor * val_m;
                        let val_res = result[(i, c)];
                        result.data[r * n + c] -= factor * val_res;
                    }
                }
            }
        }
        Ok(result)
    }

    fn swap_rows(&mut self, r1: usize, r2: usize) {
        for c in 0..self.cols {
            let idx1 = r1 * self.cols + c;
            let idx2 = r2 * self.cols + c;
            self.data.swap(idx1, idx2);
        }
    }

    pub fn diagonal(&self) -> Vec<f64> {
        let n = std::cmp::min(self.rows, self.cols);
        let mut diag = Vec::with_capacity(n);
        for i in 0..n {
            diag.push(self[(i, i)]);
        }
        diag
    }

    pub fn subtract(&self, other: &Matrix) -> Result<Matrix, StatsError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(StatsError::DimensionMismatch {
                expected: format!("{}x{}", self.rows, self.cols),
                actual: format!("{}x{}", other.rows, other.cols),
            });
        }
        let new_data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        Ok(Matrix::new(new_data, self.rows, self.cols).unwrap())
    }

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
                    sum += self[(r, p)] * other[(p, c)]; 
                }
                result.data[r * n + c] = sum;
            }
        }
        Ok(result)
    }

    // --- NEW HELPERS (Cleaned up) ---

    /// Multiplies every element in the matrix by a scalar.
    pub fn scale(&self, scalar: f64) -> Self {
        let new_data = self.data.iter().map(|&x| x * scalar).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }

    /// Stacks 'other' below 'self' (Vertical Concatenation).
    pub fn vertical_concat(top: &Matrix, bottom: &Matrix) -> Result<Self, StatsError> {
        if top.cols != bottom.cols {
             return Err(StatsError::DimensionMismatch { 
                 expected: format!("Same columns ({} vs {})", top.cols, bottom.cols), 
                 actual: "Column mismatch".into() 
             });
        }
        let mut new_data = top.data.clone();
        new_data.extend(&bottom.data);
        Ok(Matrix {
            rows: top.rows + bottom.rows,
            cols: top.cols,
            data: new_data,
        })
    }

    /// Constructs a new matrix from 4 quadrants.
    pub fn from_blocks(a: &Matrix, b: &Matrix, c: &Matrix, d: &Matrix) -> Result<Matrix, StatsError> {
        if a.rows != b.rows || c.rows != d.rows {
             return Err(StatsError::DimensionMismatch { expected: "Row alignment".into(), actual: "Misaligned".into() });
        }
        if a.cols != c.cols || b.cols != d.cols {
             return Err(StatsError::DimensionMismatch { expected: "Col alignment".into(), actual: "Misaligned".into() });
        }
        
        let total_rows = a.rows + c.rows;
        let total_cols = a.cols + b.cols;
        let mut result = Matrix::zeros(total_rows, total_cols);
        
        // A
        for r in 0..a.rows {
            for c_idx in 0..a.cols {
                result.data[r * total_cols + c_idx] = a.data[r * a.cols + c_idx];
            }
        }
        // B
        for r in 0..b.rows {
            for c_idx in 0..b.cols {
                result.data[r * total_cols + (a.cols + c_idx)] = b.data[r * b.cols + c_idx];
            }
        }
        // C
        for r in 0..c.rows {
            for c_idx in 0..c.cols {
                result.data[(a.rows + r) * total_cols + c_idx] = c.data[r * c.cols + c_idx];
            }
        }
        // D
        for r in 0..d.rows {
            for c_idx in 0..d.cols {
                result.data[(a.rows + r) * total_cols + (a.cols + c_idx)] = d.data[r * d.cols + c_idx];
            }
        }
        Ok(result)
    }

    /// Element-wise addition (self + other)
    /// This supports the call in src/stats/mixed.rs
    pub fn add(&self, other: &Matrix) -> Result<Matrix, StatsError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(StatsError::DimensionMismatch {
                expected: format!("{}x{}", self.rows, self.cols),
                actual: format!("{}x{}", other.rows, other.cols),
            });
        }
        let new_data = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Matrix::new(new_data, self.rows, self.cols).unwrap())
    }

}

// ========================================================
// 2. TRAIT IMPLEMENTATIONS
// ========================================================

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

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (r, c) = index;
        if r >= self.rows || c >= self.cols {
            panic!("Index out of bounds");
        }
        &self.data[r * self.cols + c]
    }
}

impl Add for Matrix {
    type Output = Result<Matrix, StatsError>;
    fn add(self, other: Self) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(StatsError::DimensionMismatch {
                expected: format!("{}x{}", self.rows, self.cols),
                actual: format!("{}x{}", other.rows, other.cols),
            });
        }
        let result_data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Ok(Matrix::new(result_data, self.rows, self.cols).unwrap())
    }
}

// Helpers like mean/sum can stay here as utility functions
pub fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() { return None; }
    let sum: f64 = data.iter().sum();
    Some(sum / data.len() as f64)
}