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

    /// Solves the linear system Ax = b using Gauss-Jordan elimination.
    /// Returns x.
    /// 'self' is A, 'rhs' is b.
    pub fn solve(&self, rhs: &Matrix) -> Result<Matrix, StatsError> {
        if self.rows != self.cols {
            return Err(StatsError::NotSquare(self.rows, self.cols));
        }
        if self.rows != rhs.rows {
            return Err(StatsError::DimensionMismatch {
                expected: format!("A.rows ({}) == b.rows ({})", self.rows, rhs.rows),
                actual: format!("{} vs {}", self.rows, rhs.rows),
            });
        }

        let n = self.rows;
        // We clone both because we need to mutate them to reduce A to Identity
        let mut aug_a = self.clone(); 
        let mut aug_b = rhs.clone();

        for i in 0..n {
            // 1. Pivot: Find the best row
            let mut pivot_row = i;
            let mut max_val = aug_a[(i, i)].abs();
            
            for r in (i + 1)..n {
                if aug_a[(r, i)].abs() > max_val {
                    max_val = aug_a[(r, i)].abs();
                    pivot_row = r;
                }
            }

            if max_val < 1e-10 {
                return Err(StatsError::SingularMatrix);
            }

            // 2. Swap Rows in BOTH A and b
            if pivot_row != i {
                aug_a.swap_rows(i, pivot_row);
                aug_b.swap_rows(i, pivot_row);
            }

            // 3. Normalize the pivot row
            let pivot = aug_a[(i, i)];
            for c in 0..n {
                aug_a.data[i * n + c] /= pivot;
            }
            // Apply to RHS (note: rhs can have multiple columns, so iterate cols)
            for c in 0..aug_b.cols {
                aug_b.data[i * aug_b.cols + c] /= pivot;
            }

            // 4. Eliminate other rows
            for r in 0..n {
                if r != i {
                    let factor = aug_a[(r, i)];
                    
                    // Subtract from A
                    for c in 0..n {
                        let val = aug_a[(i, c)];
                        aug_a.data[r * n + c] -= factor * val;
                    }
                    
                    // Subtract from b
                    for c in 0..aug_b.cols {
                        let val = aug_b[(i, c)];
                        aug_b.data[r * aug_b.cols + c] -= factor * val;
                    }
                }
            }
        }

        // At this point, aug_a is Identity, and aug_b is the solution x
        Ok(aug_b)
    }

    // ... inside impl Matrix ...

    // --- QR DECOMPOSITION (The Numerical Gold Standard) ---

    /// Computes the Reduced QR Decomposition: A = Q * R
    /// Q is mxn (orthogonal columns), R is nxn (upper triangular).
    /// Uses Householder reflections for numerical stability.
    pub fn qr(&self) -> Result<(Matrix, Matrix), StatsError> {
        let m = self.rows;
        let n = self.cols;
        
        // We will modify 'q' in-place to accumulate reflections, then extract R.
        // For a full implementation, we'd store Householder vectors, 
        // but for learning, explicit Q construction is clearer.
        let mut q = Matrix::identity(m);
        let mut r = self.clone();

        for k in 0..n {
            // 1. Get the column vector x below the diagonal in column k
            // We need to zero out elements r[k+1..m, k]
            
            // Norm of the sub-column x
            let mut norm_x_sq = 0.0;
            for i in k..m {
                norm_x_sq += r[(i, k)] * r[(i, k)];
            }
            let norm_x = norm_x_sq.sqrt();
            
            // If column is already 0, skip (singular-ish)
            if norm_x < 1e-15 { continue; }

            // 2. Construct Householder Vector 'u'
            // u = x ± ||x|| * e_1
            // Sign choice prevents catastrophic cancellation: u[0] += sign(x[0]) * norm
            let alpha = if r[(k, k)] >= 0.0 { -norm_x } else { norm_x };
            
            // We only need the non-zero part of u (from row k to m)
            // But to apply it, we usually think of it as size m.
            // Let's optimize: Store u_vec physically as size (m-k)
            let mut u_vec = Vec::with_capacity(m - k);
            
            // u[0] = x[0] - alpha
            u_vec.push(r[(k, k)] - alpha);
            
            // Copy rest of x
            for i in (k + 1)..m {
                u_vec.push(r[(i, k)]);
            }

            // Normalize u: v = u / ||u||
            let mut norm_u_sq = 0.0;
            for val in &u_vec { norm_u_sq += val * val; }
            let norm_u = norm_u_sq.sqrt();
            
            if norm_u < 1e-15 { continue; } // Should not happen if norm_x > 0

            for val in &mut u_vec { *val /= norm_u; }

            // 3. Apply Householder Reflection to R: R = (I - 2vv^T) R
            // We only update the submatrix R[k..m, k..n]
            // R = R - 2 * v * (v^T * R)
            
            for j in k..n {
                // Dot product v . col_j
                let mut dot = 0.0;
                for i in 0..(m - k) {
                    dot += u_vec[i] * r[(k + i, j)];
                }

                // Subtract
                for i in 0..(m - k) {
                    r.data[(k + i) * n + j] -= 2.0 * u_vec[i] * dot;
                }
            }

            // 4. Apply Householder Reflection to Q: Q = Q * (I - 2vv^T)
            // Note order! We accumulate Q by right-multiplying H_k
            // Q_new = Q_old - 2 * (Q_old * v) * v^T
            
            for i in 0..m {
                // Dot product row_i_of_Q . v
                let mut dot = 0.0;
                for l in 0..(m - k) {
                    dot += q[(i, k + l)] * u_vec[l];
                }

                // Subtract
                for l in 0..(m - k) {
                    q.data[i * m + (k + l)] -= 2.0 * dot * u_vec[l];
                }
            }
        }
        
        // Return Reduced Q (first n columns) and Reduced R (top n rows)
        // We currently have Q as mxm and R as mxn (with zeros at bottom).
        // Let's chop them.
        
        let q_reduced = if m > n {
            let mut data = Vec::with_capacity(m * n);
            for r in 0..m {
                for c in 0..n {
                    data.push(q[(r, c)]);
                }
            }
            Matrix::new(data, m, n).unwrap()
        } else {
            q 
        };

        let r_reduced = if m > n {
             let mut data = Vec::with_capacity(n * n);
             for r in 0..n {
                 for c in 0..n {
                     data.push(r[(r, c)]);
                 }
             }
             Matrix::new(data, n, n).unwrap()
        } else {
            r
        };

        Ok((q_reduced, r_reduced))
    }

    /// Solves Ax = b using QR Decomposition.
    /// 1. A = Q * R
    /// 2. R * x = Q^T * b
    /// 3. Back-substitution for x
    pub fn qr_solve(&self, rhs: &Matrix) -> Result<Matrix, StatsError> {
        let (q, r) = self.qr()?;
        
        // 1. Calculate y = Q^T * b
        // Q is mxn, b is mx1 (or mxp). Q^T is nxm.
        // y will be nx1.
        let qt = q.transpose();
        let y = qt.multiply(rhs)?;

        // 2. Solve R * x = y using Back Substitution
        // R is nxn upper triangular.
        let n = r.rows;
        let mut x = Matrix::zeros(n, rhs.cols);

        for k in 0..rhs.cols {
            for i in (0..n).rev() {
                let mut sum = y[(i, k)];
                for j in (i + 1)..n {
                    sum -= r[(i, j)] * x[(j, k)];
                }
                
                if r[(i, i)].abs() < 1e-15 {
                     return Err(StatsError::SingularMatrix);
                }
                x.data[i * rhs.cols + k] = sum / r[(i, i)];
            }
        }

        Ok(x)
    }


    /// Performs Cholesky decomposition: A = L * L^T.
    /// Returns L (Lower Triangular Matrix).
    /// Requires matrix to be Symmetric Positive Definite (SPD).
    pub fn cholesky(&self) -> Result<Matrix, StatsError> {
        if self.rows != self.cols {
            return Err(StatsError::NotSquare(self.rows, self.cols));
        }
        
        let n = self.rows;
        let mut l = Matrix::zeros(n, n);

        for i in 0..n {
            for j in 0..=i {
                let mut sum = self[(i, j)];
                
                for k in 0..j {
                    sum -= l[(i, k)] * l[(j, k)];
                }

                if i == j {
                    // Diagonal element
                    if sum <= 0.0 {
                        return Err(StatsError::Generic("Matrix is not Positive Definite".into()));
                    }
                    l.data[i * n + j] = sum.sqrt();
                } else {
                    // Off-diagonal
                    l.data[i * n + j] = sum / l[(j, j)];
                }
            }
        }
        Ok(l)
    }

    /// Solves Ax = b using Cholesky decomposition.
    pub fn cholesky_solve(&self, rhs: &Matrix) -> Result<Matrix, StatsError> {
        let l = self.cholesky()?;
        let n = self.rows;
        
        // 1. Forward Substitution: Solve L * y = b
        let mut y = Matrix::zeros(n, rhs.cols);
        for k in 0..rhs.cols {
            for i in 0..n {
                let mut sum = rhs[(i, k)];
                for j in 0..i {
                    sum -= l[(i, j)] * y[(j, k)];
                }
                y.data[i * rhs.cols + k] = sum / l[(i, i)];
            }
        }

        // 2. Backward Substitution: Solve L^T * x = y
        let mut x = Matrix::zeros(n, rhs.cols);
        for k in 0..rhs.cols {
            for i in (0..n).rev() {
                let mut sum = y[(i, k)];
                for j in (i + 1)..n {
                    sum -= l[(j, i)] * x[(j, k)];
                }
                x.data[i * rhs.cols + k] = sum / l[(i, i)];
            }
        }

        Ok(x)
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

// --- SPARSE MATRIX SUPPORT ---

/// Compressed Sparse Row (CSR) Matrix.
/// Efficient format for arithmetic operations (row-slicing is fast).
/// 
/// Example:
/// [ 1, 0, 2 ]
/// [ 0, 0, 3 ]
/// [ 4, 5, 6 ]
///
/// values:      [ 1, 2, 3, 4, 5, 6 ] (Only non-zeros)
/// col_indices: [ 0, 2, 2, 0, 1, 2 ] (Column for each value)
/// row_ptr:     [ 0, 2, 3, 6 ]       (Index where each row starts. Last is total len)
#[derive(Debug, Clone)]
pub struct CSRMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f64>,
    pub col_indices: Vec<usize>,
    pub row_ptr: Vec<usize>,
}

impl CSRMatrix {
    /// Creates a new empty CSR Matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        // row_ptr always has length rows + 1. Starts with 0.
        let row_ptr = vec![0; rows + 1];
        CSRMatrix {
            rows,
            cols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptr,
        }
    }

    /// Converts a Dense Matrix to Sparse CSR format
    /// (Useful for testing our sparse logic against dense logic)
    pub fn from_dense(matrix: &Matrix) -> Self {
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = Vec::with_capacity(matrix.rows + 1);
        
        row_ptr.push(0); // Start of Row 0

        for r in 0..matrix.rows {
            for c in 0..matrix.cols {
                let val = matrix[(r, c)];
                if val.abs() > 1e-10 { // Treat small numbers as zero
                    values.push(val);
                    col_indices.push(c);
                }
            }
            // The start of the NEXT row is the current total number of values
            row_ptr.push(values.len());
        }

        CSRMatrix {
            rows: matrix.rows,
            cols: matrix.cols,
            values,
            col_indices,
            row_ptr,
        }
    }

    /// Sparse Matrix - Vector Multiplication
    /// y = A * x
    /// O(nnz) - Only iterates over non-zeros!
    pub fn multiply_vector(&self, x: &[f64]) -> Result<Vec<f64>, StatsError> {
        if x.len() != self.cols {
            return Err(StatsError::DimensionMismatch {
                expected: format!("Vector len {}", self.cols),
                actual: format!("Vector len {}", x.len()),
            });
        }

        let mut result = vec![0.0; self.rows];

        // Iterate over rows
        for r in 0..self.rows {
            let row_start = self.row_ptr[r];
            let row_end = self.row_ptr[r + 1];

            let mut sum = 0.0;
            // The Slice Trick: We only look at values for this row
            for idx in row_start..row_end {
                let val = self.values[idx];
                let col = self.col_indices[idx];
                sum += val * x[col];
            }
            result[r] = sum;
        }

        Ok(result)
    }
}