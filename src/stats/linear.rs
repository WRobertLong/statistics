use crate::linalg::Matrix;
use crate::error::StatsError;

// --- 1. THE DEFAULT INTERFACE (Convenience Wrapper) ---

/// Default OLS solver. Currently uses QR Decomposition.
/// 
/// We use QR as the default because it works directly on X, avoiding the 
/// formation of X'X. This keeps the condition number as cond(X) rather 
/// than cond(X)^2, providing significantly better numerical stability 
/// for ill-conditioned data.
pub fn fit_ols(x: &Matrix, y: &Matrix) -> Result<Matrix, StatsError> {
    fit_ols_qr(x, y)
}

// --- 2. THE SOLVERS ---

/// The "Gold Standard": Uses QR Decomposition.
/// Slowest (O(2mn^2)), but most numerically stable.
/// Does NOT form X'X, preserving precision.
pub fn fit_ols_qr(x: &Matrix, y: &Matrix) -> Result<Matrix, StatsError> {
    // Solve X * beta = y directly using QR
    // This avoids squaring the condition number of X
    let beta = x.qr_solve(y)?;
    Ok(beta)
}

/// The "Production" version: Uses Cholesky.
/// Fastest (O(mn^2 + n^3/3)), but requires X'X to be positive definite.
/// Use this for performance unless you have collinear data.
pub fn fit_ols_cholesky(x: &Matrix, y: &Matrix) -> Result<Matrix, StatsError> {
    let xt = x.transpose();
    
    // 1. Calculate Gram Matrix: X^T * X
    // This is guaranteed to be Symmetric, and Positive Definite if X is full rank
    let xt_x = xt.multiply(x)?; 
    
    // 2. Calculate Moment Matrix: X^T * y
    let xt_y = xt.multiply(y)?;

    // 3. Solve using Cholesky
    // This is faster than GJ and catches singular matrices (collinearity) gracefully
    let beta = xt_x.cholesky_solve(&xt_y)?;

    Ok(beta)
}

/// The "Robust" version: Uses Gauss-Jordan Elimination.
/// Handles singular matrices explicitly, but O(n^3) and forms X'X.
pub fn fit_ols_gj(x: &Matrix, y: &Matrix) -> Result<Matrix, StatsError> {
    // 1. Calculate X^T
    let xt = x.transpose();

    // 2. Calculate Gram Matrix: X^T * X (The "A" in Ax=b)
    let xt_x = xt.multiply(x)?;

    // 3. Calculate Moment Matrix: X^T * y (The "b" in Ax=b)
    let xt_y = xt.multiply(y)?;

    // 4. Solve for beta directly using Row Operations
    let beta = xt_x.solve(&xt_y)?;

    Ok(beta)
}

/// The "Educational" version: Naive Inverse.
/// Do not use in production. 
/// Works by direct inversion (.inverse) and is numerically unstable.
pub fn fit_ols_naive(x: &Matrix, y: &Matrix) -> Result<Matrix, StatsError> {
    // 1. Calculate X^T
    let xt = x.transpose();

    // 2. Calculate Gram Matrix: X^T * X
    let xt_x = xt.multiply(x)?;

    // 3. Calculate Inverse: (X^T * X)^-1
    // WARNING: This is numerically unstable if X is ill-conditioned!
    let xt_x_inv = xt_x.inverse()?;

    // 4. Calculate Moment Matrix: X^T * y
    let xt_y = xt.multiply(y)?;

    // 5. Calculate Coefficients: (X^T * X)^-1 * (X^T * y)
    let beta = xt_x_inv.multiply(&xt_y)?;

    Ok(beta)
}

// --- 3. TESTS ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Matrix;

    #[test]
    fn test_ols_simple_linear() {
        // Data: y = 2x + 1
        let x_data = vec![
            1.0, 1.0, 
            1.0, 2.0, 
            1.0, 3.0
        ];
        let x = Matrix::new(x_data, 3, 2).unwrap();

        let y_data = vec![3.0, 5.0, 7.0];
        let y = Matrix::new(y_data, 3, 1).unwrap();

        // Use the default wrapper (which now uses QR)
        let beta = fit_ols(&x, &y).expect("OLS fit failed");

        let intercept = beta[(0, 0)];
        let slope = beta[(1, 0)];

        assert!((intercept - 1.0).abs() < 1e-6, "Intercept wrong");
        assert!((slope - 2.0).abs() < 1e-6, "Slope wrong");
    }
}