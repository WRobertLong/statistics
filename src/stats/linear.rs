use crate::linalg::Matrix;
use crate::error::StatsError;

/// Performs Ordinary Least Squares (OLS) Linear Regression.
/// Returns the vector of coefficients (beta).
/// Solves (X^T * X) * beta = (X^T * y)
pub fn fit_ols(x: &Matrix, y: &Matrix) -> Result<Matrix, StatsError> {
    // 1. Calculate X^T
    let xt = x.transpose();

    // 2. Calculate Gram Matrix: X^T * X (The "A" in Ax=b)
    let xt_x = xt.multiply(x)?;

    // 3. Calculate Moment Matrix: X^T * y (The "b" in Ax=b)
    let xt_y = xt.multiply(y)?;

    // 4. Solve for beta directly (No explicit inverse!)
    // We are solving: (XtX) * beta = (XtY)
    let beta = xt_x.solve(&xt_y)?;

    Ok(beta)
}





/// Performs Ordinary Least Squares (OLS) Linear Regression.
/// Returns the vector of coefficients (beta).
pub fn fit_ols_basic(x: &Matrix, y: &Matrix) -> Result<Matrix, StatsError> {
    // 1. Calculate X^T
    let xt = x.transpose();

    // 2. Calculate Gram Matrix: X^T * X
    let xt_x = xt.multiply(x)?;

    // 3. Calculate Inverse: (X^T * X)^-1
    let xt_x_inv = xt_x.inverse()?;

    // 4. Calculate Moment Matrix: X^T * y
    let xt_y = xt.multiply(y)?;

    // 5. Calculate Coefficients: (X^T * X)^-1 * (X^T * y)
    let beta = xt_x_inv.multiply(&xt_y)?;

    Ok(beta)
}