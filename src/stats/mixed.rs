// src/stats/mixed.rs
use crate::linalg::Matrix;
use crate::error::StatsError;

/// Solves Henderson's Mixed Model Equations.
pub fn solve_mme(x: &Matrix, z: &Matrix, y: &Matrix, lambda: f64) -> Result<Matrix, StatsError> {
    let xt = x.transpose();
    let zt = z.transpose();
    
    let xt_x = xt.multiply(x)?;
    let xt_z = xt.multiply(z)?;
    let zt_x = zt.multiply(x)?;
    
    let zt_z = zt.multiply(z)?;
    let g_inv_scaled = Matrix::identity(z.cols).scale(lambda); 
    let bottom_right = zt_z.add(&g_inv_scaled)?;
    
    let lhs = Matrix::from_blocks(&xt_x, &xt_z, &zt_x, &bottom_right)?;
    
    let xt_y = xt.multiply(y)?;
    let zt_y = zt.multiply(y)?;
    let rhs = Matrix::vertical_concat(&xt_y, &zt_y)?;
    
    let lhs_inv = lhs.inverse()?;
    let solution = lhs_inv.multiply(&rhs)?;
    
    Ok(solution)
}