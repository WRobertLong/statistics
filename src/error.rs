// src/error.rs
use thiserror::Error;

#[derive(Error, Debug, Clone)] // Ensure Clone is here
pub enum StatsError {
    #[error("Dimension mismatch: Operation requires matrices of shape {expected} but got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    #[error("Matrix is not square: Expected square matrix but got {0}x{1}")]
    NotSquare(usize, usize),

    #[error("Matrix is singular and cannot be inverted.")]
    SingularMatrix,
    
    #[error("Data length does not match matrix dimensions.")]
    DataLengthMismatch,

    #[error("Generic error: {0}")]
    Generic(String),
}