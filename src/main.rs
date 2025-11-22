// src/main.rs

use statistics::linalg;

fn main() {
    println!("--- RUST STATS LIBRARY TESTS ---");

    // =========================================================================
    // 1. Univariate Test (Mean)
    // =========================================================================

    // 1. Test with floats (Easy)
    let my_data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
    
    // match handles the Option result (Some or None)
    match linalg::mean(&my_data) {
        Some(m) => println!("Mean: {:.2}", m),
        None => println!("Vector was empty!"),
    }

    match linalg::median(&my_data) {
        Some(m) => println!("Median: {:.2}", m),
        None => println!("Vector was empty!"),
    }

    match linalg::sample_variance(&my_data) {
        Some(m) => println!("Sample Variance|: {:.2}", m),
        None => println!("Vector was empty!"),
    }

    // =========================================================================
    // 2. Matrix Tests (Creation, Indexing, Transpose)
    // =========================================================================
    
    // Define a 3x2 Matrix (Data stored row-major: [1, 2], [3, 4], [5, 6])
    let matrix_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    
    // Create the Matrix
    let a = match linalg::Matrix::new(matrix_data, 3, 2) {
        Some(m) => m,
        None => {
            println!("\n❌ ERROR: Failed to create matrix. Data length mismatch.");
            return;
        }
    };

    println!("\n--- Original Matrix A (3x2) ---");
    println!("{}", a); // Uses fmt::Display
    
    // Test Indexing (Uses std::ops::Index implementation)
    // Element at row 2, col 1 should be 6.0
    let val_2_1 = a[(2, 1)]; 
    println!("Element at A[2, 1] is: {:.1}", val_2_1);
    
    // Test Transpose
    let a_t = a.transpose();
    
    println!("\n--- Transposed Matrix A^T (2x3) ---");
    println!("{}", a_t);
    
    // Test Indexing on Transpose (Element at A^T[1, 2] should also be 6.0)
    let val_t_1_2 = a_t[(1, 2)];
    println!("Element at A^T[1, 2] is: {:.1}", val_t_1_2); 

    println!("\nAll tests completed successfully! 🎉");



    // 2. The Integer Problem
    // let integers = vec![1, 2, 3, 4, 5];
    // linalg::mean(&integers); // <--- THIS WILL ERROR if you uncomment it!
    // Error: expected `&[f64]`, found `&[i32]`
}