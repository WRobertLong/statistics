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

    println!("--- MATRIX INVERSION TEST ---");

    // 1. Define a 2x2 Matrix (that we know is invertible)
    // Matrix A = [[4, 7], [2, 6]]
    // Determinant = (4*6) - (7*2) = 24 - 14 = 10.
    // Since Det != 0, it is invertible.
    let data = vec![4.0, 7.0, 2.0, 6.0];
    let a = linalg::Matrix::new(data, 2, 2).unwrap();

    println!("Original Matrix A:\n{}", a);

    // 2. Calculate Inverse
    match a.inverse() {
        Ok(a_inv) => {
            println!("Inverse Matrix A^-1:\n{}", a_inv);
            
            // MANUAL CHECK:
            // Inverse should be: [[0.6, -0.7], [-0.2, 0.4]]
            
            // 3. Verify: A * A^-1 should equal Identity
            println!("Verification (A * A^-1):");
            let identity_check = a.multiply(&a_inv).unwrap();
            println!("{}", identity_check);
            
            // The result should look like:
            // 1.0000  0.0000
            // 0.0000  1.0000
        },
        Err(e) => println!("Inversion failed: {}", e),
    }

    println!("\n--- SINGULAR MATRIX TEST (Should Fail) ---");
    // Matrix B = [[1, 2], [2, 4]]
    // Det = 4 - 4 = 0. This is singular.
    let singular_data = vec![1.0, 2.0, 2.0, 4.0];
    let b = linalg::Matrix::new(singular_data, 2, 2).unwrap();
    
    match b.inverse() {
        Ok(_) => println!("Error: Singular matrix should not have inverted!"),
        Err(e) => println!("Success! Caught expected error: {}", e),
    }

    println!("--- OLS LINEAR REGRESSION TEST ---");
    println!("Target Model: y = 2x + 1");

    // 1. Create Design Matrix X (4 rows, 2 columns)
    // Col 0: Intercept (1.0)
    // Col 1: Variable x (1, 2, 3, 4)
    let x_data = vec![
        1.0, 1.0,  // Row 1
        1.0, 2.0,  // Row 2
        1.0, 3.0,  // Row 3
        1.0, 4.0   // Row 4
    ];
    let x = linalg::Matrix::new(x_data, 4, 2).unwrap();

    // 2. Create Target Vector y (4 rows, 1 column)
    // y = 2(1)+1 = 3
    // y = 2(2)+1 = 5
    // ...
    let y_data = vec![3.0, 5.0, 7.0, 9.0];
    let y = linalg::Matrix::new(y_data, 4, 1).unwrap();

    println!("Design Matrix X:\n{}", x);
    println!("Target Vector y:\n{}", y);

    // 3. Run Regression
    match linalg::ols(&x, &y) {
        Ok(beta) => {
            println!("--------------------------------");
            println!("Estimated Coefficients (Beta):");
            println!("{}", beta);
            
            println!("Interpretation:");
            println!("Intercept (Beta_0): {:.4}", beta[(0, 0)]);
            println!("Slope (Beta_1):     {:.4}", beta[(1, 0)]);
        },
        Err(e) => println!("Regression Failed: {}", e),
    }

    println!("\nAll tests completed successfully! 🎉");




}