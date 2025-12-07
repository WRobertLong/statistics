// src/main.rs

use std::io::{self, Write};
use std::collections::HashMap;
use statistics::linalg::{self, Matrix, CSRMatrix};
use statistics::stats::univariate; 
use statistics::stats::linear;     

fn main() {
    println!("========================================");
    println!("   RUST STATS LIBRARY + REPL DEMO       ");
    println!("========================================");

    // =========================================================================
    // 1. Univariate Test (Mean, Median, Variance) - [PRESERVED]
    // =========================================================================
    println!("\n[1] UNIVARIATE STATISTICS");
    println!("-------------------------");

    let my_data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
    
    match univariate::mean(&my_data) {
        Some(m) => println!("Mean:            {:.2}", m),
        None => println!("Vector was empty!"),
    }

    match univariate::median(&my_data) {
        Some(m) => println!("Median:          {:.2}", m),
        None => println!("Vector was empty!"),
    }

    match univariate::sample_variance(&my_data) {
        Some(m) => println!("Sample Variance: {:.2}", m),
        None => println!("Vector was empty!"),
    }

    // =========================================================================
    // 2. Matrix Tests (Inverse & Singular) - [PRESERVED]
    // =========================================================================
    println!("\n[2] MATRIX INVERSION");
    println!("--------------------");

    // 1. Invertible Matrix
    let data = vec![4.0, 7.0, 2.0, 6.0];
    let a = linalg::Matrix::new(data, 2, 2).unwrap();

    println!("Original Matrix A:\n{}", a);

    match a.inverse() {
        Ok(a_inv) => {
            println!("Inverse Matrix A^-1:\n{}", a_inv);
            println!("Verification (A * A^-1):");
            let identity_check = a.multiply(&a_inv).unwrap();
            println!("{}", identity_check);
        },
        Err(e) => println!("Inversion failed: {}", e),
    }

    // 2. Singular Matrix
    println!("Testing Singular Matrix (Should Fail):");
    let singular_data = vec![1.0, 2.0, 2.0, 4.0];
    let b = linalg::Matrix::new(singular_data, 2, 2).unwrap();
    
    match b.inverse() {
        Ok(_) => println!("Error: Singular matrix should not have inverted!"),
        Err(e) => println!("Success! Caught expected error: {}", e),
    }

    // =========================================================================
    // 3. OLS Linear Regression Test - [PRESERVED]
    // =========================================================================
    println!("\n[3] OLS LINEAR REGRESSION");
    println!("-------------------------");
    println!("Target Model: y = 2x + 1");

    // Design Matrix X (Intercept + Slope)
    let x_data = vec![
        1.0, 1.0,  
        1.0, 2.0,  
        1.0, 3.0,  
        1.0, 4.0   
    ];
    let x = linalg::Matrix::new(x_data, 4, 2).unwrap();

    // Target Vector y
    let y_data = vec![3.0, 5.0, 7.0, 9.0];
    let y = linalg::Matrix::new(y_data, 4, 1).unwrap();

    match linear::fit_ols(&x, &y) {
        Ok(beta) => {
            println!("Estimated Coefficients (Beta):");
            println!("{}", beta);
            println!("Intercept: {:.4}", beta[(0, 0)]);
            println!("Slope:     {:.4}", beta[(1, 0)]);
        },
        Err(e) => println!("Regression Failed: {}", e),
    }

    // =========================================================================
    // 4. Sparse Matrix Demo - [NEW]
    // =========================================================================
    println!("\n[4] SPARSE MATRIX (CSR) DEMO");
    println!("----------------------------");
    
    // 1  0  0  5
    // 0  2  0  0
    // 0  0  3  0
    let sparse_data = vec![
        1.0, 0.0, 0.0, 5.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0
    ];
    
    let dense_for_sparse = Matrix::new(sparse_data, 3, 4).unwrap();
    let sparse = CSRMatrix::from_dense(&dense_for_sparse);
    
    println!("Dense form:\n{}", dense_for_sparse);
    println!("CSR Values:      {:?}", sparse.values);      
    println!("CSR Col Indices: {:?}", sparse.col_indices); 
    println!("CSR Row Ptrs:    {:?}", sparse.row_ptr);     

    let v = vec![1.0, 1.0, 1.0, 1.0];
    match sparse.multiply_vector(&v) {
        Ok(res) => println!("\nResult of Sparse * [1,1,1,1]:\n{:?}", res),
        Err(e) => println!("Error: {}", e),
    }

    // =========================================================================
    // 5. Interactive REPL - [NEW]
    // =========================================================================
    println!("\n[5] INTERACTIVE REPL");
    println!("--------------------");
    println!("Commands:");
    println!("  Assign:  $ A = [1, 2, 3, 4, rows=2]");
    println!("  Print:   $ A");
    println!("  Exit:    exit");
    println!("--------------------\n");

    let mut variables: HashMap<String, Matrix> = HashMap::new();

    loop {
        print!("$ ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.eq_ignore_ascii_case("exit") {
            break;
        }
        if input.is_empty() { continue; }

        if input.contains('=') {
            // --- ASSIGNMENT MODE ---
            let parts: Vec<&str> = input.splitn(2, '=').collect();
            let var_name = parts[0].trim().to_string();
            let content = parts[1].trim();

            if content.starts_with('[') && content.ends_with(']') {
                let inner = &content[1..content.len()-1];
                let params: Vec<&str> = inner.split(',').collect();

                let mut data = Vec::new();
                let mut rows = 0;

                for p in params {
                    let p = p.trim();
                    if p.to_lowercase().starts_with("rows") {
                        let kv: Vec<&str> = p.split('=').collect();
                        if kv.len() == 2 {
                            if let Ok(r) = kv[1].trim().parse::<usize>() {
                                rows = r;
                            }
                        }
                    } else if p.to_lowercase().contains("byrow") {
                         println!("(Note: 'byrow' is ignored, assuming row-major)");
                    } else {
                        if let Ok(val) = p.parse::<f64>() {
                            data.push(val);
                        }
                    }
                }

                if rows > 0 && !data.is_empty() {
                    let cols = data.len() / rows;
                    match Matrix::new(data, rows, cols) {
                        Some(m) => {
                            println!("Created Matrix '{}' ({}x{})", var_name, rows, cols);
                            println!("{}", m);
                            variables.insert(var_name, m);
                        },
                        None => println!("Error: Data length does not match dimensions."),
                    }
                } else {
                    println!("Error: Please specify 'rows=X' and ensure data is valid.");
                }
            } else {
                println!("Error: Matrix definition must be in format [1, 2, ...]");
            }
        } else {
            // --- INSPECTION MODE ---
            if let Some(m) = variables.get(input) {
                println!("{}", m);
            } else {
                println!("Unknown variable or command: '{}'", input);
            }
        }
    }
}