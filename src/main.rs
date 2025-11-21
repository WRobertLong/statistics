// src/main.rs
mod linalg; // Import the module (if inside the same crate for testing)

fn main() {
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

    // 2. The Integer Problem
    // let integers = vec![1, 2, 3, 4, 5];
    // linalg::mean(&integers); // <--- THIS WILL ERROR if you uncomment it!
    // Error: expected `&[f64]`, found `&[i32]`
}