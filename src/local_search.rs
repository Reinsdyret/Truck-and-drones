use rand::random_range;

use crate::{Solution, TruckAndDroneInstance};
use crate::operators::Operator;

/// Run local search with random operator selection from a given list
pub fn run_local_search(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&dyn Operator],
    max_iterations: usize,
) -> Solution {
    let mut best_solution = init_solution.clone();
    let mut best_score = best_solution.verify_and_cost(instance)
        .map(|r| r.total_time)
        .unwrap_or(f64::INFINITY);
    
    for iter in 0..max_iterations {
        let op_idx = random_range(0..operators.len());
        let operator = operators[op_idx];
        
        if let Some(new_solution) = operator.apply(&best_solution, instance) {
            if let Ok(result) = new_solution.verify_and_cost(instance) {
                if result.total_time < best_score {
                    best_solution = new_solution;
                    best_score = result.total_time;
                    println!("Iter {}: New best = {:.2}", iter, best_score);
                }
            }
        }
    }
    
    println!("Final best score: {:.2}", best_score);
    best_solution
}
