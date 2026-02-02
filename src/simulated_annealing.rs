use std::fs::File;
use std::io::{BufWriter, Write};
use rand::random_range;

use crate::{Solution, TruckAndDroneInstance};
use crate::operators::Operator;

/// Adaptive operator selector with weights updated based on performance
pub struct AdaptiveOperatorSelector {
    weights: Vec<f64>,
    uses: Vec<usize>,
    improvements: Vec<f64>,  // Total improvement found
    decay: f64,              // Weight decay factor (0-1)
    min_weight: f64,         // Minimum weight to prevent starvation
}

impl AdaptiveOperatorSelector {
    pub fn new(n_operators: usize, decay: f64) -> Self {
        Self {
            weights: vec![1.0; n_operators],
            uses: vec![0; n_operators],
            improvements: vec![0.0; n_operators],
            decay,
            min_weight: 0.1,
        }
    }
    
    /// Select operator index based on weights (roulette wheel)
    pub fn select(&self) -> usize {
        let total: f64 = self.weights.iter().sum();
        let mut r = rand::random::<f64>() * total;
        
        for (i, &w) in self.weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        self.weights.len() - 1
    }
    
    /// Record that operator was used
    pub fn record_use(&mut self, op_idx: usize) {
        self.uses[op_idx] += 1;
    }
    
    /// Record improvement found by operator (negative delta = improvement)
    pub fn record_improvement(&mut self, op_idx: usize, improvement: f64) {
        if improvement > 0.0 {
            self.improvements[op_idx] += improvement;
        }
    }
    
    /// Update weights based on performance (call periodically)
    pub fn update_weights(&mut self) {
        for i in 0..self.weights.len() {
            if self.uses[i] > 0 {
                // Performance = improvement per use
                let performance = self.improvements[i] / self.uses[i] as f64;
                // Update weight: decay old weight, add performance bonus
                self.weights[i] = (self.weights[i] * self.decay + performance).max(self.min_weight);
            }
        }
        
        // Normalize weights
        let total: f64 = self.weights.iter().sum();
        let n = self.weights.len() as f64;
        if total > 0.0 {
            let scale = total / n;
            for w in &mut self.weights {
                *w /= scale;
            }
        }
        
        // Reset counters for next period
        self.uses.fill(0);
        self.improvements.fill(0.0);
    }
    
    pub fn get_weights(&self) -> &[f64] {
        &self.weights
    }
}

pub fn run_simulated_annealing(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&dyn Operator],
    iterations: usize,
) -> Solution {
    run_simulated_annealing_with_params(init_solution, instance, operators, 0.8, 0.01, iterations, 10000)
}

pub fn run_simulated_annealing_with_params(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&dyn Operator],
    accept_prob: f64,
    t_final_ratio: f64,  // Final temp as ratio of t_zero (e.g. 0.01 = 1% of initial)
    iterations: usize,
    reheat_interval: usize,  // Reheat if no improvement for this many iterations
) -> Solution {
    let (delta_avg, mut incumbent, mut best_solution, mut incumbent_score, mut best_score) =
        estimate_avg_delta(init_solution, instance, operators, accept_prob);

    // Calculate initial temperature from average delta
    let t_zero = if delta_avg <= 0.0 {
        1.0
    } else {
        delta_avg / (-accept_prob.ln())
    };
    let t_final = t_zero * t_final_ratio;
    
    // Cooling rate
    let alpha = if iterations <= 1 {
        1.0
    } else {
        (t_final / t_zero).powf(1.0 / (iterations as f64 - 1.0))
    };
    
    let mut temp = t_zero;
    let mut no_improve_count = 0;

    // CSV logging
    let output_path = "simulated_annealing_scores.csv";
    let file = File::create(output_path).expect("failed to create simulated_annealing_scores.csv");
    let mut writer = BufWriter::new(file);
    writeln!(writer, "iter,best_score,incumbent_score,temp").expect("failed to write header");

    for i in 0..iterations {
        if i % 100_000 == 0 {
            println!("Iter {}: best = {:.2}, incumbent = {:.2}, temp = {:.4}", i, best_score, incumbent_score, temp);
        }
        
        // Pick random operator
        let op_idx = random_range(0..operators.len());
        let operator = operators[op_idx];
        
        let new_solution = match operator.apply(&incumbent, instance) {
            Some(sol) => sol,
            None => {
                temp *= alpha;
                no_improve_count += 1;
                continue;
            }
        };
        
        // Check feasibility and get score
        let new_score = match new_solution.verify_and_cost(instance) {
            Ok(result) => result.total_time,
            Err(_) => {
                temp *= alpha;
                no_improve_count += 1;
                continue;
            }
        };

        // Delta (negative = improvement since we minimize)
        let delta = new_score - incumbent_score;

        if delta < 0.0 {
            // Better solution - accept
            incumbent = new_solution;
            incumbent_score = new_score;
            no_improve_count = 0;
            if new_score < best_score {
                best_solution = incumbent.clone();
                best_score = new_score;
            }
        } else {
            // Worse solution - accept with probability
            let p = (-delta / temp).exp();
            if rand::random::<f64>() < p {
                incumbent = new_solution;
                incumbent_score = new_score;
            }
            no_improve_count += 1;
        }

        // Reheat if stuck
        if no_improve_count >= reheat_interval {
            temp = t_zero * 0.5; // Reheat to 50% of initial temp
            incumbent = best_solution.clone(); // Restart from best
            incumbent_score = best_score;
            no_improve_count = 0;
            println!("Iter {}: REHEAT, restarting from best = {:.2}", i, best_score);
        }

        writeln!(writer, "{},{:.4},{:.4},{:.6}", i, best_score, incumbent_score, temp)
            .expect("failed to write score row");
        
        temp *= alpha;
    }

    println!("Final best score: {:.2}", best_score);
    best_solution
}

/// Simulated annealing with adaptive operator weights
pub fn run_adaptive_sa(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&dyn Operator],
    iterations: usize,
    weight_update_interval: usize,
    reheat_interval: usize,
    decay: f64,
) -> Solution {
    let (delta_avg, mut incumbent, mut best_solution, mut incumbent_score, mut best_score) =
        estimate_avg_delta(init_solution, instance, operators, 0.8);

    let t_zero = if delta_avg <= 0.0 { 1.0 } else { delta_avg / (-0.8f64.ln()) };
    let t_final = t_zero * 0.01;
    let alpha = if iterations <= 1 { 1.0 } else { (t_final / t_zero).powf(1.0 / (iterations as f64 - 1.0)) };
    let mut temp = t_zero;
    let mut no_improve_count = 0;

    let mut selector = AdaptiveOperatorSelector::new(operators.len(), decay);

    let output_path = "adaptive_sa_scores.csv";
    let file = File::create(output_path).expect("failed to create csv");
    let mut writer = BufWriter::new(file);
    
    // Header with weight columns
    let header = format!("iter,best_score,incumbent_score,temp,{}", 
        (0..operators.len()).map(|i| format!("w{}", i)).collect::<Vec<_>>().join(","));
    writeln!(writer, "{}", header).expect("failed to write header");

    for i in 0..iterations {
        if i % 100_000 == 0 {
            println!("Iter {}: best = {:.2}, incumbent = {:.2}, temp = {:.4}, weights = {:?}", 
                     i, best_score, incumbent_score, temp, selector.get_weights());
        }
        
        // Select operator adaptively
        let op_idx = selector.select();
        selector.record_use(op_idx);
        let operator = operators[op_idx];
        
        let new_solution = match operator.apply(&incumbent, instance) {
            Some(sol) => sol,
            None => {
                temp *= alpha;
                no_improve_count += 1;
                continue;
            }
        };
        
        let new_score = match new_solution.verify_and_cost(instance) {
            Ok(result) => result.total_time,
            Err(_) => {
                temp *= alpha;
                no_improve_count += 1;
                continue;
            }
        };

        let delta = new_score - incumbent_score;

        if delta < 0.0 {
            // Record improvement (negative delta means improvement)
            selector.record_improvement(op_idx, -delta);
            
            incumbent = new_solution;
            incumbent_score = new_score;
            no_improve_count = 0;
            if new_score < best_score {
                best_solution = incumbent.clone();
                best_score = new_score;
            }
        } else {
            let p = (-delta / temp).exp();
            if rand::random::<f64>() < p {
                incumbent = new_solution;
                incumbent_score = new_score;
            }
            no_improve_count += 1;
        }

        // Reheat if stuck
        if no_improve_count >= reheat_interval {
            temp = t_zero * 0.5;
            incumbent = best_solution.clone();
            incumbent_score = best_score;
            no_improve_count = 0;
            println!("Iter {}: REHEAT, restarting from best = {:.2}", i, best_score);
        }

        // Update weights periodically
        if (i + 1) % weight_update_interval == 0 {
            selector.update_weights();
        }

        // Log with weights
        let weights_str = selector.get_weights().iter()
            .map(|w| format!("{:.4}", w))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(writer, "{},{:.4},{:.4},{:.6},{}", i, best_score, incumbent_score, temp, weights_str)
            .expect("failed to write row");
        
        temp *= alpha;
    }

    println!("Final best score: {:.2}", best_score);
    println!("Final weights: {:?}", selector.get_weights());
    best_solution
}

/// Estimate average delta from initial random walk to set initial temperature
fn estimate_avg_delta(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&dyn Operator],
    accept_prob: f64,
) -> (f64, Solution, Solution, f64, f64) {
    let mut incumbent = init_solution.clone();
    let mut best_solution = init_solution.clone();
    let mut incumbent_score = incumbent.verify_and_cost(instance)
        .map(|r| r.total_time)
        .unwrap_or(f64::INFINITY);
    let mut best_score = incumbent_score;

    let mut deltas: Vec<f64> = Vec::new();

    for _ in 0..1000 {
        let op_idx = random_range(0..operators.len());
        let operator = operators[op_idx];
        
        let new_solution = match operator.apply(&incumbent, instance) {
            Some(sol) => sol,
            None => continue,
        };
        
        let new_score = match new_solution.verify_and_cost(instance) {
            Ok(result) => result.total_time,
            Err(_) => continue,
        };

        let delta = new_score - incumbent_score;

        if delta < 0.0 {
            // Improvement
            incumbent = new_solution;
            incumbent_score = new_score;
            if new_score < best_score {
                best_solution = incumbent.clone();
                best_score = new_score;
            }
        } else {
            // Worsening - record delta and sometimes accept
            deltas.push(delta);
            if rand::random::<f64>() < accept_prob {
                incumbent = new_solution;
                incumbent_score = new_score;
            }
        }
    }

    let avg_delta = if deltas.is_empty() {
        1.0
    } else {
        deltas.iter().sum::<f64>() / deltas.len() as f64
    };

    (avg_delta, incumbent, best_solution, incumbent_score, best_score)
}
