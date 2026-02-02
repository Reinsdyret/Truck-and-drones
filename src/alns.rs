use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;

use crate::{Solution, TruckAndDroneInstance};
use crate::operators::Operator;

/// ALNS with time limit
pub fn alns_timed(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&dyn Operator],
    time_limit: Duration,
) -> Solution {
    let instant = Instant::now();
    let mut rng = rand::rng();
    
    let mut incumbent = init_solution.clone();
    let mut incumbent_cost = incumbent.verify_and_cost(instance)
        .map(|r| r.total_time).unwrap_or(f64::INFINITY);
    let mut best_sol = incumbent.clone();
    let mut best_cost = incumbent_cost;
    
    let weights: Vec<f64> = vec![1.0 / operators.len() as f64; operators.len()];
    let mut dist = WeightedIndex::new(&weights).unwrap();
    
    let escape_condition = 500;
    let escape_size = 3;
    let mut iterations_since_improvement = 0;
    let mut iterations: usize = 0;
    
    let mut time_since_last_report = Instant::now();

    while instant.elapsed() < time_limit {
        if time_since_last_report.elapsed().as_secs() > 10 {
            println!("Iter {}: best = {:.2}, incumbent = {:.2}", iterations, best_cost, incumbent_cost);
            time_since_last_report = Instant::now();
        }
        
        iterations += 1;
        iterations_since_improvement += 1;
        
        // Adaptive threshold based on remaining time
        let progress = instant.elapsed().as_secs_f64() / time_limit.as_secs_f64();
        let d = 0.2 * (1.0 - progress) * best_cost;
        
        // Reset to best if stuck too long
        if iterations_since_improvement > escape_condition * 5 {
            incumbent = best_sol.clone();
            incumbent_cost = best_cost;
            iterations_since_improvement = 0;
        }
        
        // Escape if stuck
        if iterations_since_improvement >= escape_condition {
            incumbent = escape(instance, &incumbent, operators, &mut dist, &mut rng, escape_size);
            incumbent_cost = incumbent.verify_and_cost(instance)
                .map(|r| r.total_time).unwrap_or(f64::INFINITY);
            
            if incumbent_cost < best_cost {
                best_cost = incumbent_cost;
                best_sol = incumbent.clone();
                iterations_since_improvement = 0;
            }
        }
        
        // Choose and apply operator
        let op_idx = dist.sample(&mut rng);
        let new_solution = match operators[op_idx].apply(&incumbent, instance) {
            Some(sol) => sol,
            None => continue,
        };
        
        let new_cost = match new_solution.verify_and_cost(instance) {
            Ok(result) => result.total_time,
            Err(_) => continue,
        };
        
        let delta = new_cost - incumbent_cost;
        
        if delta < 0.0 {
            // Improvement
            incumbent = new_solution;
            incumbent_cost = new_cost;
            
            if incumbent_cost < best_cost {
                iterations_since_improvement = 0;
                best_cost = incumbent_cost;
                best_sol = incumbent.clone();
                println!("Iter {}: NEW BEST = {:.2}", iterations, best_cost);
            }
        } else if incumbent_cost < best_cost + d {
            // Accept within threshold
            incumbent = new_solution;
            incumbent_cost = new_cost;
        }
    }
    
    println!("Final best: {:.2} after {} iterations", best_cost, iterations);
    best_sol
}

/// ALNS with SA acceptance and adaptive weights
pub fn alns(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&dyn Operator],
    max_iterations: usize,
) -> Solution {
    let mut rng = rand::rng();
    
    let mut incumbent = init_solution.clone();
    let mut incumbent_cost = incumbent.verify_and_cost(instance)
        .map(|r| r.total_time).unwrap_or(f64::INFINITY);
    let mut best_sol = incumbent.clone();
    let mut best_cost = incumbent_cost;
    
    let n_ops = operators.len();
    let mut weights: Vec<f64> = vec![1.0 / n_ops as f64; n_ops];
    let mut dist = WeightedIndex::new(&weights).unwrap();
    
    let mut operator_use_counts = vec![0usize; n_ops];
    let mut operator_points = vec![0i32; n_ops];
    
    let r = 0.2; // Weight update rate
    let segment_size = 100;
    let reheat_interval = 10000;
    let mut iterations_since_improvement = 0;
    
    // SA temperature setup
    let t_zero = incumbent_cost * 0.05; // Start temp = 5% of initial cost
    let t_final = t_zero * 0.001;
    let alpha = (t_final / t_zero).powf(1.0 / max_iterations as f64);
    let mut temp = t_zero;
    
    // CSV logging
    let file = File::create("alns_scores.csv").expect("failed to create csv");
    let mut writer = BufWriter::new(file);
    let header = format!("iter,best,incumbent,temp,{}", 
        (0..n_ops).map(|i| format!("w{}", i)).collect::<Vec<_>>().join(","));
    writeln!(writer, "{}", header).unwrap();

    for iter in 0..max_iterations {
        if iter % 10000 == 0 {
            println!("Iter {}: best = {:.2}, incumbent = {:.2}, temp = {:.4}, weights = {:?}", 
                     iter, best_cost, incumbent_cost, temp, weights);
        }
        
        iterations_since_improvement += 1;
        
        // Reheat if stuck
        if iterations_since_improvement >= reheat_interval {
            temp = t_zero * 0.5;
            incumbent = best_sol.clone();
            incumbent_cost = best_cost;
            iterations_since_improvement = 0;
            println!("Iter {}: REHEAT", iter);
        }
        
        // Choose and apply operator
        let op_idx = dist.sample(&mut rng);
        operator_use_counts[op_idx] += 1;
        
        let new_solution = match operators[op_idx].apply(&incumbent, instance) {
            Some(sol) => sol,
            None => {
                temp *= alpha;
                continue;
            }
        };
        
        let new_cost = match new_solution.verify_and_cost(instance) {
            Ok(result) => result.total_time,
            Err(_) => {
                temp *= alpha;
                continue;
            }
        };
        
        let delta = new_cost - incumbent_cost;
        
        if delta < 0.0 {
            // Improvement - always accept
            incumbent = new_solution;
            incumbent_cost = new_cost;
            iterations_since_improvement = 0;
            
            if incumbent_cost < best_cost {
                operator_points[op_idx] += 5; // New best
                best_cost = incumbent_cost;
                best_sol = incumbent.clone();
            } else {
                operator_points[op_idx] += 3; // Improved incumbent
            }
        } else {
            // SA acceptance for worse solutions
            let p = (-delta / temp).exp();
            if rng.random::<f64>() < p {
                incumbent = new_solution;
                incumbent_cost = new_cost;
                operator_points[op_idx] += 1;
            }
        }
        
        // Update weights periodically
        if (iter + 1) % segment_size == 0 {
            for i in 0..n_ops {
                if operator_use_counts[i] > 0 {
                    let score = operator_points[i] as f64 / operator_use_counts[i] as f64;
                    weights[i] = (weights[i] * (1.0 - r) + r * score).max(0.05);
                }
            }
            
            // Normalize
            let sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }
            
            dist = WeightedIndex::new(&weights).unwrap();
            operator_points.fill(0);
            operator_use_counts.fill(0);
        }
        
        // Log
        let weights_str = weights.iter().map(|w| format!("{:.4}", w)).collect::<Vec<_>>().join(",");
        writeln!(writer, "{},{:.4},{:.4},{:.6},{}", iter, best_cost, incumbent_cost, temp, weights_str).unwrap();
        
        temp *= alpha;
    }
    
    println!("Final best: {:.2}", best_cost);
    println!("Final weights: {:?}", weights);
    best_sol
}

/// Escape by applying operator multiple times, accepting any feasible move
fn escape(
    instance: &TruckAndDroneInstance,
    solution: &Solution,
    operators: &[&dyn Operator],
    dist: &mut WeightedIndex<f64>,
    rng: &mut impl Rng,
    escape_iterations: usize,
) -> Solution {
    let mut current = solution.clone();
    let best_cost = current.verify_and_cost(instance)
        .map(|r| r.total_time).unwrap_or(f64::INFINITY);
    
    for _ in 0..escape_iterations {
        let op_idx = dist.sample(rng);
        let new = match operators[op_idx].apply(&current, instance) {
            Some(sol) => sol,
            None => continue,
        };
        
        let cost = match new.verify_and_cost(instance) {
            Ok(result) => result.total_time,
            Err(_) => continue,
        };
        
        current = new;
        
        if cost < best_cost {
            break;
        }
    }
    
    current
}
