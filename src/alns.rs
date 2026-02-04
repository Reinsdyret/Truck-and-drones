use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;

use crate::{Solution, TruckAndDroneInstance};
use crate::operators::Operator;

/// Parallel ALNS: multiple chains with periodic synchronization
pub fn run_parallel_alns(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
    duration: Duration,
    num_chains: usize,
    sync_interval: Duration,
    stop_flag: Option<Arc<AtomicBool>>,
) -> Solution {
    run_parallel_alns_with_status(
        init_solution,
        instance,
        operators,
        duration,
        num_chains,
        sync_interval,
        Duration::from_secs(10),  // Log to CSV every 10 seconds
        stop_flag,
    )
}

/// Parallel ALNS with configurable status interval
pub fn run_parallel_alns_with_status(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
    duration: Duration,
    num_chains: usize,
    sync_interval: Duration,
    status_interval: Duration,
    stop_flag: Option<Arc<AtomicBool>>,
) -> Solution {
    let init_cost = init_solution.cost(instance).unwrap_or(f64::INFINITY);
    println!("Starting Parallel ALNS with {} chains", num_chains);
    println!("Initial cost: {:.2}", init_cost);
    println!("Sync interval: {:?}, Status interval: {:?}", sync_interval, status_interval);
    
    // Shared state
    let global_best: Arc<Mutex<(Solution, f64)>> = Arc::new(Mutex::new((init_solution.clone(), init_cost)));
    let total_iterations = Arc::new(AtomicU64::new(0));
    let improvements = Arc::new(AtomicU64::new(0));
    let stop = stop_flag.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    
    // Per-operator stats
    let op_uses: Arc<Vec<AtomicU64>> = Arc::new((0..operators.len()).map(|_| AtomicU64::new(0)).collect());
    let op_improvements: Arc<Vec<AtomicU64>> = Arc::new((0..operators.len()).map(|_| AtomicU64::new(0)).collect());
    
    let start = Instant::now();
    
    // Spawn threads using scoped threads (allows borrowing non-static data)
    thread::scope(|s| {
        // Monitor thread
        let global_best_monitor = Arc::clone(&global_best);
        let total_iters_monitor = Arc::clone(&total_iterations);
        let improvements_monitor = Arc::clone(&improvements);
        let stop_monitor = Arc::clone(&stop);
        let op_uses_monitor = Arc::clone(&op_uses);
        let op_improvements_monitor = Arc::clone(&op_improvements);
        let n_ops = operators.len();
        
        s.spawn(move || {
            // Create CSV file for logging
            let csv_file = File::create("parallel_alns_scores.csv").expect("Failed to create CSV");
            let mut csv_writer = BufWriter::new(csv_file);
            writeln!(csv_writer, "seconds,iterations,best_score,improvements,iters_per_sec").unwrap();
            
            let mut last_status = Instant::now();
            while start.elapsed() < duration && !stop_monitor.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(100));
                
                if last_status.elapsed() >= status_interval {
                    let (_, best_cost) = &*global_best_monitor.lock().unwrap();
                    let iters = total_iters_monitor.load(Ordering::Relaxed);
                    let imprs = improvements_monitor.load(Ordering::Relaxed);
                    let elapsed = start.elapsed().as_secs();
                    let iters_per_sec = if elapsed > 0 { iters / elapsed } else { 0 };
                    
                    // Write to CSV
                    writeln!(csv_writer, "{},{},{:.4},{},{}", 
                        elapsed, iters, best_cost, imprs, iters_per_sec).unwrap();
                    csv_writer.flush().unwrap();
                    
                    println!("\n=== ALNS Status @ {}s ===", elapsed);
                    println!("Global best: {:.2}", best_cost);
                    println!("Total iterations: {} ({}/sec)", iters, iters_per_sec);
                    println!("Global improvements: {}", imprs);
                    
                    // Operator stats
                    print!("Operators: ");
                    for i in 0..n_ops {
                        let uses = op_uses_monitor[i].load(Ordering::Relaxed);
                        let impr = op_improvements_monitor[i].load(Ordering::Relaxed);
                        let rate = if uses > 0 { impr as f64 / uses as f64 * 100.0 } else { 0.0 };
                        print!("[Op{}: {}/{} ({:.1}%)] ", i, impr, uses, rate);
                    }
                    println!("\n");
                    
                    last_status = Instant::now();
                }
            }
            
            // Final CSV entry
            let (_, best_cost) = &*global_best_monitor.lock().unwrap();
            let iters = total_iters_monitor.load(Ordering::Relaxed);
            let imprs = improvements_monitor.load(Ordering::Relaxed);
            let elapsed = start.elapsed().as_secs();
            let iters_per_sec = if elapsed > 0 { iters / elapsed } else { 0 };
            writeln!(csv_writer, "{},{},{:.4},{},{}", 
                elapsed, iters, best_cost, imprs, iters_per_sec).unwrap();
        });
        
        // Worker chains
        for chain_id in 0..num_chains {
            let global_best_chain = Arc::clone(&global_best);
            let total_iters_chain = Arc::clone(&total_iterations);
            let improvements_chain = Arc::clone(&improvements);
            let stop_chain = Arc::clone(&stop);
            let op_uses_chain = Arc::clone(&op_uses);
            let op_improvements_chain = Arc::clone(&op_improvements);
            
            s.spawn(move || {
                run_alns_chain(
                    chain_id,
                    init_solution,
                    instance,
                    operators,
                    duration,
                    sync_interval,
                    global_best_chain,
                    total_iters_chain,
                    improvements_chain,
                    stop_chain,
                    op_uses_chain,
                    op_improvements_chain,
                );
            });
        }
    });
    
    let (best_sol, best_cost) = {
        let guard = global_best.lock().unwrap();
        (guard.0.clone(), guard.1)
    };
    
    println!("\n=== Parallel ALNS Complete ===");
    println!("Final best: {:.2}", best_cost);
    println!("Total iterations: {}", total_iterations.load(Ordering::Relaxed));
    
    best_sol
}

/// Individual ALNS chain with periodic sync - LEAN VERSION
fn run_alns_chain(
    chain_id: usize,
    _init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
    duration: Duration,
    sync_interval: Duration,
    global_best: Arc<Mutex<(Solution, f64)>>,
    total_iterations: Arc<AtomicU64>,
    improvements: Arc<AtomicU64>,
    stop_flag: Arc<AtomicBool>,
    op_uses: Arc<Vec<AtomicU64>>,
    op_improvements: Arc<Vec<AtomicU64>>,
) {
    let mut rng = rand::rng();
    let n_ops = operators.len();
    
    // Initialize from global best
    let (mut incumbent, mut incumbent_cost) = {
        let guard = global_best.lock().unwrap();
        (guard.0.clone(), guard.1)
    };
    let mut best_solution = incumbent.clone();
    let mut best_cost = incumbent_cost;
    
    // Simple adaptive weights (ALNS-style)
    let mut weights: Vec<f64> = vec![1.0; n_ops];
    let mut op_scores: Vec<f64> = vec![0.0; n_ops];
    let mut op_counts: Vec<usize> = vec![0; n_ops];
    let min_weight = 0.1;
    
    // Warmup: estimate average delta to calibrate temperature (like SA does)
    let mut deltas: Vec<f64> = Vec::new();
    for _ in 0..500 {
        let op_idx = rng.random_range(0..n_ops);
        if let Some(new) = operators[op_idx].apply(&incumbent, instance) {
            if let Ok(result) = new.verify_and_cost(instance) {
                let delta = result.total_time - incumbent_cost;
                deltas.push(delta.abs());
                if delta < 0.0 {
                    incumbent = new;
                    incumbent_cost = result.total_time;
                    if incumbent_cost < best_cost {
                        best_cost = incumbent_cost;
                        best_solution = incumbent.clone();
                    }
                }
            }
        }
    }
    let delta_avg = if deltas.is_empty() { best_cost * 0.02 } else { 
        deltas.iter().sum::<f64>() / deltas.len() as f64 
    };
    
    // Temperature calibrated to accept 80% of moves initially
    let t_zero = if delta_avg <= 0.0 { 1.0 } else { delta_avg / (-0.8f64.ln()) };
    let mut temp = t_zero;
    let reheat_interval = 100_000;  // ~1 second at 100k iter/sec before reheat
    let mut no_improve_count = 0;
    
    let start = Instant::now();
    let mut last_sync = Instant::now();
    let mut local_iters: u64 = 0;
    
    // Local counters
    let mut local_op_uses: Vec<u64> = vec![0; n_ops];
    let mut local_op_impr: Vec<u64> = vec![0; n_ops];
    
    while start.elapsed() < duration && !stop_flag.load(Ordering::SeqCst) {
        local_iters += 1;
        
        // Flush counters every 1000 iters
        if local_iters % 100_000 == 0 {
            total_iterations.fetch_add(100_000, Ordering::Relaxed);
            for i in 0..n_ops {
                if local_op_uses[i] > 0 {
                    op_uses[i].fetch_add(local_op_uses[i], Ordering::Relaxed);
                    local_op_uses[i] = 0;
                }
                if local_op_impr[i] > 0 {
                    op_improvements[i].fetch_add(local_op_impr[i], Ordering::Relaxed);
                    local_op_impr[i] = 0;
                }
            }
        }
        
        // Update weights every 5000 iters
        if local_iters % 500_000 == 0 {
            for i in 0..n_ops {
                if op_counts[i] > 0 {
                    let perf = op_scores[i] / op_counts[i] as f64;
                    weights[i] = (weights[i] * 0.8 + perf).max(min_weight);
                }
            }
            // Quick normalize
            let sum: f64 = weights.iter().sum();
            if sum > 0.0 {
                for w in &mut weights { *w /= sum; }
            }
            op_scores.fill(0.0);
            op_counts.fill(0);
        }
        
        // Sync with global best
        if last_sync.elapsed() >= sync_interval {
            let mut global = global_best.lock().unwrap();
            
            if best_cost < global.1 {
                global.0 = best_solution.clone();
                global.1 = best_cost;
                improvements.fetch_add(1, Ordering::Relaxed);
                println!("[Chain {}] NEW GLOBAL BEST: {:.2}", chain_id, best_cost);
            }
            
            if global.1 < best_cost || rng.random::<f64>() < 0.3 {
                best_solution = global.0.clone();
                best_cost = global.1;
                incumbent = best_solution.clone();
                incumbent_cost = best_cost;
            }
            
            drop(global);
            last_sync = Instant::now();
        }
        
        // Select operator (fast weighted selection)
        let op_idx = {
            let total: f64 = weights.iter().sum();
            let mut r = rng.random::<f64>() * total;
            let mut idx = 0;
            for (i, &w) in weights.iter().enumerate() {
                r -= w;
                if r <= 0.0 { idx = i; break; }
            }
            idx
        };
        op_counts[op_idx] += 1;
        local_op_uses[op_idx] += 1;
        
        let new_solution = match operators[op_idx].apply(&incumbent, instance) {
            Some(sol) => sol,
            None => { no_improve_count += 1; continue; }
        };
        
        let new_cost = match new_solution.verify_and_cost(instance) {
            Ok(result) => result.total_time,
            Err(_) => { no_improve_count += 1; continue; }
        };
        
        let delta = new_cost - incumbent_cost;
        
        if delta < 0.0 {
            // Improvement
            local_op_impr[op_idx] += 1;
            op_scores[op_idx] += -delta;  // Track magnitude
            
            incumbent = new_solution;
            incumbent_cost = new_cost;
            no_improve_count = 0;
            
            if new_cost < best_cost {
                best_solution = incumbent.clone();
                best_cost = new_cost;
            }
        } else {
            // SA acceptance
            let p = (-delta / temp).exp();
            if rng.random::<f64>() < p {
                incumbent = new_solution;
                incumbent_cost = new_cost;
            }
            no_improve_count += 1;
        }
        
        // Adaptive cooling
        let progress = start.elapsed().as_secs_f64() / duration.as_secs_f64();
        temp = (temp * (0.99999 - 0.00009 * (1.0 - progress))).max(0.001);
        
        // Reheat if stuck - with aggressive perturbation!
        if no_improve_count >= reheat_interval {
            temp = t_zero * (0.5 + 0.5 * rng.random::<f64>());  // Higher reheat temp
            incumbent = best_solution.clone();
            incumbent_cost = best_cost;
            
            // Apply 5-10 random moves to escape the basin
            let num_perturb = 5 + rng.random_range(0..6);
            for _ in 0..num_perturb {
                let op_idx = rng.random_range(0..n_ops);
                if let Some(new) = operators[op_idx].apply(&incumbent, instance) {
                    if let Ok(result) = new.verify_and_cost(instance) {
                        incumbent = new;
                        incumbent_cost = result.total_time;
                        // Keep if it's actually better
                        if incumbent_cost < best_cost {
                            best_cost = incumbent_cost;
                            best_solution = incumbent.clone();
                        }
                    }
                }
            }
            
            no_improve_count = 0;
        }
    }
    
    // Final sync
    {
        let mut global = global_best.lock().unwrap();
        if best_cost < global.1 {
            global.0 = best_solution;
            global.1 = best_cost;
            improvements.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    total_iterations.fetch_add(local_iters % 1000, Ordering::Relaxed);
}

/// Weighted random selection
#[inline]
fn select_weighted(weights: &[f64], rng: &mut impl Rng) -> usize {
    let total: f64 = weights.iter().sum();
    let mut r = rng.random::<f64>() * total;
    
    for (i, &w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

/// ALNS with time limit
pub fn alns_timed(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
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
    operators: &[&(dyn Operator + Sync)],
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
    operators: &[&(dyn Operator + Sync)],
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
