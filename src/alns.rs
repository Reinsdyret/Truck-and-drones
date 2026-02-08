use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;

use crate::{Solution, TruckAndDroneInstance};
use crate::operators::{Operator, WildChange};

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
    stop_flag: Option<Arc<AtomicBool>>,
) -> Solution {
    alns(init_solution, instance, operators, &[], time_limit, stop_flag)
}

/// ALNS with SA acceptance, adaptive weights, and configurable escape operators
pub fn alns_with_escape(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
    escape_operators: &[&(dyn Operator + Sync)],
    time_limit: Duration,
    stop_flag: Option<Arc<AtomicBool>>,
) -> Solution {
    alns(init_solution, instance, operators, escape_operators, time_limit, stop_flag)
}

/// ALNS with SA acceptance and adaptive weights
pub fn alns(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
    escape_operators: &[&(dyn Operator + Sync)],
    time_limit: Duration,
    stop_flag: Option<Arc<AtomicBool>>,
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
    let mut operator_points = vec![0.0_f64; n_ops];
    let mut seen_solutions: HashSet<String> = HashSet::new();
    seen_solutions.insert(incumbent.to_string());
    
    let gamma = 0.2; // Weight adjustment factor (ALNS literature)
    let min_weight = 0.05;
    
    // Record-to-Record Travel: no temperature needed
    
    // Escape: use escape_operators if provided, otherwise fall back to WildChange
    let shake = WildChange::new(2, 5);
    let has_escape_ops = !escape_operators.is_empty();
    let escape_interval = Duration::from_secs(30);
    
    // let segment_duration = Duration::from_secs_f64(
    //     (time_limit.as_secs_f64() / 100.0).max(1.0),
    // );
    let segment_duration = Duration::from_secs(20);
    let log_interval = Duration::from_secs(1);
    let status_interval = Duration::from_secs(10);
    
    // CSV logging
    let file = File::create("alns_scores.csv").expect("failed to create csv");
    let mut writer = BufWriter::new(file);
    let header = format!("elapsed_secs,iter,best,incumbent,deviation,{}", 
        (0..n_ops).map(|i| format!("w{}", i)).collect::<Vec<_>>().join(","));
    writeln!(writer, "{}", header).unwrap();

    let start = Instant::now();
    let mut last_segment = Instant::now();
    let mut last_log = Instant::now();
    let mut last_status = Instant::now();
    let mut last_improvement = Instant::now();
    let mut iter: usize = 0;
    
    let should_stop = || {
        stop_flag
            .as_ref()
            .map(|flag| flag.load(Ordering::SeqCst))
            .unwrap_or(false)
    };
    while start.elapsed() < time_limit && !should_stop() {
        iter += 1;
        
        if last_status.elapsed() >= status_interval {
            let d = 0.2 * ((time_limit.as_secs_f64() - start.elapsed().as_secs_f64()) / time_limit.as_secs_f64()) * best_cost;
            println!(
                "[{:.0}s] Iter {}: best = {:.2}, incumbent = {:.2}, deviation = {:.2}, weights = {:?}",
                start.elapsed().as_secs_f64(),
                iter,
                best_cost,
                incumbent_cost,
                d,
                weights
            );
            last_status = Instant::now();
        }
        
        // Escape if stuck for long (time-based)
        if last_improvement.elapsed() >= escape_interval {
            if has_escape_ops {
                // Pick a random escape operator
                let esc_idx = rng.random_range(0..escape_operators.len());
                if let Some(escaped) = escape_operators[esc_idx].apply(&best_sol, instance) {
                    if let Ok(result) = escaped.verify_and_cost(instance) {
                        println!(
                            "[{:.0}s] ESCAPE via op {}: {:.2} -> {:.2}",
                            start.elapsed().as_secs_f64(), esc_idx, incumbent_cost, result.total_time
                        );
                        incumbent = escaped;
                        incumbent_cost = result.total_time;
                        if incumbent_cost < best_cost {
                            best_sol = incumbent.clone();
                            best_cost = incumbent_cost;
                        }
                    }
                }
            } else {
                // Fallback: WildChange shake
                if let Some(shaken) = shake.apply(&incumbent, instance) {
                    if let Ok(result) = shaken.verify_and_cost(instance) {
                        incumbent = shaken;
                        incumbent_cost = result.total_time;
                    }
                }
            }
            last_improvement = Instant::now();
        }
        
        // Choose and apply operator
        let op_idx = dist.sample(&mut rng);
        operator_use_counts[op_idx] += 1;
        
        let new_solution = match operators[op_idx].apply(&incumbent, instance) {
            Some(sol) => sol,
            None => continue,
        };
        
        let new_cost = match new_solution.verify_and_cost(instance) {
            Ok(result) => result.total_time,
            Err(_) => continue,
        };
        let is_new_solution = seen_solutions.insert(new_solution.to_string());
        
        // Record-to-Record Travel acceptance
        let remaining = (time_limit.as_secs_f64() - start.elapsed().as_secs_f64()).max(0.0);
        let d = 0.2 * (remaining / time_limit.as_secs_f64()) * best_cost;
        
        if new_cost < incumbent_cost {
            // Improving move - always accept
            incumbent = new_solution.clone();
            incumbent_cost = new_cost;

            if incumbent_cost < best_cost {
                operator_points[op_idx] += 5.0;
                best_cost = incumbent_cost;
                best_sol = incumbent.clone();
                last_improvement = Instant::now();
            } else {
                operator_points[op_idx] += 3.0;
                last_improvement = Instant::now();
            }
        } else if new_cost < best_cost + d {
            // Within allowed deviation from record - accept
            incumbent = new_solution.clone();
            incumbent_cost = new_cost;
            if is_new_solution {
                operator_points[op_idx] += 1.0;
            }
        } else if is_new_solution {
            operator_points[op_idx] += 1.0;
        }
        
        // Update weights periodically
        if last_segment.elapsed() >= segment_duration {
            for i in 0..n_ops {
                if operator_use_counts[i] > 0 {
                    let score = operator_points[i] / operator_use_counts[i] as f64;
                    weights[i] = (1.0 - gamma) * weights[i] + gamma * score;
                }
                weights[i] = weights[i].max(min_weight);
            }
            
            let sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }
            
            dist = WeightedIndex::new(&weights).unwrap();
            operator_points.fill(0.0);
            operator_use_counts.fill(0);
            last_segment = Instant::now();
        }
        
        // Log
        if last_log.elapsed() >= log_interval {
            let weights_str = weights.iter().map(|w| format!("{:.4}", w)).collect::<Vec<_>>().join(",");
            writeln!(
                writer,
                "{:.2},{},{:.4},{:.4},{:.6},{}",
                start.elapsed().as_secs_f64(),
                iter,
                best_cost,
                incumbent_cost,
                d,
                weights_str
            ).unwrap();
            last_log = Instant::now();
        }
    }
    
    let stopped_early = should_stop();
    if stopped_early {
        println!("\n>>> STOPPED EARLY by user request");
        println!("Current best solution: {}", best_sol);
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
