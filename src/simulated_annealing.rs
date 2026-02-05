use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use rand::random_range;

use crate::{Solution, TruckAndDroneInstance};
use crate::operators::Operator;

/// Set up Ctrl+C handler, returns flag that becomes true when pressed
pub fn setup_stop_signal() -> Arc<AtomicBool> {
    let stop_flag = Arc::new(AtomicBool::new(false));
    let flag_clone = stop_flag.clone();
    
    ctrlc::set_handler(move || {
        println!("\n>>> Ctrl+C received, stopping gracefully...");
        flag_clone.store(true, Ordering::SeqCst);
    }).expect("Error setting Ctrl+C handler");
    
    stop_flag
}

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
            min_weight: 0.01,
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
                self.weights[i] = self.weights[i] * self.decay + performance;
            } else {
                // Decay unused operators but don't let them die
                self.weights[i] = self.weights[i] * self.decay;
            }
        }
        
        // Enforce minimum weight BEFORE normalization
        for w in &mut self.weights {
            *w = w.max(self.min_weight);
        }
        
        // Normalize weights to average 1.0
        let total: f64 = self.weights.iter().sum();
        let n = self.weights.len() as f64;
        if total > 0.0 {
            for w in &mut self.weights {
                *w = *w * n / total;
            }
        }
        
        // Enforce minimum weight AFTER normalization too
        for w in &mut self.weights {
            *w = w.max(self.min_weight);
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
    operators: &[&(dyn Operator + Sync)],
    iterations: usize,
) -> Solution {
    run_simulated_annealing_with_params(init_solution, instance, operators, 0.8, 0.01, iterations, 10000)
}

pub fn run_simulated_annealing_with_params(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
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
    operators: &[&(dyn Operator + Sync)],
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

/// Perturb a solution by applying random operators multiple times
fn perturb_solution(
    solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
    num_perturbations: usize,
    stop_flag: Option<&AtomicBool>,
) -> (Solution, f64) {
    let mut current = solution.clone();
    let mut current_score = current.verify_and_cost(instance)
        .map(|r| r.total_time)
        .unwrap_or(f64::INFINITY);
    
    for _ in 0..num_perturbations {
        if stop_flag
            .map(|flag| flag.load(Ordering::SeqCst))
            .unwrap_or(false)
        {
            break;
        }
        let op_idx = random_range(0..operators.len());
        if let Some(new_sol) = operators[op_idx].apply(&current, instance) {
            if let Ok(result) = new_sol.verify_and_cost(instance) {
                current = new_sol;
                current_score = result.total_time;
            }
        }
    }
    
    (current, current_score)
}

/// Timed version of simulated annealing with adaptive operators and perturbation on reheat
/// Pass a stop_flag from setup_stop_signal() to enable graceful Ctrl+C stopping
pub fn run_simulated_annealing_timed(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
    duration: Duration,
    stop_flag: Option<Arc<AtomicBool>>,
) -> Solution {
    let (delta_avg, mut incumbent, mut best_solution, mut incumbent_score, mut best_score) =
        estimate_avg_delta(init_solution, instance, operators, 0.8);

    // Calculate initial temperature from average delta
    let t_zero = if delta_avg <= 0.0 {
        1.0
    } else {
        delta_avg / (-0.8f64.ln())
    };
    
    let mut temp = t_zero;
    let reheat_interval = 15000;  // Iterations without improvement before reheat
    let weight_update_interval = 5000;  // Update adaptive weights periodically
    let mut no_improve_count = 0;
    let mut reheat_count = 0;
    let mut stopped_early = false;

    // Adaptive operator selection
    let mut selector = AdaptiveOperatorSelector::new(operators.len(), 0.8);

    // CSV logging
    let output_path = "simulated_annealing_timed_scores.csv";
    let file = File::create(output_path).expect("failed to create csv");
    let mut writer = BufWriter::new(file);
    
    // Header with weight columns
    let header = format!("elapsed_secs,iter,best_score,incumbent_score,temp,{}", 
        (0..operators.len()).map(|i| format!("w{}", i)).collect::<Vec<_>>().join(","));
    writeln!(writer, "{}", header).expect("failed to write header");

    let start = Instant::now();
    let mut i: u64 = 0;
    let mut last_print = Instant::now();

    println!("Starting timed SA (adaptive + perturbation) for {:?}", duration);
    println!("Press Ctrl+C to stop early and get current best solution");
    println!("Initial temp: {:.4}, best: {:.2}", t_zero, best_score);

    // Check both time and stop flag
    let should_stop = || {
        if let Some(ref flag) = stop_flag {
            if flag.load(Ordering::SeqCst) {
                return true;
            }
        }
        false
    };

    while start.elapsed() < duration && !should_stop() {
        i += 1;
        
        // Progress print every 10 seconds
        if last_print.elapsed() >= Duration::from_secs(10) {
            let elapsed = start.elapsed().as_secs_f64();
            let remaining = duration.as_secs_f64() - elapsed;
            println!(
                "[{:.0}s / {:.0}s remaining] Iter {}: best = {:.2}, incumbent = {:.2}, temp = {:.4}, weights = {:?}",
                elapsed, remaining, i, best_score, incumbent_score, temp, selector.get_weights()
            );
            last_print = Instant::now();
        }
        
        // Adaptive operator selection
        let op_idx = selector.select();
        selector.record_use(op_idx);
        let operator = operators[op_idx];
        
        let new_solution = match operator.apply(&incumbent, instance) {
            Some(sol) => sol,
            None => {
                no_improve_count += 1;
                continue;
            }
        };
        
        // Check feasibility and get score
        let new_score = match new_solution.verify_and_cost(instance) {
            Ok(result) => result.total_time,
            Err(_) => {
                no_improve_count += 1;
                continue;
            }
        };

        // Delta (negative = improvement since we minimize)
        let delta = new_score - incumbent_score;

        if delta < 0.0 {
            // Record improvement for adaptive weights
            selector.record_improvement(op_idx, -delta);
            
            // Better solution - accept
            incumbent = new_solution;
            incumbent_score = new_score;
            no_improve_count = 0;
            if new_score < best_score {
                best_solution = incumbent.clone();
                best_score = new_score;
                println!(
                    "[{:.0}s] NEW BEST: {:.2} (iter {}, op {})",
                    start.elapsed().as_secs_f64(), best_score, i, op_idx
                );
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

        // Adaptive cooling: slow down as we get closer to end
        let progress = start.elapsed().as_secs_f64() / duration.as_secs_f64();
        let alpha = 0.99999 - 0.00009 * (1.0 - progress);
        temp *= alpha;
        temp = temp.max(0.001);

        // Update adaptive weights periodically
        if i % weight_update_interval as u64 == 0 {
            selector.update_weights();
        }

        // Reheat with perturbation if stuck
        if no_improve_count >= reheat_interval {
            reheat_count += 1;
            
            // Alternate between different reheat strategies
            let reheat_temp = match reheat_count % 3 {
                0 => t_zero * 0.7,  // High reheat
                1 => t_zero * 0.5,  // Medium reheat  
                _ => t_zero * 0.3,  // Low reheat
            };
            temp = reheat_temp;
            
            // Perturb the best solution instead of just copying it
            let num_perturbations = 5 + (reheat_count % 10);  // Vary perturbation strength
            let (perturbed, perturbed_score) = perturb_solution(
                &best_solution,
                instance,
                operators,
                num_perturbations,
                stop_flag.as_deref(),
            );
            
            incumbent = perturbed;
            incumbent_score = perturbed_score;
            no_improve_count = 0;
            
            println!(
                "[{:.0}s] REHEAT #{} at iter {}: temp={:.4}, perturbed with {} moves, score {:.2} -> {:.2}",
                start.elapsed().as_secs_f64(), reheat_count, i, reheat_temp, num_perturbations, best_score, incumbent_score
            );
        }

        // Log every 1000 iterations
        if i % 1000 == 0 {
            let weights_str = selector.get_weights().iter()
                .map(|w| format!("{:.4}", w))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(
                writer,
                "{:.2},{},{:.4},{:.4},{:.6},{}",
                start.elapsed().as_secs_f64(), i, best_score, incumbent_score, temp, weights_str
            ).expect("failed to write row");
        }
    }

    // Check if we stopped early
    stopped_early = should_stop();

    let total_secs = start.elapsed().as_secs_f64();
    if stopped_early {
        println!("\n>>> STOPPED EARLY by user request");
    }
    println!("\nCompleted {} iterations in {:.1}s ({:.0} iter/sec)", i, total_secs, i as f64 / total_secs);
    println!("Total reheats: {}", reheat_count);
    println!("Final weights: {:?}", selector.get_weights());
    println!("Final best score: {:.2}", best_score);
    best_solution
}

/// Estimate average delta from initial random walk to set initial temperature
fn estimate_avg_delta(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
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

/// Parallel SA with multiple synchronized chains
/// Runs N chains in parallel, synchronizing every sync_interval to share best solutions
/// One thread monitors and prints status every status_interval
pub fn run_parallel_sa(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
    duration: Duration,
    num_chains: usize,
    sync_interval: Duration,
    stop_flag: Option<Arc<AtomicBool>>,
) -> Solution {
    run_parallel_sa_with_status(
        init_solution,
        instance,
        operators,
        duration,
        num_chains,
        sync_interval,
        Duration::from_secs(100), // Default status interval
        stop_flag,
    )
}

/// Parallel SA with configurable status update interval
pub fn run_parallel_sa_with_status(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
    duration: Duration,
    num_chains: usize,
    sync_interval: Duration,
    status_interval: Duration,
    stop_flag: Option<Arc<AtomicBool>>,
) -> Solution {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicU64;
    use std::thread;
    
    println!("===========================================");
    println!("  Parallel SA: {} worker chains", num_chains);
    println!("  Sync interval: {:?}", sync_interval);
    println!("  Status updates: every {:?}", status_interval);
    println!("  Total duration: {:?}", duration);
    println!("  Press Ctrl+C to stop early");
    println!("===========================================\n");
    
    // Shared state
    let global_best: Arc<Mutex<(Solution, f64)>> = Arc::new(Mutex::new((
        init_solution.clone(),
        init_solution.verify_and_cost(instance).map(|r| r.total_time).unwrap_or(f64::INFINITY)
    )));
    let total_iterations: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let improvements: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    
    // Per-operator statistics
    let num_ops = operators.len();
    let op_uses: Arc<Vec<AtomicU64>> = Arc::new((0..num_ops).map(|_| AtomicU64::new(0)).collect());
    let op_improvements: Arc<Vec<AtomicU64>> = Arc::new((0..num_ops).map(|_| AtomicU64::new(0)).collect());
    
    let start = Instant::now();
    let stop_flag = stop_flag.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    
    // Use scoped threads
    thread::scope(|s| {
        // Spawn monitoring thread
        let global_best_mon = global_best.clone();
        let stop_flag_mon = stop_flag.clone();
        let total_iters_mon = total_iterations.clone();
        let improvements_mon = improvements.clone();
        let op_uses_mon = op_uses.clone();
        let op_improvements_mon = op_improvements.clone();
        
        s.spawn(move || {
            let mut last_best = f64::INFINITY;
            
            while start.elapsed() < duration && !stop_flag_mon.load(Ordering::SeqCst) {
                thread::sleep(status_interval);
                
                if stop_flag_mon.load(Ordering::SeqCst) {
                    break;
                }
                
                let elapsed = start.elapsed().as_secs_f64();
                let remaining = duration.as_secs_f64() - elapsed;
                let iters = total_iters_mon.load(Ordering::Relaxed);
                let impr = improvements_mon.load(Ordering::Relaxed);
                let current_best = global_best_mon.lock().unwrap().1;
                
                let improved_marker = if current_best < last_best { " *** IMPROVED ***" } else { "" };
                last_best = current_best;
                
                // Collect operator stats
                let op_stats: Vec<(u64, u64)> = op_uses_mon.iter()
                    .zip(op_improvements_mon.iter())
                    .map(|(u, i)| (u.load(Ordering::Relaxed), i.load(Ordering::Relaxed)))
                    .collect();
                let mut op_weights: Vec<f64> = op_stats
                    .iter()
                    .map(|(uses, imps)| {
                        if *uses > 0 {
                            (*imps as f64 / *uses as f64).max(0.01)
                        } else {
                            0.01
                        }
                    })
                    .collect();
                let weight_sum: f64 = op_weights.iter().sum();
                if weight_sum > 0.0 {
                    for w in &mut op_weights {
                        *w /= weight_sum;
                    }
                }
                
                println!("\n╔══════════════════════════════════════════════════╗");
                println!("║  STATUS UPDATE @ {:.0}s ({:.0}s remaining)", elapsed, remaining);
                println!("╠══════════════════════════════════════════════════╣");
                println!("║  Global best: {:.2}{}", current_best, improved_marker);
                println!("║  Total iterations: {} ({:.0}/sec)", iters, iters as f64 / elapsed);
                println!("║  Improvements found: {}", impr);
                println!("╠──────────────────────────────────────────────────╣");
                println!("║  Operator Stats (uses / improvements / rate):");
                for (i, (uses, imps)) in op_stats.iter().enumerate() {
                    let rate = if *uses > 0 { *imps as f64 / *uses as f64 * 100.0 } else { 0.0 };
                    println!("║    Op {}: {:>8} / {:>6} / {:>5.2}%", i, uses, imps, rate);
                }
                println!("║  Operator Weights: {:?}", op_weights);
                println!("╚══════════════════════════════════════════════════╝\n");
            }
        });
        
        // Spawn worker chains
        for chain_id in 0..num_chains {
            let global_best = global_best.clone();
            let stop_flag = stop_flag.clone();
            let init_sol = init_solution.clone();
            let total_iters = total_iterations.clone();
            let impr_counter = improvements.clone();
            let op_uses_chain = op_uses.clone();
            let op_impr_chain = op_improvements.clone();
            
            s.spawn(move || {
                run_sa_chain_with_counters(
                    chain_id,
                    &init_sol,
                    instance,
                    operators,
                    duration,
                    sync_interval,
                    global_best,
                    stop_flag,
                    total_iters,
                    impr_counter,
                    op_uses_chain,
                    op_impr_chain,
                )
            });
        }
    });
    
    let (best_sol, best_score) = global_best.lock().unwrap().clone();
    let total_iters = total_iterations.load(Ordering::Relaxed);
    let total_impr = improvements.load(Ordering::Relaxed);
    
    println!("\n╔══════════════════════════════════════════╗");
    println!("║        PARALLEL SA COMPLETE              ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Total time: {:.1}s", start.elapsed().as_secs_f64());
    println!("║  Total iterations: {}", total_iters);
    println!("║  Iterations/sec: {:.0}", total_iters as f64 / start.elapsed().as_secs_f64());
    println!("║  Total improvements: {}", total_impr);
    println!("║  FINAL BEST: {:.2}", best_score);
    println!("╚══════════════════════════════════════════╝");
    
    best_sol
}

/// SA chain with shared counters for monitoring
fn run_sa_chain_with_counters(
    chain_id: usize,
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
    duration: Duration,
    sync_interval: Duration,
    global_best: Arc<std::sync::Mutex<(Solution, f64)>>,
    stop_flag: Arc<AtomicBool>,
    total_iterations: Arc<std::sync::atomic::AtomicU64>,
    improvements: Arc<std::sync::atomic::AtomicU64>,
    op_uses: Arc<Vec<std::sync::atomic::AtomicU64>>,
    op_improvements: Arc<Vec<std::sync::atomic::AtomicU64>>,
) {
    // Initialize chain
    let (delta_avg, mut incumbent, mut best_solution, mut incumbent_score, mut best_score) =
        estimate_avg_delta(init_solution, instance, operators, 0.8);
    
    let t_zero = if delta_avg <= 0.0 { 1.0 } else { delta_avg / (-0.8f64.ln()) };
    let mut temp = t_zero;
    let reheat_interval = 1_000_000;
    let mut no_improve_count = 0;
    
    let start = Instant::now();
    let mut last_sync = Instant::now();
    let mut local_iters: u64 = 0;
    let mut last_improvement = Instant::now();
    
    // Adaptive operator selection (ALNS-style)
    let mut selector = AdaptiveOperatorSelector::new(operators.len(), 0.8);
    let weight_update_interval = 50_000;
    let single_op = operators.len() == 1;
    let track_seen = operators.len() > 1;
    let mut seen_solutions: HashSet<String> = HashSet::new();
    if track_seen {
        seen_solutions.insert(incumbent.to_string());
    }
    
    // Local counters to reduce atomic contention
    let mut local_op_uses: Vec<u64> = vec![0; operators.len()];
    let mut local_op_impr: Vec<u64> = vec![0; operators.len()];
    
    while start.elapsed() < duration && !stop_flag.load(Ordering::SeqCst) {
        local_iters += 1;
        
        // Update global counters periodically (every 1000 iters to reduce contention)
        if local_iters % 1000 == 0 {
            total_iterations.fetch_add(1000, Ordering::Relaxed);
            // Flush local operator stats
            for i in 0..operators.len() {
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
        
        // Update adaptive weights periodically
        if !single_op && local_iters % weight_update_interval as u64 == 0 {
            selector.update_weights();
        }
        
        // Periodic sync with global best
        if last_sync.elapsed() >= sync_interval {
            let mut global = global_best.lock().unwrap();
            
            if best_score < global.1 {
                global.0 = best_solution.clone();
                global.1 = best_score;
                improvements.fetch_add(1, Ordering::Relaxed);
                println!("[Chain {}] NEW GLOBAL BEST: {:.2}", chain_id, best_score);
            }
            
            if global.1 < best_score || rand::random::<f64>() < 0.3 {
                best_solution = global.0.clone();
                best_score = global.1;
                incumbent = best_solution.clone();
                incumbent_score = best_score;
            }
            
            drop(global);
            last_sync = Instant::now();
        }
        
        // Adaptive operator selection (weighted by performance)
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }
        let op_idx = if single_op { 0 } else { selector.select() };
        if !single_op {
            selector.record_use(op_idx);
        }
        local_op_uses[op_idx] += 1;
        let operator = operators[op_idx];
        
        let new_solution = match operator.apply(&incumbent, instance) {
            Some(sol) => sol,
            None => {
                no_improve_count += 1;
                continue;
            }
        };
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }
        
        let new_score = match new_solution.verify_and_cost(instance) {
            Ok(result) => result.total_time,
            Err(_) => {
                no_improve_count += 1;
                continue;
            }
        };
        
        let delta = new_score - incumbent_score;
        let is_new_solution = if track_seen {
            seen_solutions.insert(new_solution.to_string())
        } else {
            false
        };
        let mut points = 0.0;
        if new_score < best_score {
            let global_best_score = global_best.lock().unwrap().1;
            if new_score < global_best_score {
                points = 5.0;
            } else {
                points = 3.0;
            }
        } else if new_score < incumbent_score {
            points = 3.0;
        } else if is_new_solution {
            points = 1.0;
        }
        if points > 0.0 {
            local_op_impr[op_idx] += points as u64;
            selector.record_improvement(op_idx, points);
        }
        
        if delta < 0.0 {
            incumbent = new_solution;
            incumbent_score = new_score;
            no_improve_count = 0;
            if new_score < best_score {
                best_solution = incumbent.clone();
                best_score = new_score;
                last_improvement = Instant::now();
            }
        } else {
            let p = (-delta / temp).exp();
            if rand::random::<f64>() < p {
                incumbent = new_solution;
                incumbent_score = new_score;
            }
            no_improve_count += 1;
        }
        
        // Adaptive cooling
        let progress = start.elapsed().as_secs_f64() / duration.as_secs_f64();
        let alpha = 0.99999 - 0.00009 * (1.0 - progress);
        temp = (temp * alpha).max(0.001);
        
        // Reheat if stuck
        if no_improve_count >= reheat_interval {
            temp = t_zero * (0.3 + 0.4 * rand::random::<f64>());
            incumbent = best_solution.clone();
            incumbent_score = best_score;
            no_improve_count = 0;
        }

    }
    
    // Final sync
    {
        let mut global = global_best.lock().unwrap();
        if best_score < global.1 {
            global.0 = best_solution;
            global.1 = best_score;
            improvements.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    // Final iteration count update
    total_iterations.fetch_add(local_iters % 1000, Ordering::Relaxed);
}

/// Individual SA chain that periodically syncs with global best
fn run_sa_chain(
    chain_id: usize,
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    operators: &[&(dyn Operator + Sync)],
    duration: Duration,
    sync_interval: Duration,
    global_best: Arc<std::sync::Mutex<(Solution, f64)>>,
    stop_flag: Arc<AtomicBool>,
) {
    
    // Initialize chain
    let (delta_avg, mut incumbent, mut best_solution, mut incumbent_score, mut best_score) =
        estimate_avg_delta(init_solution, instance, operators, 0.8);
    
    let t_zero = if delta_avg <= 0.0 { 1.0 } else { delta_avg / (-0.8f64.ln()) };
    let mut temp = t_zero;
    let reheat_interval = 15000;
    let mut no_improve_count = 0;
    let mut reheat_count = 0;
    
    let start = Instant::now();
    let mut last_sync = Instant::now();
    let mut iter: u64 = 0;
    
    // Main SA loop
    while start.elapsed() < duration && !stop_flag.load(Ordering::SeqCst) {
        iter += 1;
        
        // === Periodic sync with global best ===
        if last_sync.elapsed() >= sync_interval {
            let mut global = global_best.lock().unwrap();
            
            // Update global if we have better
            if best_score < global.1 {
                global.0 = best_solution.clone();
                global.1 = best_score;
                println!(
                    "[Chain {}] {:.0}s: NEW GLOBAL BEST {:.2}",
                    chain_id, start.elapsed().as_secs_f64(), best_score
                );
            }
            
            // Adopt global best if it's better than ours (with some probability)
            if global.1 < best_score || rand::random::<f64>() < 0.3 {
                best_solution = global.0.clone();
                best_score = global.1;
                // Restart from global best with perturbation
                incumbent = best_solution.clone();
                incumbent_score = best_score;
            }
            
            drop(global);
            last_sync = Instant::now();
        }
        
        // Pick random operator
        let op_idx = random_range(0..operators.len());
        let operator = operators[op_idx];
        
        let new_solution = match operator.apply(&incumbent, instance) {
            Some(sol) => sol,
            None => {
                no_improve_count += 1;
                continue;
            }
        };
        
        let new_score = match new_solution.verify_and_cost(instance) {
            Ok(result) => result.total_time,
            Err(_) => {
                no_improve_count += 1;
                continue;
            }
        };
        
        let delta = new_score - incumbent_score;
        
        if delta < 0.0 {
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
        
        // Adaptive cooling
        let progress = start.elapsed().as_secs_f64() / duration.as_secs_f64();
        let alpha = 0.99999 - 0.00009 * (1.0 - progress);
        temp = (temp * alpha).max(0.001);
        
        // Reheat if stuck
        if no_improve_count >= reheat_interval {
            reheat_count += 1;
            temp = t_zero * (0.3 + 0.4 * rand::random::<f64>()); // Random reheat level
            incumbent = best_solution.clone();
            incumbent_score = best_score;
            no_improve_count = 0;
        }
    }
    
    // Final sync
    {
        let mut global = global_best.lock().unwrap();
        if best_score < global.1 {
            global.0 = best_solution;
            global.1 = best_score;
        }
    }
    
    println!(
        "[Chain {}] Done: {} iters, {} reheats, best = {:.2}",
        chain_id, iter, reheat_count, best_score
    );
}
