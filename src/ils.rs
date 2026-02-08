use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use rand::Rng;
use rayon::prelude::*;

use crate::{Solution, TruckAndDroneInstance};
use crate::operators::Operator;

/// Run parallel Iterated Local Search using rayon.
///
/// Each thread runs its own ILS loop:
///   1. Local search with fast operators (accept only improvements)
///   2. When stuck, perturbation via DestroyRepair from best known
///   3. Sync best solution across threads periodically
pub fn run_parallel_ils(
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    local_search_ops: &[&(dyn Operator + Sync)],
    perturbation_ops: &[&(dyn Operator + Sync)],
    duration: Duration,
    num_chains: usize,
    sync_interval: Duration,
    stop_flag: Option<Arc<AtomicBool>>,
) -> Solution {
    // Configure rayon to use the requested number of chains
    // (+ 1 for the monitor, handled via std::thread)
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_chains)
        .build_global()
        .ok(); // Ignore if already initialized

    let init_score = init_solution.verify_and_cost(instance)
        .map(|r| r.total_time).unwrap_or(f64::INFINITY);

    println!("===========================================");
    println!("  Parallel ILS (rayon): {} worker chains", num_chains);
    println!("  Local search ops: {}", local_search_ops.len());
    println!("  Perturbation ops: {}", perturbation_ops.len());
    println!("  Sync interval: {:?}", sync_interval);
    println!("  Total duration: {:?}", duration);
    println!("  Initial score: {:.2}", init_score);
    println!("  Press Ctrl+C to stop early");
    println!("===========================================\n");

    // Shared state
    let global_best: Arc<Mutex<(Solution, f64)>> = Arc::new(Mutex::new((
        init_solution.clone(), init_score,
    )));
    let total_iterations: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let improvements: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let perturbations: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));

    // Per-operator statistics
    let num_ls_ops = local_search_ops.len();
    let op_uses: Arc<Vec<AtomicU64>> = Arc::new((0..num_ls_ops).map(|_| AtomicU64::new(0)).collect());
    let op_improvements: Arc<Vec<AtomicU64>> = Arc::new((0..num_ls_ops).map(|_| AtomicU64::new(0)).collect());

    // CSV logging
    let log_file = File::create("ils_sync_scores.csv").expect("failed to create ils_sync_scores.csv");
    let sync_logger: Arc<Mutex<BufWriter<File>>> = Arc::new(Mutex::new(BufWriter::new(log_file)));
    {
        let mut writer = sync_logger.lock().unwrap();
        writeln!(writer, "iter,best_score,incumbent_score,elapsed_secs,chain_id,local_iters,global_best_score")
            .expect("failed to write header");
    }

    let start = Instant::now();
    let stop_flag = stop_flag.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

    // Spawn monitor thread via std::thread (it just sleeps + prints)
    let global_best_mon = global_best.clone();
    let stop_flag_mon = stop_flag.clone();
    let total_iters_mon = total_iterations.clone();
    let improvements_mon = improvements.clone();
    let perturbations_mon = perturbations.clone();
    let op_uses_mon = op_uses.clone();
    let op_improvements_mon = op_improvements.clone();

    let monitor = std::thread::spawn(move || {
        let status_interval = Duration::from_secs(60);
        let mut last_best = f64::INFINITY;

        while start.elapsed() < duration && !stop_flag_mon.load(Ordering::SeqCst) {
            std::thread::sleep(status_interval);
            if stop_flag_mon.load(Ordering::SeqCst) { break; }

            let elapsed = start.elapsed().as_secs_f64();
            let remaining = duration.as_secs_f64() - elapsed;
            let iters = total_iters_mon.load(Ordering::Relaxed);
            let impr = improvements_mon.load(Ordering::Relaxed);
            let perturbs = perturbations_mon.load(Ordering::Relaxed);
            let current_best = global_best_mon.lock().unwrap().1;

            let improved_marker = if current_best < last_best { " *** IMPROVED ***" } else { "" };
            last_best = current_best;

            let op_stats: Vec<(u64, u64)> = op_uses_mon.iter()
                .zip(op_improvements_mon.iter())
                .map(|(u, i)| (u.load(Ordering::Relaxed), i.load(Ordering::Relaxed)))
                .collect();

            println!("\n╔══════════════════════════════════════════════════╗");
            println!("║  ILS STATUS @ {:.0}s ({:.0}s remaining)", elapsed, remaining);
            println!("╠══════════════════════════════════════════════════╣");
            println!("║  Global best: {:.2}{}", current_best, improved_marker);
            println!("║  Total iterations: {} ({:.0}/sec)", iters, iters as f64 / elapsed.max(0.001));
            println!("║  Improvements: {} | Perturbations: {}", impr, perturbs);
            println!("╠──────────────────────────────────────────────────╣");
            println!("║  Operator Stats (uses / improvements / rate):");
            for (i, (uses, imps)) in op_stats.iter().enumerate() {
                let rate = if *uses > 0 { *imps as f64 / *uses as f64 * 100.0 } else { 0.0 };
                println!("║    Op {}: {:>8} / {:>6} / {:>5.2}%", i, uses, imps, rate);
            }
            println!("╚══════════════════════════════════════════════════╝\n");
        }
    });

    // Run all ILS chains in parallel via rayon
    (0..num_chains).into_par_iter().for_each(|chain_id| {
        run_ils_chain(
            chain_id,
            init_solution,
            instance,
            local_search_ops,
            perturbation_ops,
            duration,
            sync_interval,
            global_best.clone(),
            stop_flag.clone(),
            total_iterations.clone(),
            improvements.clone(),
            perturbations.clone(),
            op_uses.clone(),
            op_improvements.clone(),
            start,
            sync_logger.clone(),
        );
    });

    // Wait for monitor to finish
    let _ = monitor.join();

    let (best_sol, best_score) = global_best.lock().unwrap().clone();
    let total_iters = total_iterations.load(Ordering::Relaxed);

    println!("\n╔══════════════════════════════════════════╗");
    println!("║        PARALLEL ILS COMPLETE             ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Total time: {:.1}s", start.elapsed().as_secs_f64());
    println!("║  Total iterations: {}", total_iters);
    println!("║  Iterations/sec: {:.0}", total_iters as f64 / start.elapsed().as_secs_f64().max(0.001));
    println!("║  FINAL BEST: {:.2}", best_score);
    println!("╚══════════════════════════════════════════╝");

    best_sol
}

fn run_ils_chain(
    chain_id: usize,
    init_solution: &Solution,
    instance: &TruckAndDroneInstance,
    local_search_ops: &[&(dyn Operator + Sync)],
    perturbation_ops: &[&(dyn Operator + Sync)],
    duration: Duration,
    sync_interval: Duration,
    global_best: Arc<Mutex<(Solution, f64)>>,
    stop_flag: Arc<AtomicBool>,
    total_iterations: Arc<AtomicU64>,
    improvements: Arc<AtomicU64>,
    perturbations: Arc<AtomicU64>,
    op_uses: Arc<Vec<AtomicU64>>,
    op_improvements: Arc<Vec<AtomicU64>>,
    start_time: Instant,
    sync_logger: Arc<Mutex<BufWriter<File>>>,
) {
    let mut incumbent = init_solution.clone();
    let mut incumbent_score = init_solution.verify_and_cost(instance)
        .map(|r| r.total_time).unwrap_or(f64::INFINITY);
    let mut best_solution = incumbent.clone();
    let mut best_score = incumbent_score;

    let start = Instant::now();
    let mut last_sync = Instant::now();
    let mut local_iters: u64 = 0;
    let mut rng = rand::rng();

    // No-improvement timeout: perturb after this many seconds without new best
    let perturb_timeout = Duration::from_secs(15);
    let mut last_best_improvement = Instant::now();

    // Local operator stats to batch flush
    let mut local_op_uses: Vec<u64> = vec![0; local_search_ops.len()];
    let mut local_op_impr: Vec<u64> = vec![0; local_search_ops.len()];

    let stop_check_interval = 512_u64;

    while start.elapsed() < duration {
        if local_iters % stop_check_interval == 0 && stop_flag.load(Ordering::Relaxed) {
            break;
        }
        local_iters += 1;

        // Flush counters every 1000 iters
        if local_iters % 1000 == 0 {
            total_iterations.fetch_add(1000, Ordering::Relaxed);
            for i in 0..local_search_ops.len() {
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

        // Sync with global best
        if last_sync.elapsed() >= sync_interval {
            let mut global = global_best.lock().unwrap();

            if best_score < global.1 {
                global.0 = best_solution.clone();
                global.1 = best_score;
                improvements.fetch_add(1, Ordering::Relaxed);
                println!("[ILS Chain {}] NEW GLOBAL BEST: {:.2}", chain_id, best_score);
            }

            // Adopt global best if ours is worse
            if global.1 < best_score {
                best_solution = global.0.clone();
                best_score = global.1;
                incumbent = best_solution.clone();
                incumbent_score = best_score;
                last_best_improvement = Instant::now();
            }

            // Log
            {
                let elapsed_secs = start_time.elapsed().as_secs_f64();
                let total_iters_snapshot = total_iterations.load(Ordering::Relaxed);
                let mut writer = sync_logger.lock().unwrap();
                writeln!(
                    writer,
                    "{},{:.4},{:.4},{:.2},{},{},{}",
                    total_iters_snapshot,
                    best_score,
                    incumbent_score,
                    elapsed_secs,
                    chain_id,
                    local_iters,
                    global.1
                ).ok();
            }

            drop(global);
            last_sync = Instant::now();
        }

        // === PERTURBATION: if stuck for too long, destroy-repair from best ===
        if last_best_improvement.elapsed() >= perturb_timeout {
            let p_idx = rng.random_range(0..perturbation_ops.len());
            if let Some(perturbed) = perturbation_ops[p_idx].apply(&best_solution, instance) {
                if let Ok(result) = perturbed.verify_and_cost(instance) {
                    incumbent = perturbed;
                    incumbent_score = result.total_time;
                    if incumbent_score < best_score {
                        best_solution = incumbent.clone();
                        best_score = incumbent_score;
                    }
                    perturbations.fetch_add(1, Ordering::Relaxed);
                }
            }
            last_best_improvement = Instant::now();
        }

        // === LOCAL SEARCH: pick random fast operator, accept only improvements ===
        let op_idx = rng.random_range(0..local_search_ops.len());
        local_op_uses[op_idx] += 1;

        let new_solution = match local_search_ops[op_idx].apply(&incumbent, instance) {
            Some(sol) => sol,
            None => continue,
        };

        let new_score = match new_solution.verify_and_cost(instance) {
            Ok(result) => result.total_time,
            Err(_) => continue,
        };

        // Accept only improvements
        if new_score < incumbent_score {
            incumbent = new_solution;
            incumbent_score = new_score;
            local_op_impr[op_idx] += 1;

            if new_score < best_score {
                best_solution = incumbent.clone();
                best_score = incumbent_score;
                last_best_improvement = Instant::now();
            }
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
    total_iterations.fetch_add(local_iters % 1000, Ordering::Relaxed);
}
