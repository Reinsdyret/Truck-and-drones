use std::time::Duration;

use Truck_and_drones::simulated_annealing::{run_parallel_sa, setup_stop_signal};
use Truck_and_drones::{parse_file, Solution};
use Truck_and_drones::operators::{DestroyRepair, GreedyReinsertSmart, TwoOptTruck};

fn main() {
    let file = "src/data/Truck_Drone_Contest.txt";

    let instance = parse_file(file);
    println!("Loaded instance with {} customers", instance.num_customers);

    // let solution = Solution::trivial(instance.num_customers, 2);
    let solution = Solution::best_til_now();
    println!("Initial cost: {:.2}", solution.cost(&instance).unwrap());

    let smart_op = GreedyReinsertSmart::new(15);
    let destroy_4 = DestroyRepair::new(4);
    let two_opt = TwoOptTruck;
    
    let operators: Vec<&(dyn Truck_and_drones::operators::Operator + Sync)> = vec![
        &smart_op,
        &destroy_4,
        &two_opt,
    ];

    // Set up Ctrl+C handler for graceful stopping
    let stop_flag = setup_stop_signal();

    // Parallel SA: use all CPU threads minus one for monitoring
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let worker_chains = available_threads.saturating_sub(1).max(1);
    println!("Using {} worker chains ({} logical threads)", worker_chains, available_threads);
    let best = run_parallel_sa(
        &solution,
        &instance,
        &operators,
        Duration::from_secs(60 * 60),
        worker_chains,
        Duration::from_secs(60),
        Some(stop_flag),
    );
    // let best = alns_timed(
    //     &solution, 
    //     &instance, 
    //     &operators, 
    //     Duration::from_secs(60 * 30),  // Total duration
    //     Some(stop_flag),
    // );
    // Status updates print every 100 seconds automatically
    
    // Alternative: Parallel SA
    // let best = run_parallel_sa(
    //     &solution, 
    //     &instance, 
    //     &operators, 
    //     Duration::from_secs(60 * 30),
    //     3,
    //     Duration::from_secs(5),
    //     Some(stop_flag)
    // );
    
    // Single-threaded SA (for comparison)
    // let best = run_simulated_annealing_timed(&solution, &instance, &operators, Duration::from_secs(60 * 30), Some(stop_flag));
    // let best = alns_timed(&solution, &instance, &operators, Duration::from_secs(60 * 30));
    // let best = genetic_algorithm(
    //     &instance,
    //     2,      // n_drones
    //     100,    // population_size
    //     1000,   // generations
    //     0.1,    // mutation_rate
    //     5,      // elite_count (keep top N unchanged)
    // );
    
    println!("\nFinal solution: {}", best);
}
