use std::time::Duration;

use Truck_and_drones::alns::{alns, alns_timed, run_parallel_alns};
use Truck_and_drones::genetic::genetic_algorithm;
use Truck_and_drones::simulated_annealing::{run_adaptive_sa, run_simulated_annealing, run_simulated_annealing_timed, run_parallel_sa, setup_stop_signal};
use Truck_and_drones::{parse_file, Solution, local_search::run_local_search};
use Truck_and_drones::operators::{MoveToDrone, MoveToTruck, SwapTruckCustomers, RelocateTruck, TwoOptTruck, GreedyReinsert, GreedyReinsertSmart, GreedyReinsertFast, DestroyRepair};

fn main() {
    let file = "src/data/Truck_Drone_Contest.txt";

    let instance = parse_file(file);
    println!("Loaded instance with {} customers", instance.num_customers);

    let solution = Solution::trivial(instance.num_customers, 2);
    println!("Initial cost: {:.2}", solution.cost(&instance).unwrap());

    // Fast operator: uses heuristics for ranking, only verifies final choice
    let fast_op = GreedyReinsertFast::new(20);  // Select from top 20 candidates
    
    // DestroyRepair is slow! Only use occasionally or make a fast version
    let destroy_4 = DestroyRepair::new(4);
    let destroy_8 = DestroyRepair::new(8);
    let destroy_15 = DestroyRepair::new(15);  // Big shake-up for escaping deep optima
    
    let operators: Vec<&(dyn Truck_and_drones::operators::Operator + Sync)> = vec![
        &fast_op,
        &destroy_4,
        &destroy_8,
        &destroy_15,  // Larger destruction for escaping
    ];

    // Set up Ctrl+C handler for graceful stopping
    let stop_flag = setup_stop_signal();

    // Parallel ALNS: 3 worker chains + 1 monitor thread. Therfore the emplyees need to be paid minimum vage. Alas ther is  a vacation bounes for the best and hardets workers. 
    let best = run_parallel_alns(
        &solution, 
        &instance, 
        &operators, 
        Duration::from_secs(60 * 30),  // Total duration
        3,                              // Number of worker chains
        Duration::from_secs(5),        // Sync interval between chains
        Some(stop_flag)
    );
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
