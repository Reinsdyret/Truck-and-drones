use std::time::Duration;

use Truck_and_drones::simulated_annealing::{run_parallel_sa_multi_start, setup_stop_signal};
use Truck_and_drones::{parse_file, Solution};
use Truck_and_drones::operators::{DestroyRepair, GreedyReinsertSmart, TwoOptTruck, MoveToDrone, MoveToTruck};

fn main() {
    let file = "src/data/Truck_Drone_Contest.txt";

    let instance = parse_file(file);
    println!("Loaded instance with {} customers", instance.num_customers);

    // 4 diverse starting solutions from different basins
    let solutions = vec![
        Solution::best_til_now(),
        Solution::second_best_til_now(),
        Solution::third_best_til_now(),
        Solution::fourth_best_til_now(),
    ];
    
    for (i, sol) in solutions.iter().enumerate() {
        println!("Solution {}: cost = {:.2}", i, sol.cost(&instance).unwrap());
    }

    // Operators: fine-grained + structural moves
    let smart_op = GreedyReinsertSmart::new(15);
    let destroy_4 = DestroyRepair::new(4);
    let destroy_8 = DestroyRepair::new(8);
    let two_opt = TwoOptTruck;
    let move_to_drone = MoveToDrone;
    let move_to_truck = MoveToTruck;
    
    let operators: Vec<&(dyn Truck_and_drones::operators::Operator + Sync)> = vec![
        &smart_op,
        &destroy_4,
        &destroy_8,
        &two_opt,
        &move_to_drone,
        &move_to_truck,
    ];

    // Set up Ctrl+C handler for graceful stopping
    let stop_flag = setup_stop_signal();

    // Parallel SA: use all CPU threads minus one for monitoring
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let worker_chains = available_threads.saturating_sub(1).max(1);
    println!("Running {} parallel SA chains for 11 hours from {} diverse starts", worker_chains, solutions.len());

    let best = run_parallel_sa_multi_start(
        &solutions,
        &instance,
        &operators,
        Duration::from_secs(60 * 60 * 11),
        worker_chains,
        Duration::from_secs(30),
        Some(stop_flag),
    );
    
    println!("\nFinal solution: {}", best);
}
