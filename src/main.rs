use std::time::Duration;

use Truck_and_drones::alns::{alns, alns_timed};
use Truck_and_drones::genetic::genetic_algorithm;
use Truck_and_drones::simulated_annealing::{run_adaptive_sa, run_simulated_annealing};
use Truck_and_drones::{parse_file, Solution, local_search::run_local_search};
use Truck_and_drones::operators::{MoveToDrone, MoveToTruck, SwapTruckCustomers, RelocateTruck};

fn main() {
    let file = "src/data/Truck_Drone_Contest.txt";

    let instance = parse_file(file);
    println!("Loaded instance with {} customers", instance.num_customers);

    let solution = Solution::trivial(instance.num_customers, 2);
    println!("Initial cost: {:.2}", solution.cost(&instance).unwrap());

    let operators: Vec<&dyn Truck_and_drones::operators::Operator> = vec![
        &MoveToDrone,
        &MoveToTruck,
        &SwapTruckCustomers,
        &RelocateTruck,
    ];

    // let best = run_simulated_annealing(&solution, &instance, &operators, 1000000);
    let best = genetic_algorithm(
        &instance,
        2,      // n_drones
        100,    // population_size
        1000,   // generations
        0.1,    // mutation_rate
        5,      // elite_count (keep top N unchanged)
    );
    
    println!("\nFinal solution: {}", best);
}
