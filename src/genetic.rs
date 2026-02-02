use rand::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::{Solution, TruckAndDroneInstance};

/// Individual: truck route (permutation of customers)
#[derive(Clone)]
struct Individual {
    truck_customers: Vec<usize>,  // Customers on truck (order matters)
    fitness: f64,
}

/// Genetic Algorithm with greedy drone assignment
pub fn genetic_algorithm(
    instance: &TruckAndDroneInstance,
    n_drones: usize,
    population_size: usize,
    generations: usize,
    mutation_rate: f64,
    elite_count: usize,
) -> Solution {
    let mut rng = rand::rng();
    let n_customers = instance.num_customers;
    
    println!("Initializing population of {}...", population_size);
    
    // Initialize population - start with all customers on truck
    let mut population: Vec<Individual> = (0..population_size)
        .map(|i| {
            let mut customers: Vec<usize> = (1..=n_customers).collect();
            customers.shuffle(&mut rng);
            let fitness = evaluate_fast(&customers, instance, n_drones);
            if i % 10 == 0 { print!("."); }
            Individual { truck_customers: customers, fitness }
        })
        .collect();
    println!(" done");
    
    // Sort by fitness (lower is better)
    population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    
    let mut best = population[0].clone();
    println!("Initial best: {:.2}", best.fitness);
    
    // CSV logging
    let file = File::create("genetic_scores.csv").expect("failed to create csv");
    let mut writer = BufWriter::new(file);
    writeln!(writer, "gen,best,avg,worst").unwrap();
    
    for generation in 0..generations {
        let mut new_population = Vec::with_capacity(population_size);
        
        // Elitism - keep best individuals
        for i in 0..elite_count.min(population_size) {
            new_population.push(population[i].clone());
        }
        
        // Generate rest through crossover and mutation
        while new_population.len() < population_size {
            // Tournament selection
            let parent1 = tournament_select(&population, 3, &mut rng);
            let parent2 = tournament_select(&population, 3, &mut rng);
            
            // Crossover (Order Crossover - OX)
            let mut child_customers = order_crossover(&parent1.truck_customers, &parent2.truck_customers, &mut rng);
            
            // Mutation (swap mutation)
            if rng.random::<f64>() < mutation_rate {
                swap_mutate(&mut child_customers, &mut rng);
            }
            
            // Sometimes do inversion mutation for larger changes
            if rng.random::<f64>() < mutation_rate * 0.5 {
                inversion_mutate(&mut child_customers, &mut rng);
            }
            
            let fitness = evaluate_fast(&child_customers, instance, n_drones);
            new_population.push(Individual { truck_customers: child_customers, fitness });
        }
        
        population = new_population;
        population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        
        if population[0].fitness < best.fitness {
            best = population[0].clone();
            println!("Gen {}: NEW BEST = {:.2}", generation, best.fitness);
        }
        
        // Print every generation for now
        let avg: f64 = population.iter().map(|i| i.fitness).sum::<f64>() / population.len() as f64;
        println!("Gen {}: best = {:.2}, avg = {:.2}", generation, best.fitness, avg);
        writeln!(writer, "{},{:.4},{:.4},{:.4}", generation, best.fitness, avg, population.last().unwrap().fitness).unwrap();
    }
    
    // Build final solution from best individual with full greedy
    println!("Building final solution with full greedy drone assignment...");
    build_solution_with_greedy_drones(&best.truck_customers, instance, n_drones)
}

/// Fast evaluation - just evaluate truck route (no drones)
/// This is a proxy for solution quality - good truck routes will also be good with drones
fn evaluate_fast(
    truck_customers: &[usize],
    instance: &TruckAndDroneInstance,
    n_drones: usize,
) -> f64 {
    // Build truck-only solution
    let mut truck_route: Vec<usize> = vec![0];
    truck_route.extend(truck_customers);
    truck_route.push(0);
    
    let solution = Solution {
        truck_route,
        drone_deliveries: vec![Vec::new(); n_drones],
        drone_launch_sites: vec![Vec::new(); n_drones],
        drone_landing_sites: vec![Vec::new(); n_drones],
    };
    
    solution.verify_and_cost(instance)
        .map(|r| r.total_time)
        .unwrap_or(f64::INFINITY)
}

/// Evaluate a truck route by greedily assigning drones
fn evaluate_with_greedy_drones(
    truck_customers: &[usize],
    instance: &TruckAndDroneInstance,
    n_drones: usize,
) -> f64 {
    let solution = build_solution_with_greedy_drones(truck_customers, instance, n_drones);
    solution.verify_and_cost(instance)
        .map(|r| r.total_time)
        .unwrap_or(f64::INFINITY)
}

/// Build a solution by greedily assigning customers to drones
fn build_solution_with_greedy_drones(
    truck_customers: &[usize],
    instance: &TruckAndDroneInstance,
    n_drones: usize,
) -> Solution {
    // Start with all customers on truck
    let mut truck_route: Vec<usize> = vec![0];
    truck_route.extend(truck_customers);
    truck_route.push(0);
    
    let mut drone_deliveries: Vec<Vec<usize>> = vec![Vec::new(); n_drones];
    let mut drone_launch_sites: Vec<Vec<usize>> = vec![Vec::new(); n_drones];
    let mut drone_landing_sites: Vec<Vec<usize>> = vec![Vec::new(); n_drones];
    
    let max_range = instance.max_flight_range as f64;
    
    // Greedily assign customers to drones
    loop {
        let mut best_assignment: Option<(usize, usize, usize, usize, f64)> = None; // (customer_idx, drone, launch, land, flight_time)
        
        // Find best customer to move to drone
        for (cust_idx, &customer) in truck_route.iter().enumerate() {
            if customer == 0 { continue; } // Skip depot
            
            // Try removing this customer
            let mut test_route = truck_route.clone();
            test_route.remove(cust_idx);
            
            // Find best launch/land pair
            for launch in 0..test_route.len() - 1 {
                for land in launch + 1..test_route.len() {
                    let launch_node = test_route[launch];
                    let land_node = test_route[land];
                    
                    let flight_time = instance.drone_travel_costs[launch_node][customer]
                        + instance.drone_travel_costs[customer][land_node];
                    
                    if flight_time <= max_range {
                        // Check if any drone can take this
                        for d in 0..n_drones {
                            if can_insert_drone_trip(&drone_launch_sites[d], &drone_landing_sites[d], launch, land) {
                                if best_assignment.is_none() || flight_time < best_assignment.unwrap().4 {
                                    best_assignment = Some((cust_idx, d, launch, land, flight_time));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        match best_assignment {
            Some((cust_idx, drone, launch, land, _)) => {
                let customer = truck_route.remove(cust_idx);
                
                // Adjust indices after removal
                let adj_launch = if launch >= cust_idx { launch } else { launch };
                let adj_land = if land > cust_idx { land - 1 } else { land };
                
                // Find insert position
                let insert_idx = drone_launch_sites[drone]
                    .iter()
                    .position(|&l| l > adj_launch)
                    .unwrap_or(drone_deliveries[drone].len());
                
                drone_deliveries[drone].insert(insert_idx, customer);
                drone_launch_sites[drone].insert(insert_idx, adj_launch);
                drone_landing_sites[drone].insert(insert_idx, adj_land);
            }
            None => break, // No more valid assignments
        }
        
        // Keep at least 1 customer on truck (besides depot)
        if truck_route.len() <= 3 { break; }
    }
    
    Solution {
        truck_route,
        drone_deliveries,
        drone_launch_sites,
        drone_landing_sites,
    }
}

fn can_insert_drone_trip(launches: &[usize], landings: &[usize], launch: usize, land: usize) -> bool {
    if launches.is_empty() { return true; }
    
    let insert_idx = launches.iter().position(|&l| l > launch).unwrap_or(launches.len());
    
    if insert_idx > 0 && launch < landings[insert_idx - 1] { return false; }
    if insert_idx < launches.len() && launches[insert_idx] < land { return false; }
    
    true
}

fn tournament_select<'a>(population: &'a [Individual], tournament_size: usize, rng: &mut impl Rng) -> &'a Individual {
    let mut best: Option<&Individual> = None;
    for _ in 0..tournament_size {
        let idx = rng.random_range(0..population.len());
        if best.is_none() || population[idx].fitness < best.unwrap().fitness {
            best = Some(&population[idx]);
        }
    }
    best.unwrap()
}

/// Order Crossover (OX) for permutations
fn order_crossover(parent1: &[usize], parent2: &[usize], rng: &mut impl Rng) -> Vec<usize> {
    let n = parent1.len();
    if n < 2 { return parent1.to_vec(); }
    
    let mut start = rng.random_range(0..n);
    let mut end = rng.random_range(0..n);
    if start > end { std::mem::swap(&mut start, &mut end); }
    
    let mut child = vec![0; n];
    let mut used = vec![false; n + 1]; // +1 for 1-indexed customers
    
    // Copy segment from parent1
    for i in start..=end {
        child[i] = parent1[i];
        if parent1[i] > 0 && parent1[i] <= n {
            used[parent1[i]] = true;
        }
    }
    
    // Fill rest from parent2 in order
    let mut pos = (end + 1) % n;
    for &gene in parent2.iter().cycle().skip(end + 1).take(n) {
        if gene > 0 && gene <= n && !used[gene] {
            // Find next empty position
            while child[pos] != 0 {
                pos = (pos + 1) % n;
            }
            child[pos] = gene;
            used[gene] = true;
            pos = (pos + 1) % n;
        }
    }
    
    child
}

fn swap_mutate(route: &mut [usize], rng: &mut impl Rng) {
    if route.len() < 2 { return; }
    let i = rng.random_range(0..route.len());
    let j = rng.random_range(0..route.len());
    route.swap(i, j);
}

fn inversion_mutate(route: &mut [usize], rng: &mut impl Rng) {
    if route.len() < 2 { return; }
    let mut i = rng.random_range(0..route.len());
    let mut j = rng.random_range(0..route.len());
    if i > j { std::mem::swap(&mut i, &mut j); }
    route[i..=j].reverse();
}
