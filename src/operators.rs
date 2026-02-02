use rand::Rng;
use crate::{Solution, TruckAndDroneInstance};

pub trait Operator {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution>;
}

/// Move a random customer from truck to a random drone
pub struct MoveToDrone;

impl Operator for MoveToDrone {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        
        // Get truck customers (excluding depot)
        let truck_customers: Vec<usize> = solution.truck_route.iter()
            .filter(|&&c| c != 0)
            .copied()
            .collect();
        
        if truck_customers.is_empty() {
            return None;
        }
        
        let customer = truck_customers[rng.random_range(0..truck_customers.len())];
        let drone_idx = rng.random_range(0..solution.drone_deliveries.len());
        
        let mut new_sol = solution.clone();
        let pos = new_sol.truck_route.iter().position(|&c| c == customer)?;
        new_sol.truck_route.remove(pos);
        
        let route_len = new_sol.truck_route.len();
        let max_range = instance.max_flight_range as f64;
        
        let mut best_pair: Option<(usize, usize, f64)> = None;
        
        for launch in 0..route_len - 1 {
            for land in launch + 1..route_len {
                let launch_node = new_sol.truck_route[launch];
                let land_node = new_sol.truck_route[land];
                
                let total_flight = instance.drone_travel_costs[launch_node][customer]
                    + instance.drone_travel_costs[customer][land_node];
                
                if total_flight <= max_range && (best_pair.is_none() || total_flight < best_pair.unwrap().2) {
                    best_pair = Some((launch, land, total_flight));
                }
            }
        }
        
        let (launch_pos, land_pos, _) = best_pair?;
        
        let launches = &new_sol.drone_launch_sites[drone_idx];
        let insert_idx = launches.iter().position(|&l| l > launch_pos).unwrap_or(new_sol.drone_deliveries[drone_idx].len());
        
        if insert_idx > 0 && launch_pos < new_sol.drone_landing_sites[drone_idx][insert_idx - 1] {
            return None;
        }
        if insert_idx < launches.len() && launches[insert_idx] < land_pos {
            return None;
        }
        
        new_sol.drone_deliveries[drone_idx].insert(insert_idx, customer);
        new_sol.drone_launch_sites[drone_idx].insert(insert_idx, launch_pos);
        new_sol.drone_landing_sites[drone_idx].insert(insert_idx, land_pos);
        
        Some(new_sol)
    }
}

/// Move a random customer from drone back to truck
pub struct MoveToTruck;

impl Operator for MoveToTruck {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        
        // Get all drone customers
        let drone_customers: Vec<usize> = solution.drone_deliveries.iter().flatten().copied().collect();
        
        if drone_customers.is_empty() {
            return None;
        }
        
        let customer = drone_customers[rng.random_range(0..drone_customers.len())];
        
        let mut new_sol = solution.clone();
        
        // Find and remove from drone
        for d in 0..new_sol.drone_deliveries.len() {
            if let Some(idx) = new_sol.drone_deliveries[d].iter().position(|&c| c == customer) {
                new_sol.drone_deliveries[d].remove(idx);
                new_sol.drone_launch_sites[d].remove(idx);
                new_sol.drone_landing_sites[d].remove(idx);
                break;
            }
        }
        
        // Cheapest insertion into truck route
        let mut best_pos = 1;
        let mut best_cost = f64::INFINITY;
        
        for i in 1..new_sol.truck_route.len() {
            let prev = new_sol.truck_route[i - 1];
            let next = new_sol.truck_route[i];
            let delta = instance.truck_travel_costs[prev][customer]
                + instance.truck_travel_costs[customer][next]
                - instance.truck_travel_costs[prev][next];
            
            if delta < best_cost {
                best_cost = delta;
                best_pos = i;
            }
        }
        
        new_sol.truck_route.insert(best_pos, customer);
        
        // Adjust drone indices
        for d in 0..new_sol.drone_deliveries.len() {
            for i in 0..new_sol.drone_launch_sites[d].len() {
                if new_sol.drone_launch_sites[d][i] >= best_pos {
                    new_sol.drone_launch_sites[d][i] += 1;
                }
                if new_sol.drone_landing_sites[d][i] >= best_pos {
                    new_sol.drone_landing_sites[d][i] += 1;
                }
            }
        }
        
        Some(new_sol)
    }
}

/// Swap two random customers in the truck route
pub struct SwapTruckCustomers;

impl Operator for SwapTruckCustomers {
    fn apply(&self, solution: &Solution, _instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        
        let truck_customers: Vec<usize> = solution.truck_route.iter()
            .filter(|&&c| c != 0)
            .copied()
            .collect();
        
        if truck_customers.len() < 2 {
            return None;
        }
        
        let idx_a = rng.random_range(0..truck_customers.len());
        let mut idx_b = rng.random_range(0..truck_customers.len());
        while idx_b == idx_a {
            idx_b = rng.random_range(0..truck_customers.len());
        }
        
        let mut new_sol = solution.clone();
        let pos_a = new_sol.truck_route.iter().position(|&c| c == truck_customers[idx_a])?;
        let pos_b = new_sol.truck_route.iter().position(|&c| c == truck_customers[idx_b])?;
        new_sol.truck_route.swap(pos_a, pos_b);
        
        Some(new_sol)
    }
}

/// Relocate a random customer to a random position in the truck route
pub struct RelocateTruck;

impl Operator for RelocateTruck {
    fn apply(&self, solution: &Solution, _instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        
        let truck_customers: Vec<usize> = solution.truck_route.iter()
            .filter(|&&c| c != 0)
            .copied()
            .collect();
        
        if truck_customers.is_empty() {
            return None;
        }
        
        let customer = truck_customers[rng.random_range(0..truck_customers.len())];
        
        let mut new_sol = solution.clone();
        let old_pos = new_sol.truck_route.iter().position(|&c| c == customer)?;
        new_sol.truck_route.remove(old_pos);
        
        let new_pos = rng.random_range(1..new_sol.truck_route.len());
        new_sol.truck_route.insert(new_pos, customer);
        
        Some(new_sol)
    }
}

/// Greedily assign truck customers to drones based on shortest flight time
pub struct GreedyDroneAssign;

impl Operator for GreedyDroneAssign {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut new_sol = solution.clone();
        let max_range = instance.max_flight_range as f64;
        let n_drones = new_sol.drone_deliveries.len();
        
        // Keep assigning until no more beneficial assignments
        loop {
            let mut best_assignment: Option<(usize, usize, usize, usize, f64)> = None; // (customer, drone, launch, land, flight_time)
            
            // Get current truck customers
            let truck_customers: Vec<usize> = new_sol.truck_route.iter()
                .filter(|&&c| c != 0)
                .copied()
                .collect();
            
            if truck_customers.len() <= 1 {
                break; // Keep at least one customer on truck
            }
            
            for &customer in &truck_customers {
                let pos = new_sol.truck_route.iter().position(|&c| c == customer).unwrap();
                
                // Try removing this customer and finding best drone assignment
                let mut test_route = new_sol.truck_route.clone();
                test_route.remove(pos);
                
                for launch in 0..test_route.len() - 1 {
                    for land in launch + 1..test_route.len() {
                        let launch_node = test_route[launch];
                        let land_node = test_route[land];
                        
                        let flight_time = instance.drone_travel_costs[launch_node][customer]
                            + instance.drone_travel_costs[customer][land_node];
                        
                        if flight_time <= max_range {
                            // Check if this is the best so far
                            if best_assignment.is_none() || flight_time < best_assignment.unwrap().4 {
                                // Find a drone that can take this assignment
                                for d in 0..n_drones {
                                    if can_insert_drone_trip(&new_sol, d, launch, land) {
                                        best_assignment = Some((customer, d, launch, land, flight_time));
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            match best_assignment {
                Some((customer, drone, launch, land, _)) => {
                    // Remove customer from truck
                    let pos = new_sol.truck_route.iter().position(|&c| c == customer).unwrap();
                    new_sol.truck_route.remove(pos);
                    
                    // Adjust indices for removal
                    let adj_launch = if launch >= pos { launch } else { launch };
                    let adj_land = if land > pos { land - 1 } else { land };
                    
                    // Find insert position in drone's list
                    let insert_idx = new_sol.drone_launch_sites[drone]
                        .iter()
                        .position(|&l| l > adj_launch)
                        .unwrap_or(new_sol.drone_deliveries[drone].len());
                    
                    new_sol.drone_deliveries[drone].insert(insert_idx, customer);
                    new_sol.drone_launch_sites[drone].insert(insert_idx, adj_launch);
                    new_sol.drone_landing_sites[drone].insert(insert_idx, adj_land);
                }
                None => break, // No more valid assignments
            }
        }
        
        // Only return if we made at least one assignment
        if new_sol.drone_deliveries.iter().map(|d| d.len()).sum::<usize>() 
            > solution.drone_deliveries.iter().map(|d| d.len()).sum::<usize>() {
            Some(new_sol)
        } else {
            None
        }
    }
}

/// Check if a drone trip can be inserted without violating sequencing
fn can_insert_drone_trip(solution: &Solution, drone: usize, launch: usize, land: usize) -> bool {
    let launches = &solution.drone_launch_sites[drone];
    let landings = &solution.drone_landing_sites[drone];
    
    if launches.is_empty() {
        return true;
    }
    
    // Find where this would be inserted
    let insert_idx = launches.iter().position(|&l| l > launch).unwrap_or(launches.len());
    
    // Check previous trip: our launch must be >= its landing
    if insert_idx > 0 && launch < landings[insert_idx - 1] {
        return false;
    }
    
    // Check next trip: its launch must be >= our landing
    if insert_idx < launches.len() && launches[insert_idx] < land {
        return false;
    }
    
    true
}
