use rand::Rng;
use std::collections::HashSet;
use rayon::prelude::*;
use crate::{Solution, TruckAndDroneInstance};

pub trait Operator: Sync {
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

/// 2-opt: reverse a segment of the truck route
pub struct TwoOptTruck;

impl Operator for TwoOptTruck {
    fn apply(&self, solution: &Solution, _instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        
        // Need at least 2 customers (excluding depots) for 2-opt to make sense
        if solution.truck_route.len() < 4 {
            return None;
        }
        
        // Pick two distinct positions (excluding first and last depot)
        // Valid range: 1 to len-2 (inclusive)
        let max_idx = solution.truck_route.len() - 2;
        if max_idx < 2 {
            return None;
        }
        
        let i = rng.random_range(1..=max_idx);
        let mut j = rng.random_range(1..=max_idx);
        while j == i {
            j = rng.random_range(1..=max_idx);
        }
        
        // Ensure i < j
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        
        let mut new_sol = solution.clone();
        
        // Reverse the segment [i, j] inclusive
        new_sol.truck_route[i..=j].reverse();
        
        Some(new_sol)
    }
}

/// 3-opt: reorder two segments of the truck route
pub struct ThreeOptTruck;

impl Operator for ThreeOptTruck {
    fn apply(&self, solution: &Solution, _instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        let len = solution.truck_route.len();
        if len < 6 {
            return None;
        }

        let max_idx = len - 2;
        let mut i = rng.random_range(1..=max_idx);
        let mut j = rng.random_range(1..=max_idx);
        let mut k = rng.random_range(1..=max_idx);
        while i == j || j == k || i == k {
            i = rng.random_range(1..=max_idx);
            j = rng.random_range(1..=max_idx);
            k = rng.random_range(1..=max_idx);
        }

        let mut cuts = [i, j, k];
        cuts.sort_unstable();
        let (i, j, k) = (cuts[0], cuts[1], cuts[2]);

        let a = &solution.truck_route[..i];
        let b = &solution.truck_route[i..j];
        let c = &solution.truck_route[j..k];
        let d = &solution.truck_route[k..];

        let pattern = rng.random_range(0..5);
        let mut new_route = Vec::with_capacity(len);
        new_route.extend_from_slice(a);
        match pattern {
            0 => {
                new_route.extend(b.iter().rev().copied());
                new_route.extend_from_slice(c);
            }
            1 => {
                new_route.extend_from_slice(b);
                new_route.extend(c.iter().rev().copied());
            }
            2 => {
                new_route.extend(c.iter().rev().copied());
                new_route.extend_from_slice(b);
            }
            3 => {
                new_route.extend_from_slice(c);
                new_route.extend(b.iter().rev().copied());
            }
            _ => {
                new_route.extend(b.iter().rev().copied());
                new_route.extend(c.iter().rev().copied());
            }
        }
        new_route.extend_from_slice(d);

        let mut new_sol = solution.clone();
        new_sol.truck_route = new_route;
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

/// Greedy assignment: remove a random non-rendezvous customer and reinsert best
pub struct GreedyAssignment;

impl Operator for GreedyAssignment {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        let rendezvous = collect_rendezvous_customers(solution);
        let mut candidates: Vec<usize> = solution.truck_route
            .iter()
            .filter(|&&c| c != 0 && !rendezvous.contains(&c))
            .copied()
            .collect();
        candidates.extend(
            solution.drone_deliveries
                .iter()
                .flatten()
                .filter(|&&c| !rendezvous.contains(&c))
                .copied(),
        );

        if candidates.is_empty() {
            return None;
        }

        let customer = candidates[rng.random_range(0..candidates.len())];
        let (base_sol, _) = remove_customer_from_solution(solution, customer)?;
        insert_customer_best(&base_sol, customer, instance)
    }
}

/// General assignment: remove 2-5 non-rendezvous customers and reinsert greedily
pub struct GeneralAssignment {
    pub min_remove: usize,
    pub max_remove: usize,
}

impl GeneralAssignment {
    pub fn new(min_remove: usize, max_remove: usize) -> Self {
        Self { min_remove, max_remove }
    }
}

impl Operator for GeneralAssignment {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        let rendezvous = collect_rendezvous_customers(solution);
        let mut candidates: Vec<usize> = solution.truck_route
            .iter()
            .filter(|&&c| c != 0 && !rendezvous.contains(&c))
            .copied()
            .collect();
        candidates.extend(
            solution.drone_deliveries
                .iter()
                .flatten()
                .filter(|&&c| !rendezvous.contains(&c))
                .copied(),
        );

        if candidates.len() < self.min_remove {
            return None;
        }

        let max_remove = self.max_remove.min(candidates.len());
        let num_remove = rng.random_range(self.min_remove..=max_remove);

        let mut to_remove = Vec::with_capacity(num_remove);
        for _ in 0..num_remove {
            let idx = rng.random_range(0..candidates.len());
            to_remove.push(candidates.swap_remove(idx));
        }

        let mut current = solution.clone();
        for customer in &to_remove {
            let (next, _) = remove_customer_from_solution(&current, *customer)?;
            current = next;
        }

        for customer in to_remove {
            current = insert_customer_best(&current, customer, instance)?;
        }

        Some(current)
    }
}

/// Origin-destination relocation: move a rendezvous customer and replan drones
pub struct OriginDestinationRelocation;

impl Operator for OriginDestinationRelocation {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        let rendezvous = collect_rendezvous_customers(solution);
        let candidates: Vec<usize> = rendezvous.into_iter().filter(|&c| c != 0).collect();
        if candidates.is_empty() {
            return None;
        }

        let customer = candidates[rng.random_range(0..candidates.len())];
        let (mut base_sol, _) = remove_customer_from_solution(solution, customer)?;

        let route_len = base_sol.truck_route.len();
        let mut best_pos = 1;
        let mut best_delta = f64::INFINITY;
        for i in 1..route_len {
            let prev = base_sol.truck_route[i - 1];
            let next = base_sol.truck_route[i];
            let delta = instance.truck_travel_costs[prev][customer]
                + instance.truck_travel_costs[customer][next]
                - instance.truck_travel_costs[prev][next];
            if delta < best_delta {
                best_delta = delta;
                best_pos = i;
            }
        }

        base_sol.truck_route.insert(best_pos, customer);
        for d in 0..base_sol.drone_deliveries.len() {
            for i in 0..base_sol.drone_launch_sites[d].len() {
                if base_sol.drone_launch_sites[d][i] >= best_pos {
                    base_sol.drone_launch_sites[d][i] += 1;
                }
                if base_sol.drone_landing_sites[d][i] >= best_pos {
                    base_sol.drone_landing_sites[d][i] += 1;
                }
            }
        }

        plan_drone_trips(&base_sol, instance)
    }
}

/// Drone planner: replan launch/land indices for existing drone deliveries
pub struct DronePlanner;

impl Operator for DronePlanner {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
        plan_drone_trips(solution, instance)
    }
}

/// Wild change: randomly reinsert 2-5 customers, then repair
pub struct WildChange {
    pub min_changes: usize,
    pub max_changes: usize,
}

impl WildChange {
    pub fn new(min_changes: usize, max_changes: usize) -> Self {
        Self { min_changes, max_changes }
    }
}

impl Operator for WildChange {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        let mut all_customers: Vec<usize> = solution.truck_route
            .iter()
            .filter(|&&c| c != 0)
            .copied()
            .collect();
        all_customers.extend(solution.drone_deliveries.iter().flatten().copied());

        if all_customers.is_empty() {
            return None;
        }

        let max_changes = self.max_changes.min(all_customers.len());
        let num_changes = rng.random_range(self.min_changes..=max_changes);

        let mut to_change = Vec::with_capacity(num_changes);
        for _ in 0..num_changes {
            let idx = rng.random_range(0..all_customers.len());
            to_change.push(all_customers.swap_remove(idx));
        }

        let mut current = solution.clone();
        for customer in &to_change {
            let (next, _) = remove_customer_from_solution(&current, *customer)?;
            current = next;
        }

        for customer in to_change {
            let mut inserted = false;
            let attempts = 20;
            for _ in 0..attempts {
                if rng.random::<f64>() < 0.5 {
                    // Try random truck insertion
                    let pos = rng.random_range(1..current.truck_route.len());
                    let mut candidate = current.clone();
                    candidate.truck_route.insert(pos, customer);
                    for d in 0..candidate.drone_deliveries.len() {
                        for i in 0..candidate.drone_launch_sites[d].len() {
                            if candidate.drone_launch_sites[d][i] >= pos {
                                candidate.drone_launch_sites[d][i] += 1;
                            }
                            if candidate.drone_landing_sites[d][i] >= pos {
                                candidate.drone_landing_sites[d][i] += 1;
                            }
                        }
                    }
                    if candidate.verify_and_cost(instance).is_ok() {
                        current = candidate;
                        inserted = true;
                        break;
                    }
                } else {
                    // Try random drone insertion
                    let route_len = current.truck_route.len();
                    if route_len < 2 {
                        continue;
                    }
                    let launch = rng.random_range(0..route_len - 1);
                    let land = rng.random_range(launch + 1..route_len);
                    let launch_node = current.truck_route[launch];
                    let land_node = current.truck_route[land];
                    let flight = instance.drone_travel_costs[launch_node][customer]
                        + instance.drone_travel_costs[customer][land_node];
                    if flight > instance.max_flight_range as f64 {
                        continue;
                    }
                    let drone_idx = rng.random_range(0..current.drone_deliveries.len());
                    if !can_insert_drone_trip_fast(&current, drone_idx, launch, land) {
                        continue;
                    }
                    let insert_idx = current.drone_launch_sites[drone_idx]
                        .iter()
                        .position(|&l| l > launch)
                        .unwrap_or(current.drone_deliveries[drone_idx].len());
                    let mut candidate = current.clone();
                    candidate.drone_deliveries[drone_idx].insert(insert_idx, customer);
                    candidate.drone_launch_sites[drone_idx].insert(insert_idx, launch);
                    candidate.drone_landing_sites[drone_idx].insert(insert_idx, land);
                    if candidate.verify_and_cost(instance).is_ok() {
                        current = candidate;
                        inserted = true;
                        break;
                    }
                }
            }

            if !inserted {
                current = insert_customer_best(&current, customer, instance)?;
            }
        }

        Some(current)
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

/// Fast greedy reinsert operator: remove a random customer and reinsert at best position
/// This is designed for high iteration throughput with good solution quality
pub struct GreedyReinsert;

impl Operator for GreedyReinsert {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        
        // Collect all customers (truck + drone)
        let truck_customers: Vec<usize> = solution.truck_route.iter()
            .filter(|&&c| c != 0)
            .copied()
            .collect();
        let drone_customers: Vec<(usize, usize)> = solution.drone_deliveries.iter()
            .enumerate()
            .flat_map(|(d, custs)| custs.iter().map(move |&c| (c, d)))
            .collect();
        
        let total = truck_customers.len() + drone_customers.len();
        if total == 0 {
            return None;
        }
        
        // Pick random customer
        let choice = rng.random_range(0..total);
        let (customer, from_drone) = if choice < truck_customers.len() {
            (truck_customers[choice], None)
        } else {
            let (c, d) = drone_customers[choice - truck_customers.len()];
            (c, Some(d))
        };
        
        // Create new solution and remove customer
        let mut new_sol = solution.clone();
        
        if let Some(drone_idx) = from_drone {
            // Remove from drone
            let idx = new_sol.drone_deliveries[drone_idx].iter().position(|&c| c == customer)?;
            new_sol.drone_deliveries[drone_idx].remove(idx);
            new_sol.drone_launch_sites[drone_idx].remove(idx);
            new_sol.drone_landing_sites[drone_idx].remove(idx);
        } else {
            // Remove from truck
            let pos = new_sol.truck_route.iter().position(|&c| c == customer)?;
            new_sol.truck_route.remove(pos);
            
            // Adjust drone indices for removed truck position
            for d in 0..new_sol.drone_deliveries.len() {
                for i in 0..new_sol.drone_launch_sites[d].len() {
                    if new_sol.drone_launch_sites[d][i] > pos {
                        new_sol.drone_launch_sites[d][i] -= 1;
                    }
                    if new_sol.drone_landing_sites[d][i] > pos {
                        new_sol.drone_landing_sites[d][i] -= 1;
                    }
                }
            }
        }
        
        // Find best insertion: either truck or drone
        let max_range = instance.max_flight_range as f64;
        let route_len = new_sol.truck_route.len();
        
        // === Evaluate best truck insertion (delta cost) ===
        let mut best_truck_pos = 1;
        let mut best_truck_delta = f64::INFINITY;
        
        for i in 1..route_len {
            let prev = new_sol.truck_route[i - 1];
            let next = new_sol.truck_route[i];
            let delta = instance.truck_travel_costs[prev][customer]
                + instance.truck_travel_costs[customer][next]
                - instance.truck_travel_costs[prev][next];
            
            if delta < best_truck_delta {
                best_truck_delta = delta;
                best_truck_pos = i;
            }
        }
        
        // === Evaluate best drone insertion ===
        // We look for the shortest flight time drone trip
        let mut best_drone: Option<(usize, usize, usize, usize, f64)> = None; // (drone_idx, insert_idx, launch, land, flight_time)
        
        for launch in 0..route_len.saturating_sub(1) {
            let launch_node = new_sol.truck_route[launch];
            let dist_to_cust = instance.drone_travel_costs[launch_node][customer];
            
            // Early skip if just getting to customer exceeds range
            if dist_to_cust > max_range {
                continue;
            }
            
            for land in (launch + 1)..route_len {
                let land_node = new_sol.truck_route[land];
                let flight_time = dist_to_cust + instance.drone_travel_costs[customer][land_node];
                
                if flight_time > max_range {
                    continue;
                }
                
                // Check if this is better than current best drone option
                if best_drone.is_none() || flight_time < best_drone.unwrap().4 {
                    // Find a drone that can take this trip
                    for d in 0..new_sol.drone_deliveries.len() {
                        if can_insert_drone_trip_fast(&new_sol, d, launch, land) {
                            let insert_idx = new_sol.drone_launch_sites[d]
                                .iter()
                                .position(|&l| l > launch)
                                .unwrap_or(new_sol.drone_deliveries[d].len());
                            best_drone = Some((d, insert_idx, launch, land, flight_time));
                            break;
                        }
                    }
                }
            }
        }
        
        // === Decide: truck or drone? ===
        // Heuristic: prefer drone if flight_time is reasonable (saves truck time)
        // We compare truck delta against a threshold based on drone option
        let use_drone = if let Some((_, _, _, _, flight_time)) = best_drone {
            // Use drone if it exists and truck insertion is expensive
            // or with some probability to maintain diversity
            flight_time < best_truck_delta * 1.5 || rng.random::<f64>() < 0.3
        } else {
            false
        };
        
        if use_drone {
            let (drone_idx, insert_idx, launch, land, _) = best_drone.unwrap();
            new_sol.drone_deliveries[drone_idx].insert(insert_idx, customer);
            new_sol.drone_launch_sites[drone_idx].insert(insert_idx, launch);
            new_sol.drone_landing_sites[drone_idx].insert(insert_idx, land);
        } else {
            // Insert into truck
            new_sol.truck_route.insert(best_truck_pos, customer);
            
            // Adjust drone indices for the new truck position
            for d in 0..new_sol.drone_deliveries.len() {
                for i in 0..new_sol.drone_launch_sites[d].len() {
                    if new_sol.drone_launch_sites[d][i] >= best_truck_pos {
                        new_sol.drone_launch_sites[d][i] += 1;
                    }
                    if new_sol.drone_landing_sites[d][i] >= best_truck_pos {
                        new_sol.drone_landing_sites[d][i] += 1;
                    }
                }
            }
        }
        
        Some(new_sol)
    }
}

/// Fast check for drone trip insertion (inlined logic)
#[inline]
fn can_insert_drone_trip_fast(solution: &Solution, drone: usize, launch: usize, land: usize) -> bool {
    let launches = &solution.drone_launch_sites[drone];
    let landings = &solution.drone_landing_sites[drone];
    
    if launches.is_empty() {
        return true;
    }
    
    // Binary search would be faster for large drone schedules, but linear is fine for typical sizes
    let insert_idx = launches.iter().position(|&l| l > launch).unwrap_or(launches.len());
    
    // Check constraints
    (insert_idx == 0 || launch >= landings[insert_idx - 1]) &&
    (insert_idx >= launches.len() || launches[insert_idx] >= land)
}

fn collect_rendezvous_customers(solution: &Solution) -> HashSet<usize> {
    let mut rendezvous = HashSet::new();
    for d in 0..solution.drone_launch_sites.len() {
        for &pos in solution.drone_launch_sites[d].iter().chain(solution.drone_landing_sites[d].iter()) {
            if pos < solution.truck_route.len() {
                rendezvous.insert(solution.truck_route[pos]);
            }
        }
    }
    rendezvous
}

fn remove_customer_from_solution(
    solution: &Solution,
    customer: usize,
) -> Option<(Solution, Option<usize>)> {
    let mut new_sol = solution.clone();
    if let Some(pos) = new_sol.truck_route.iter().position(|&c| c == customer) {
        new_sol.truck_route.remove(pos);
        for d in 0..new_sol.drone_deliveries.len() {
            for i in 0..new_sol.drone_launch_sites[d].len() {
                if new_sol.drone_launch_sites[d][i] > pos {
                    new_sol.drone_launch_sites[d][i] -= 1;
                }
                if new_sol.drone_landing_sites[d][i] > pos {
                    new_sol.drone_landing_sites[d][i] -= 1;
                }
            }
        }
        return Some((new_sol, Some(pos)));
    }

    for d in 0..new_sol.drone_deliveries.len() {
        if let Some(idx) = new_sol.drone_deliveries[d].iter().position(|&c| c == customer) {
            new_sol.drone_deliveries[d].remove(idx);
            new_sol.drone_launch_sites[d].remove(idx);
            new_sol.drone_landing_sites[d].remove(idx);
            return Some((new_sol, None));
        }
    }

    None
}

fn insert_customer_best(
    base_sol: &Solution,
    customer: usize,
    instance: &TruckAndDroneInstance,
) -> Option<Solution> {
    let route_len = base_sol.truck_route.len();
    let max_range = instance.max_flight_range as f64;
    let mut candidates: Vec<(bool, f64, usize, usize, usize, usize)> = Vec::new();

    for i in 1..route_len {
        let prev = base_sol.truck_route[i - 1];
        let next = base_sol.truck_route[i];
        let delta = instance.truck_travel_costs[prev][customer]
            + instance.truck_travel_costs[customer][next]
            - instance.truck_travel_costs[prev][next];
        candidates.push((true, delta, i, 0, 0, 0));
    }

    for launch in 0..route_len.saturating_sub(1) {
        let launch_node = base_sol.truck_route[launch];
        let dist_to_cust = instance.drone_travel_costs[launch_node][customer];
        if dist_to_cust > max_range {
            continue;
        }
        for land in (launch + 1)..route_len {
            let land_node = base_sol.truck_route[land];
            let flight = dist_to_cust + instance.drone_travel_costs[customer][land_node];
            if flight > max_range {
                continue;
            }
            for d in 0..base_sol.drone_deliveries.len() {
                if can_insert_drone_trip_fast(base_sol, d, launch, land) {
                    let insert_idx = base_sol.drone_launch_sites[d]
                        .iter()
                        .position(|&l| l > launch)
                        .unwrap_or(base_sol.drone_deliveries[d].len());
                    candidates.push((false, flight, d, insert_idx, launch, land));
                    break;
                }
            }
        }
    }

    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (is_truck, _score, pos_or_drone, insert_idx, launch, land) in candidates {
        let mut candidate = base_sol.clone();
        if is_truck {
            let insert_pos = pos_or_drone;
            candidate.truck_route.insert(insert_pos, customer);
            for d in 0..candidate.drone_deliveries.len() {
                for i in 0..candidate.drone_launch_sites[d].len() {
                    if candidate.drone_launch_sites[d][i] >= insert_pos {
                        candidate.drone_launch_sites[d][i] += 1;
                    }
                    if candidate.drone_landing_sites[d][i] >= insert_pos {
                        candidate.drone_landing_sites[d][i] += 1;
                    }
                }
            }
        } else {
            let drone_idx = pos_or_drone;
            candidate.drone_deliveries[drone_idx].insert(insert_idx, customer);
            candidate.drone_launch_sites[drone_idx].insert(insert_idx, launch);
            candidate.drone_landing_sites[drone_idx].insert(insert_idx, land);
        }
        if candidate.verify_and_cost(instance).is_ok() {
            return Some(candidate);
        }
    }

    None
}

fn plan_drone_trips(solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
    let mut new_sol = solution.clone();
    let route_len = new_sol.truck_route.len();
    let max_range = instance.max_flight_range as f64;

    for d in 0..new_sol.drone_deliveries.len() {
        let deliveries = new_sol.drone_deliveries[d].clone();
        new_sol.drone_launch_sites[d].clear();
        new_sol.drone_landing_sites[d].clear();

        let mut prev_land = 0;
        for customer in deliveries {
            let mut best_pair: Option<(usize, usize, f64)> = None;
            for launch in prev_land..route_len.saturating_sub(1) {
                let launch_node = new_sol.truck_route[launch];
                let dist_out = instance.drone_travel_costs[launch_node][customer];
                if dist_out > max_range {
                    continue;
                }
                for land in (launch + 1)..route_len {
                    let land_node = new_sol.truck_route[land];
                    let flight = dist_out + instance.drone_travel_costs[customer][land_node];
                    if flight > max_range {
                        continue;
                    }
                    if best_pair.is_none() || flight < best_pair.unwrap().2 {
                        best_pair = Some((launch, land, flight));
                    }
                }
            }
            let (launch, land, _) = best_pair?;
            new_sol.drone_launch_sites[d].push(launch);
            new_sol.drone_landing_sites[d].push(land);
            prev_land = land;
        }
    }

    if new_sol.verify_and_cost(instance).is_ok() {
        Some(new_sol)
    } else {
        None
    }
}

/// Smarter greedy reinsert: evaluates actual cost for top candidates
/// Slower than GreedyReinsert but makes better decisions
/// Uses weighted random selection from top candidates for exploration
pub struct GreedyReinsertSmart {
    pub top_k: usize,  // Number of top candidates to evaluate (for truck and drone each)
}

impl GreedyReinsertSmart {
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }
    
    pub fn default() -> Self {
        Self { top_k: 10 }
    }
}

impl Operator for GreedyReinsertSmart {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        
        // Collect all customers
        let truck_customers: Vec<usize> = solution.truck_route.iter()
            .filter(|&&c| c != 0)
            .copied()
            .collect();
        let drone_customers: Vec<(usize, usize)> = solution.drone_deliveries.iter()
            .enumerate()
            .flat_map(|(d, custs)| custs.iter().map(move |&c| (c, d)))
            .collect();
        
        let total = truck_customers.len() + drone_customers.len();
        if total == 0 {
            return None;
        }
        
        // Pick random customer
        let choice = rng.random_range(0..total);
        let (customer, from_drone) = if choice < truck_customers.len() {
            (truck_customers[choice], None)
        } else {
            let (c, d) = drone_customers[choice - truck_customers.len()];
            (c, Some(d))
        };
        
        // Create base solution with customer removed
        let mut base_sol = solution.clone();
        
        if let Some(drone_idx) = from_drone {
            let idx = base_sol.drone_deliveries[drone_idx].iter().position(|&c| c == customer)?;
            base_sol.drone_deliveries[drone_idx].remove(idx);
            base_sol.drone_launch_sites[drone_idx].remove(idx);
            base_sol.drone_landing_sites[drone_idx].remove(idx);
        } else {
            let pos = base_sol.truck_route.iter().position(|&c| c == customer)?;
            base_sol.truck_route.remove(pos);
            
            // Adjust drone indices
            for d in 0..base_sol.drone_deliveries.len() {
                for i in 0..base_sol.drone_launch_sites[d].len() {
                    if base_sol.drone_launch_sites[d][i] > pos {
                        base_sol.drone_launch_sites[d][i] -= 1;
                    }
                    if base_sol.drone_landing_sites[d][i] > pos {
                        base_sol.drone_landing_sites[d][i] -= 1;
                    }
                }
            }
        }
        
        let max_range = instance.max_flight_range as f64;
        let route_len = base_sol.truck_route.len();
        
        // === Collect top truck candidates by delta heuristic ===
        let mut truck_candidates: Vec<(usize, f64)> = Vec::with_capacity(route_len);
        for i in 1..route_len {
            let prev = base_sol.truck_route[i - 1];
            let next = base_sol.truck_route[i];
            let delta = instance.truck_travel_costs[prev][customer]
                + instance.truck_travel_costs[customer][next]
                - instance.truck_travel_costs[prev][next];
            truck_candidates.push((i, delta));
        }
        // Sort by delta, take top K
        truck_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        truck_candidates.truncate(self.top_k);
        
        // === Collect top drone candidates by flight time ===
        let mut drone_candidates: Vec<(usize, usize, usize, usize, f64)> = Vec::new(); // (drone, insert_idx, launch, land, flight_time)
        
        for launch in 0..route_len.saturating_sub(1) {
            let launch_node = base_sol.truck_route[launch];
            let dist_to_cust = instance.drone_travel_costs[launch_node][customer];
            
            if dist_to_cust > max_range {
                continue;
            }
            
            for land in (launch + 1)..route_len {
                let land_node = base_sol.truck_route[land];
                let flight_time = dist_to_cust + instance.drone_travel_costs[customer][land_node];
                
                if flight_time > max_range {
                    continue;
                }
                
                // Find a drone that can take this trip
                for d in 0..base_sol.drone_deliveries.len() {
                    if can_insert_drone_trip_fast(&base_sol, d, launch, land) {
                        let insert_idx = base_sol.drone_launch_sites[d]
                            .iter()
                            .position(|&l| l > launch)
                            .unwrap_or(base_sol.drone_deliveries[d].len());
                        drone_candidates.push((d, insert_idx, launch, land, flight_time));
                        break;
                    }
                }
            }
        }
        // Sort by flight time, take top K
        drone_candidates.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap());
        drone_candidates.truncate(self.top_k);
        
        // === Evaluate actual cost for all candidates IN PARALLEL ===
        
        // Evaluate truck candidates in parallel
        let truck_evaluated: Vec<(Solution, f64)> = truck_candidates
            .par_iter()
            .filter_map(|(pos, _)| {
                let mut candidate = base_sol.clone();
                candidate.truck_route.insert(*pos, customer);
                
                // Adjust drone indices for insertion
                for d in 0..candidate.drone_deliveries.len() {
                    for i in 0..candidate.drone_launch_sites[d].len() {
                        if candidate.drone_launch_sites[d][i] >= *pos {
                            candidate.drone_launch_sites[d][i] += 1;
                        }
                        if candidate.drone_landing_sites[d][i] >= *pos {
                            candidate.drone_landing_sites[d][i] += 1;
                        }
                    }
                }
                
                candidate.verify_and_cost(instance)
                    .ok()
                    .map(|result| (candidate, result.total_time))
            })
            .collect();
        
        // Evaluate drone candidates in parallel
        let drone_evaluated: Vec<(Solution, f64)> = drone_candidates
            .par_iter()
            .filter_map(|(drone_idx, insert_idx, launch, land, _)| {
                let mut candidate = base_sol.clone();
                candidate.drone_deliveries[*drone_idx].insert(*insert_idx, customer);
                candidate.drone_launch_sites[*drone_idx].insert(*insert_idx, *launch);
                candidate.drone_landing_sites[*drone_idx].insert(*insert_idx, *land);
                
                candidate.verify_and_cost(instance)
                    .ok()
                    .map(|result| (candidate, result.total_time))
            })
            .collect();
        
        // Combine results
        let mut evaluated: Vec<(Solution, f64)> = truck_evaluated;
        evaluated.extend(drone_evaluated);
        
        if evaluated.is_empty() {
            return None;
        }
        
        // Sort by cost (best first)
        evaluated.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Weighted random selection: better solutions more likely
        // Use rank-based weighting: rank 1 gets weight N, rank 2 gets N-1, etc.
        let n = evaluated.len();
        let total_weight: usize = (1..=n).sum(); // n + (n-1) + ... + 1 = n*(n+1)/2
        let mut r = rng.random_range(0..total_weight);
        
        for (i, (solution, _cost)) in evaluated.into_iter().enumerate() {
            let weight = n - i; // Best has weight n, worst has weight 1
            if r < weight {
                return Some(solution);
            }
            r -= weight;
        }
        
        // Fallback (shouldn't reach here)
        None
    }
}

/// Ultra-fast greedy reinsert: uses heuristics only, verifies only the final choice
/// Much faster than GreedyReinsertSmart (no parallel evaluation needed)
pub struct GreedyReinsertFast {
    pub top_k: usize,
}

impl GreedyReinsertFast {
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }
}

impl Operator for GreedyReinsertFast {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        
        // Collect all customers
        let truck_customers: Vec<usize> = solution.truck_route.iter()
            .filter(|&&c| c != 0)
            .copied()
            .collect();
        let drone_customers: Vec<(usize, usize)> = solution.drone_deliveries.iter()
            .enumerate()
            .flat_map(|(d, custs)| custs.iter().map(move |&c| (c, d)))
            .collect();
        
        let total = truck_customers.len() + drone_customers.len();
        if total == 0 {
            return None;
        }
        
        // Pick random customer
        let choice = rng.random_range(0..total);
        let (customer, from_drone) = if choice < truck_customers.len() {
            (truck_customers[choice], None)
        } else {
            let (c, d) = drone_customers[choice - truck_customers.len()];
            (c, Some(d))
        };
        
        // Create base solution with customer removed
        let mut base_sol = solution.clone();
        let removed_pos: Option<usize>;
        
        if let Some(drone_idx) = from_drone {
            let idx = base_sol.drone_deliveries[drone_idx].iter().position(|&c| c == customer)?;
            base_sol.drone_deliveries[drone_idx].remove(idx);
            base_sol.drone_launch_sites[drone_idx].remove(idx);
            base_sol.drone_landing_sites[drone_idx].remove(idx);
            removed_pos = None;
        } else {
            let pos = base_sol.truck_route.iter().position(|&c| c == customer)?;
            base_sol.truck_route.remove(pos);
            removed_pos = Some(pos);
            
            // Adjust drone indices
            for d in 0..base_sol.drone_deliveries.len() {
                for i in 0..base_sol.drone_launch_sites[d].len() {
                    if base_sol.drone_launch_sites[d][i] > pos {
                        base_sol.drone_launch_sites[d][i] -= 1;
                    }
                    if base_sol.drone_landing_sites[d][i] > pos {
                        base_sol.drone_landing_sites[d][i] -= 1;
                    }
                }
            }
        }
        
        let max_range = instance.max_flight_range as f64;
        let route_len = base_sol.truck_route.len();
        
        // === Collect ALL candidates with heuristic scores ===
        // Format: (is_truck, score, data...) where lower score = better
        let mut candidates: Vec<(bool, f64, usize, usize, usize, usize)> = Vec::with_capacity(route_len * 3);
        
        // Truck candidates: (true, delta, insert_pos, 0, 0, 0)
        for i in 1..route_len {
            let prev = base_sol.truck_route[i - 1];
            let next = base_sol.truck_route[i];
            let delta = instance.truck_travel_costs[prev][customer]
                + instance.truck_travel_costs[customer][next]
                - instance.truck_travel_costs[prev][next];
            candidates.push((true, delta, i, 0, 0, 0));
        }
        
        // Drone candidates: (false, flight_time, drone, insert_idx, launch, land)
        for launch in 0..route_len.saturating_sub(1) {
            let launch_node = base_sol.truck_route[launch];
            let dist_to_cust = instance.drone_travel_costs[launch_node][customer];
            
            if dist_to_cust > max_range {
                continue;
            }
            
            for land in (launch + 1)..route_len {
                let land_node = base_sol.truck_route[land];
                let flight_time = dist_to_cust + instance.drone_travel_costs[customer][land_node];
                
                if flight_time > max_range {
                    continue;
                }
                
                // Find a drone that can take this trip
                for d in 0..base_sol.drone_deliveries.len() {
                    if can_insert_drone_trip_fast(&base_sol, d, launch, land) {
                        let insert_idx = base_sol.drone_launch_sites[d]
                            .iter()
                            .position(|&l| l > launch)
                            .unwrap_or(base_sol.drone_deliveries[d].len());
                        candidates.push((false, flight_time, d, insert_idx, launch, land));
                        break;
                    }
                }
            }
        }
        
        if candidates.is_empty() {
            return None;
        }
        
        // Sort by score (lower = better), take top K
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(self.top_k);
        
        // Weighted random: rank 1 gets weight N, rank 2 gets N-1, etc.
        let n = candidates.len();
        let total_weight: usize = (1..=n).sum();
        let mut r = rng.random_range(0..total_weight);
        
        let mut chosen_idx = 0;
        for (i, _) in candidates.iter().enumerate() {
            let weight = n - i;
            if r < weight {
                chosen_idx = i;
                break;
            }
            r -= weight;
        }
        
        let (is_truck, _, pos_or_drone, insert_idx, launch, land) = candidates[chosen_idx];
        
        // Build and verify only the chosen candidate
        let mut result = base_sol;
        
        if is_truck {
            let insert_pos = pos_or_drone;
            result.truck_route.insert(insert_pos, customer);
            
            // Adjust drone indices
            for d in 0..result.drone_deliveries.len() {
                for i in 0..result.drone_launch_sites[d].len() {
                    if result.drone_launch_sites[d][i] >= insert_pos {
                        result.drone_launch_sites[d][i] += 1;
                    }
                    if result.drone_landing_sites[d][i] >= insert_pos {
                        result.drone_landing_sites[d][i] += 1;
                    }
                }
            }
        } else {
            let drone_idx = pos_or_drone;
            result.drone_deliveries[drone_idx].insert(insert_idx, customer);
            result.drone_launch_sites[drone_idx].insert(insert_idx, launch);
            result.drone_landing_sites[drone_idx].insert(insert_idx, land);
        }
        
        // Verify feasibility
        if result.verify_and_cost(instance).is_ok() {
            Some(result)
        } else {
            // If chosen is infeasible, try the best one
            if chosen_idx != 0 {
                let (is_truck, _, pos_or_drone, insert_idx, launch, land) = candidates[0];
                let mut result = solution.clone();
                
                // Re-remove customer
                if let Some(drone_idx) = from_drone {
                    let idx = result.drone_deliveries[drone_idx].iter().position(|&c| c == customer)?;
                    result.drone_deliveries[drone_idx].remove(idx);
                    result.drone_launch_sites[drone_idx].remove(idx);
                    result.drone_landing_sites[drone_idx].remove(idx);
                } else if let Some(pos) = removed_pos {
                    result.truck_route.remove(pos);
                    for d in 0..result.drone_deliveries.len() {
                        for i in 0..result.drone_launch_sites[d].len() {
                            if result.drone_launch_sites[d][i] > pos {
                                result.drone_launch_sites[d][i] -= 1;
                            }
                            if result.drone_landing_sites[d][i] > pos {
                                result.drone_landing_sites[d][i] -= 1;
                            }
                        }
                    }
                }
                
                if is_truck {
                    let insert_pos = pos_or_drone;
                    result.truck_route.insert(insert_pos, customer);
                    for d in 0..result.drone_deliveries.len() {
                        for i in 0..result.drone_launch_sites[d].len() {
                            if result.drone_launch_sites[d][i] >= insert_pos {
                                result.drone_launch_sites[d][i] += 1;
                            }
                            if result.drone_landing_sites[d][i] >= insert_pos {
                                result.drone_landing_sites[d][i] += 1;
                            }
                        }
                    }
                } else {
                    let drone_idx = pos_or_drone;
                    result.drone_deliveries[drone_idx].insert(insert_idx, customer);
                    result.drone_launch_sites[drone_idx].insert(insert_idx, launch);
                    result.drone_landing_sites[drone_idx].insert(insert_idx, land);
                }
                
                if result.verify_and_cost(instance).is_ok() {
                    return Some(result);
                }
            }
            None
        }
    }
}

/// Destroy-Repair operator: remove multiple customers and reinsert them greedily
/// Makes larger moves that can escape local optima
pub struct DestroyRepair {
    pub num_destroy: usize,  // Number of customers to remove (3-5 typical)
}

impl DestroyRepair {
    pub fn new(num_destroy: usize) -> Self {
        Self { num_destroy }
    }
}

impl Operator for DestroyRepair {
    fn apply(&self, solution: &Solution, instance: &TruckAndDroneInstance) -> Option<Solution> {
        let mut rng = rand::rng();
        
        // Collect all customers
        let mut all_customers: Vec<usize> = solution.truck_route.iter()
            .filter(|&&c| c != 0)
            .copied()
            .collect();
        all_customers.extend(solution.drone_deliveries.iter().flatten().copied());
        
        if all_customers.len() <= self.num_destroy {
            return None;
        }
        
        // Randomly select customers to remove
        let num_to_remove = self.num_destroy.min(all_customers.len() - 1);
        let mut to_remove: Vec<usize> = Vec::with_capacity(num_to_remove);
        
        for _ in 0..num_to_remove {
            let idx = rng.random_range(0..all_customers.len());
            to_remove.push(all_customers.swap_remove(idx));
        }
        
        // Create solution with customers removed
        let mut new_sol = solution.clone();
        
        for &customer in &to_remove {
            // Try to remove from truck
            if let Some(pos) = new_sol.truck_route.iter().position(|&c| c == customer) {
                new_sol.truck_route.remove(pos);
                
                // Adjust drone indices
                for d in 0..new_sol.drone_deliveries.len() {
                    for i in 0..new_sol.drone_launch_sites[d].len() {
                        if new_sol.drone_launch_sites[d][i] > pos {
                            new_sol.drone_launch_sites[d][i] -= 1;
                        }
                        if new_sol.drone_landing_sites[d][i] > pos {
                            new_sol.drone_landing_sites[d][i] -= 1;
                        }
                    }
                }
            } else {
                // Remove from drone
                for d in 0..new_sol.drone_deliveries.len() {
                    if let Some(idx) = new_sol.drone_deliveries[d].iter().position(|&c| c == customer) {
                        new_sol.drone_deliveries[d].remove(idx);
                        new_sol.drone_launch_sites[d].remove(idx);
                        new_sol.drone_landing_sites[d].remove(idx);
                        break;
                    }
                }
            }
        }
        
        // Reinsert customers one by one using weighted random from top candidates
        let max_range = instance.max_flight_range as f64;
        const TOP_K: usize = 10;  // Consider top 10 candidates per customer
        
        for customer in to_remove {
            let route_len = new_sol.truck_route.len();
            
            // Collect all candidates: (is_truck, score, pos_or_drone, insert_idx, launch, land)
            let mut candidates: Vec<(bool, f64, usize, usize, usize, usize)> = Vec::with_capacity(route_len * 2);
            
            // Truck candidates
            for i in 1..route_len {
                let prev = new_sol.truck_route[i - 1];
                let next = new_sol.truck_route[i];
                let delta = instance.truck_travel_costs[prev][customer]
                    + instance.truck_travel_costs[customer][next]
                    - instance.truck_travel_costs[prev][next];
                candidates.push((true, delta, i, 0, 0, 0));
            }
            
            // Drone candidates
            for launch in 0..route_len.saturating_sub(1) {
                let launch_node = new_sol.truck_route[launch];
                let dist_to_cust = instance.drone_travel_costs[launch_node][customer];
                
                if dist_to_cust > max_range {
                    continue;
                }
                
                for land in (launch + 1)..route_len {
                    let land_node = new_sol.truck_route[land];
                    let flight_time = dist_to_cust + instance.drone_travel_costs[customer][land_node];
                    
                    if flight_time > max_range {
                        continue;
                    }
                    
                    for d in 0..new_sol.drone_deliveries.len() {
                        if can_insert_drone_trip_fast(&new_sol, d, launch, land) {
                            let insert_idx = new_sol.drone_launch_sites[d]
                                .iter()
                                .position(|&l| l > launch)
                                .unwrap_or(new_sol.drone_deliveries[d].len());
                            candidates.push((false, flight_time, d, insert_idx, launch, land));
                            break;
                        }
                    }
                }
            }
            
            if candidates.is_empty() {
                // Fallback: insert in truck at position 1
                new_sol.truck_route.insert(1, customer);
                for d in 0..new_sol.drone_deliveries.len() {
                    for i in 0..new_sol.drone_launch_sites[d].len() {
                        if new_sol.drone_launch_sites[d][i] >= 1 {
                            new_sol.drone_launch_sites[d][i] += 1;
                        }
                        if new_sol.drone_landing_sites[d][i] >= 1 {
                            new_sol.drone_landing_sites[d][i] += 1;
                        }
                    }
                }
                continue;
            }
            
            // Sort by score, take top K
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            candidates.truncate(TOP_K);
            
            // Weighted random: rank 1 gets weight N, rank 2 gets N-1, etc.
            let n = candidates.len();
            let total_weight: usize = (1..=n).sum();
            let mut r = rng.random_range(0..total_weight);
            
            let mut chosen_idx = 0;
            for (i, _) in candidates.iter().enumerate() {
                let weight = n - i;
                if r < weight {
                    chosen_idx = i;
                    break;
                }
                r -= weight;
            }
            
            let (is_truck, _, pos_or_drone, insert_idx, launch, land) = candidates[chosen_idx];
            
            if is_truck {
                let insert_pos = pos_or_drone;
                new_sol.truck_route.insert(insert_pos, customer);
                
                // Adjust drone indices
                for d in 0..new_sol.drone_deliveries.len() {
                    for i in 0..new_sol.drone_launch_sites[d].len() {
                        if new_sol.drone_launch_sites[d][i] >= insert_pos {
                            new_sol.drone_launch_sites[d][i] += 1;
                        }
                        if new_sol.drone_landing_sites[d][i] >= insert_pos {
                            new_sol.drone_landing_sites[d][i] += 1;
                        }
                    }
                }
            } else {
                let drone_idx = pos_or_drone;
                new_sol.drone_deliveries[drone_idx].insert(insert_idx, customer);
                new_sol.drone_launch_sites[drone_idx].insert(insert_idx, launch);
                new_sol.drone_landing_sites[drone_idx].insert(insert_idx, land);
            }
        }
        
        Some(new_sol)
    }
}
