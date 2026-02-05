// Base functions like reading of input and verifying + scoring

pub mod operators;
pub mod local_search;
pub mod simulated_annealing;
pub mod alns;
pub mod genetic;

use std::{fmt, fs::File, io::{BufRead, BufReader}, str::FromStr};

#[derive(Clone)]
pub struct TruckAndDroneInstance {
  pub num_customers: usize,
  pub max_flight_range: usize,
  pub truck_travel_costs: Vec<Vec<f64>>,
  pub drone_travel_costs: Vec<Vec<f64>>,
}

#[derive(Clone)]
pub struct Solution {
  pub truck_route: Vec<usize>,
  pub drone_deliveries: Vec<Vec<usize>>,
  pub drone_launch_sites: Vec<Vec<usize>>,
  pub drone_landing_sites: Vec<Vec<usize>>,
}


/// Result of verify_and_cost: feasibility + objective value
#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub total_time: f64,
    pub truck_arrivals: Vec<f64>,
    pub truck_departures: Vec<f64>,
}

impl Solution {
    /// Create a trivial feasible solution: all customers on truck, no drone deliveries.
    /// 
    /// Truck visits customers in order: 0 -> 1 -> 2 -> ... -> n -> 0
    pub fn trivial(num_customers: usize, num_drones: usize) -> Self {
        let truck_route = (0..=num_customers).chain(std::iter::once(0)).collect();
        let drone_deliveries = vec![Vec::new(); num_drones];
        let drone_launch_sites = vec![Vec::new(); num_drones];
        let drone_landing_sites = vec![Vec::new(); num_drones];
        
        Self { truck_route, drone_deliveries, drone_launch_sites, drone_landing_sites }
    }

    pub fn best_til_now() -> Self {
        Solution::parse("0,87,59,57,49,44,19,83,10,12,96,45,37,63,5,65,27,34,80,76,62,91,29,77,51,35,68,11,74,66,55,50,2,85,18,100,95,20,75,15,93,58,3,67,25,7,64,30,6,82,31,24,98,4,40,0 |43,72,79,8,28,86,41,17,53,36,81,48,73,97,60,22,99,9,21,54,89,70,13,23,47,92,-1,14,16,61,69,56,33,38,26,1,52,84,90,88,32,78,39,94,46,42,71 |1,2,5,8,10,12,13,15,16,21,24,26,28,30,32,34,35,36,38,42,43,45,47,48,50,53,-1,1,2,5,10,14,15,17,20,23,26,28,29,30,31,35,36,42,45,47,54 |2,5,8,10,12,13,15,16,21,24,26,28,30,32,34,35,36,38,42,43,45,47,48,50,53,56,-1,2,5,10,14,15,17,20,23,26,28,29,30,31,35,36,42,45,47,54,56").unwrap()
    }
    
    /// Parse a solution from the competition format string.
    /// 
    /// Format: `part1 |part2 |part3 |part4`
    /// - Part 1: Truck route (comma-separated nodes, 0 = depot)
    /// - Part 2: Drone deliveries (comma-separated, -1 separates drones)
    /// - Part 3: Launch sites (1-based indices into truck route, -1 separates drones)
    /// - Part 4: Landing sites (1-based indices into truck route, -1 separates drones)
    /// 
    /// Note: Parts 3 & 4 use 1-based indexing; internally we store 0-based.
    pub fn parse(s: &str) -> Result<Self, &'static str> {
        let parts: Vec<&str> = s.split('|').map(str::trim).collect();
        if parts.len() != 4 {
            return Err("Solution must have exactly 4 parts separated by '|'");
        }
        
        // Helper to parse comma-separated integers
        let parse_ints = |s: &str| -> Result<Vec<i32>, &'static str> {
            s.split(',')
                .map(|x| x.trim().parse::<i32>().map_err(|_| "Invalid integer in solution"))
                .collect()
        };
        
        // Helper to split by -1 separator into drone groups. In this particular case a drone atacche is not that likley. But never say never. While you get ./1  yopu are safe but a =1 wold be dentramental fore your health...
        let split_by_separator = |vals: Vec<i32>| -> Vec<Vec<i32>> {
            vals.split(|&x| x == -1)
                .map(|slice| slice.to_vec())
                .collect()
        };
        
        // Part 1: Truck route
        let truck_route: Vec<usize> = parse_ints(parts[0])?
            .into_iter()
            .map(|x| x as usize)
            .collect();
        
        // Part 2: Drone deliveries (split by -1)
        let part2 = parse_ints(parts[1])?;
        let drone_deliveries: Vec<Vec<usize>> = split_by_separator(part2)
            .into_iter()
            .map(|group| group.into_iter().map(|x| x as usize).collect())
            .collect();
        
        // Part 3: Launch sites (1-based -> 0-based)
        let part3 = parse_ints(parts[2])?;
        let drone_launch_sites: Vec<Vec<usize>> = split_by_separator(part3)
            .into_iter()
            .map(|group| group.into_iter().map(|x| (x - 1) as usize).collect())
            .collect();
        
        // Part 4: Landing sites (1-based -> 0-based)
        let part4 = parse_ints(parts[3])?;
        let drone_landing_sites: Vec<Vec<usize>> = split_by_separator(part4)
            .into_iter()
            .map(|group| group.into_iter().map(|x| (x - 1) as usize).collect())
            .collect();
        
        Ok(Solution {
            truck_route,
            drone_deliveries,
            drone_launch_sites,
            drone_landing_sites,
        })
    }
    
    /// Combined feasibility check and cost calculation in a single optimized pass.
    /// 
    /// Returns Ok(VerifyResult) if feasible, Err(&str) with reason if infeasible.
    pub fn verify_and_cost(&self, instance: &TruckAndDroneInstance) -> Result<VerifyResult, &'static str> {
        let n_nodes = instance.num_customers + 1;
        let route_len = self.truck_route.len();
        let n_drones = self.drone_deliveries.len();
        let max_range = instance.max_flight_range as f64;
        
        // 1. TRUCK ROUTE FEASIBILITY
        if route_len < 2 {
            return Err("Truck route too short");
        }
        if *self.truck_route.first().unwrap() != 0 || *self.truck_route.last().unwrap() != 0 {
            return Err("Truck must start and end at depot (0)");
        }
        if !self.truck_route.iter().all(|&n| n < n_nodes) {
            return Err("Invalid node in truck route");
        }
        if self.truck_route[1..route_len - 1].iter().any(|&n| n == 0) {
            return Err("Depot (0) appears in middle of truck route");
        }
        
        // 2. COMPLETENESS CHECK
        let mut customer_count = vec![0_usize; n_nodes];
        self.truck_route[1..route_len - 1].iter().for_each(|&n| customer_count[n] += 1);
        
        // Validate drone customers and count them
        if self.drone_deliveries.iter().flatten().any(|&c| c == 0 || c >= n_nodes) {
            return Err("Invalid customer in drone deliveries");
        }
        self.drone_deliveries.iter().flatten().for_each(|&c| customer_count[c] += 1);
        
        if !customer_count[1..].iter().all(|&cnt| cnt == 1) {
            return Err("Customer not served exactly once");
        }
        
        // 3. STRUCTURAL CONSISTENCY
        if self.drone_launch_sites.len() != n_drones || self.drone_landing_sites.len() != n_drones {
            return Err("Drone arrays length mismatch");
        }
        
        // Validate each drone's trips using zip and windows
        let trips_valid = (0..n_drones).all(|d| {
            let (deliveries, launches, landings) = (
                &self.drone_deliveries[d],
                &self.drone_launch_sites[d],
                &self.drone_landing_sites[d],
            );
            
            // Length check
            if deliveries.len() != launches.len() || deliveries.len() != landings.len() {
                return false;
            }
            
            // All trips: valid cells, launch < landing
            let cells_valid = launches.iter().zip(landings.iter())
                .all(|(&l, &r)| l < route_len && r < route_len && l < r);
            
            // Sequencing: each launch >= previous landing (using windows on zipped pairs)
            let sequencing_valid = launches.iter().zip(landings.iter())
                .collect::<Vec<_>>()
                .windows(2)
                .all(|w| *w[1].0 >= *w[0].1);
            
            cells_valid && sequencing_valid
        });
        
        if !trips_valid {
            return Err("Invalid drone trip structure or sequencing");
        }
        
        // 4. COST CALCULATION + FLIGHT RANGE CHECK
        let mut truck_arrivals = vec![0.0_f64; route_len];
        let mut truck_departures = vec![0.0_f64; route_len];
        let mut drone_availability = vec![0.0_f64; n_drones];
        let mut total_time = 0.0_f64;
        
        // Build landing lookup: position -> [(drone_idx, trip_idx)]
        let mut drone_landings: Vec<Vec<(usize, usize)>> = vec![Vec::new(); route_len];
        self.drone_landing_sites.iter().enumerate().for_each(|(d, sites)| {
            sites.iter().enumerate().for_each(|(t, &pos)| drone_landings[pos].push((d, t)));
        });
        
        // Process route
        for (i, (&prev_node, &curr_node)) in self.truck_route.iter().zip(self.truck_route.iter().skip(1)).enumerate() {
            let i = i + 1; // Adjust index since we're iterating pairs
            
            let truck_arrival = truck_departures[i - 1] + instance.truck_travel_costs[prev_node][curr_node];
            truck_arrivals[i] = truck_arrival;
            
            // Process landing drones, track latest return
            let drone_result = drone_landings[i].iter().try_fold(truck_arrival, |latest, &(d, t)| {
                let customer = self.drone_deliveries[d][t];
                let launch_pos = self.drone_launch_sites[d][t];
                let (launch_node, landing_node) = (self.truck_route[launch_pos], self.truck_route[i]);
                
                let flight_out = instance.drone_travel_costs[launch_node][customer];
                let flight_back = instance.drone_travel_costs[customer][landing_node];
                let total_flight = flight_out + flight_back;
                
                let actual_launch = truck_arrivals[launch_pos].max(drone_availability[d]);
                let drone_return = actual_launch + total_flight;
                
                // Flight range check (with wait time)
                let drone_wait = if curr_node != 0 { (truck_arrival - drone_return).max(0.0) } else { 0.0 };
                if total_flight + drone_wait > max_range {return Err("Drone flight exceeds max range");}
                
                drone_availability[d] = drone_return;
                total_time += actual_launch + flight_out; // Customer arrival time
                
                Ok(latest.max(drone_return))
            })?;
            
            truck_departures[i] = drone_result;
            
            if curr_node != 0 {
                total_time += truck_arrival;
            }
        }
        
        Ok(VerifyResult {
            total_time: total_time / 100.0,
            truck_arrivals,
            truck_departures,
        })
    }
    
    #[inline]
    pub fn is_feasible(&self, instance: &TruckAndDroneInstance) -> bool {
        self.verify_and_cost(instance).is_ok()
    }
    
    #[inline]
    pub fn cost(&self, instance: &TruckAndDroneInstance) -> Option<f64> {
        self.verify_and_cost(instance).ok().map(|r| r.total_time)
    }
}

/// Implement FromStr for parsing "0,1,2,0 |3,4,-1,5 |1,2,-1,1 |2,3,-1,2" format
impl FromStr for Solution {
    type Err = &'static str;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Solution::parse(s)
    }
}

/// Implement Display for serializing to competition format
impl fmt::Display for Solution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Helper to join drone groups with -1 separator
        fn join_groups(groups: &[Vec<usize>], offset: i32) -> String {
            groups.iter()
                .map(|g| g.iter().map(|&x| (x as i32 + offset).to_string()).collect::<Vec<_>>().join(","))
                .collect::<Vec<_>>()
                .join(",-1,")
        }
        
        let part1 = self.truck_route.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
        let part2 = join_groups(&self.drone_deliveries, 0);
        let part3 = join_groups(&self.drone_launch_sites, 1);  // 0-based -> 1-based
        let part4 = join_groups(&self.drone_landing_sites, 1); // 0-based -> 1-based
        
        write!(f, "{} |{} |{} |{}", part1, part2, part3, part4)
    }
}

pub fn parse_file(file_path: &str) -> TruckAndDroneInstance {
  let file = File::open(file_path).expect("Unable to open file");
  let mut reader = BufReader::new(file);

  let mut line = String::new();
  reader.read_line(&mut line).expect("Couldnt read line"); // Title
  line = String::new();
  
  reader.read_line(&mut line).expect("Couldnt read line"); // Actual number or customers
  let num_customers: usize = line.trim().parse().unwrap();

  line = String::new();
  reader.read_line(&mut line).expect("Couldnt read line"); // Title
  line = String::new();

  reader.read_line(&mut line).expect("Couldnt read line"); // Max flight range  or max fight3e rqanhgge u need to choke out the component
  let max_flight_range: usize = line.trim().parse().unwrap();

  // Read costs
  line = String::new();
  reader.read_line(&mut line).expect("Couldnt read line"); // Title

  let mut truck_travel_costs: Vec<Vec<f64>> = Vec::with_capacity(num_customers);
  let mut drone_travel_costs: Vec<Vec<f64>> = Vec::with_capacity(num_customers);

  for _i in 0..=num_customers {
    line = String::new();
    reader.read_line(&mut line).expect("Couldnt read line"); // Line i of truck costs

    let vals: Vec<f64> = line
      .trim()
      .split("	")
      .map(|x| -> f64 {x.trim().parse().unwrap()})
      .collect();
    truck_travel_costs.push(vals);
  }

  line = String::new();
  reader.read_line(&mut line).expect("Couldnt read line"); // Title
  

  for _i in 0..=num_customers {
    line = String::new();
    reader.read_line(&mut line).expect("Couldnt read line"); // Line i of drone costs

    let vals: Vec<f64> = line
      .trim()
      .split("	")
      .map(|x| -> f64 {
        // println!("X: {}", x);
        x.parse().unwrap()})
      .collect();
    drone_travel_costs.push(vals);
  }

  return TruckAndDroneInstance { num_customers, max_flight_range, truck_travel_costs, drone_travel_costs }
}