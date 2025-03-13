use crate::network_structure::NetworkStructure;
use crate::run_model::{scale_fit, ScaleParams};
use rand::distributions::WeightedIndex;
use rand::seq::SliceRandom;
// use statrs::distribution::Poisson;
use rand_distr::{Distribution, Geometric};

#[derive(Clone, Debug)]
pub struct NetworkProperties {
    pub nodal_states: Vec<State>,
    pub parameters: Vec<f64>,
    pub secondary_cases: Vec<usize>,
    pub generation: Vec<usize>,
    pub disease_from: Vec<i64>,
}

#[derive(Clone,Debug,Copy,PartialEq)]
pub enum State {
    Susceptible,
    Infected(usize),
    Recovered
}

impl NetworkProperties {

    pub fn new(network: &NetworkStructure, params: &Vec<f64>) -> NetworkProperties {
        NetworkProperties {
            nodal_states: vec![State::Susceptible; network.degrees.len()],
            parameters: params.clone(),
            secondary_cases: vec![0; network.degrees.len()],
            generation: vec![0; network.degrees.len()],
            disease_from: vec![-1; network.degrees.len()],
        }
    }

    pub fn initialize_infection_sellke_rand(&mut self, proportion_of_population: f64) {

        let number_of_infecteds: usize = match proportion_of_population as usize {
            0..=1 => {
                ((self.nodal_states.len() as f64) * proportion_of_population) as usize
            },
            _ => {
                println!("The proportion infected must be between 0 and 1");
                0
            }
        };
        // define random number generator
        let mut rng = rand::thread_rng();

        // weighted index of each individual
        let mut indicies: Vec<usize> = (0..self.nodal_states.len()).collect();
        indicies.shuffle(&mut rng);

        // infect selected individuals
        for &i in indicies.iter().take(number_of_infecteds) {
            self.nodal_states[i] = State::Infected(0);
            // self.nodal_states[i] = State::Infected(poisson_infectious_period.sample(&mut rng).round() as usize);
            self.generation[i] = 1;
        }
    }

    pub fn initialize_infection_sellke(&mut self, network: &NetworkStructure, proportion_of_population: f64, scaling: &str) {

        let number_of_infecteds: usize = match proportion_of_population as usize {
            0..=1 => {
                ((self.nodal_states.len() as f64) * proportion_of_population) as usize
            },
            _ => {
                println!("The proportion infected must be between 0 and 1");
                0
            }
        };
        // define random number generator
        let mut rng = rand::thread_rng();

        
        //we want a weighted sampling of the population
        let probabilities: Vec<f64> = match scaling {
            "log" => {
                network.degrees.iter().map(|&degrees| ((degrees + 1) as f64).ln()).collect()
            },
            "sqrt" => {
                network.degrees.iter().map(|&degrees| ((degrees + 1) as f64).sqrt()).collect()
            },
            "linear" => {
                network.degrees.iter().map(|&degrees| ((degrees + 1) as f64)).collect()
            },
            // duration = A*(k^2)*exp(-Bk) + C/(k^D) + E/k 
            // max at 1, when k=1 we want f(k) = 1
            "fit1" => {
                let scale_params = ScaleParams::new(1.92943985e-01, 2.59700437e-01,4.55889377e04,9.99839680e-01,-4.55800575e04);
                // ScaleParams::new(
                    // vec![7.94597576e-02, 1.86501075e-02, 0.27331857, 2.71714397e-01, 1.50843120e-01, 0.22501698, 0.36498229,0.3402533,0.16861257],
                    // vec![1.86727109e-01, 1.57216691e-01, 0.26578535, 2.93698607e-01, 2.33321449e-01, 0.3025751, 0.32910101,0.32746589,0.23098462],
                    // vec![6.26714288e+04, 1.15274663e+05, 55.08602862, 1.01006951e+04, 2.61115659e+02, 123.57028328, 8.26235973,6.75177633,1.15683718], 
                    // vec![9.99749380e-01, 9.99844176e-01, 0.84128853, 9.99305162e-01, 9.71511671e-01, 0.93912153, 0.62576043,0.73053706,0.28046687],
                    // vec![-6.26598777e+04, -1.15266709e+05, -46.86310399, -1.00927696e+04, -2.51956809e+02, -115.13632869, 0.72628637,2.23407002,8.0201658]);
                network.degrees.iter().enumerate().map(|(_, &degrees)| {
                    if degrees == 0 {
                        0.
                    }
                    else {
                        1./scale_fit(&scale_params, degrees as f64)
                    }
                }).collect()
            },
            "fit2" => {
                let scale_params = ScaleParams::new(5.93853399e-02,1.81040353e-01,  1.08985503e+05,  9.99930465e-01, -1.08976101e+05);
                // ScaleParams::new(
                //     vec![6.87326840e-01, 7.38784056e-02, 5.77862944e-01, 2.13415641e-01, 3.29687844e-01, 3.85160330e-01, 3.24887201e-01,0.25216752,5.27970340e-02],
                //     vec![4.04847685e-01, 1.55839474e-01, 4.98656357e-01, 2.80893647e-01, 2.70297825e-01, 2.97786924e-01, 3.61132503e-01,0.35245878,1.71913746e-01],
                //     vec![9.50900768e+04, 1.94907963e+05, 4.96158310e+04, 5.52982066e+04, 2.69952333e+04, 2.34848788e+04, 7.52264235e+03,4.50828294,7.44934579e+03], 
                //     vec![9.99885451e-01, 9.99928224e-01, 9.99822099e-01, 9.99878834e-01, 9.99780538e-01, 9.99786125e-01, 9.99440989e-01,0.62107256,9.99746592e-01],
                //     vec![-9.50788242e+04, -1.94898582e+05, -4.96091268e+04, -5.52894549e+04, -2.69858020e+04, -2.34759408e+04, -7.51374782e+03,5.22464991,-7.43959415e+03]);        
                network.degrees.iter().enumerate().map(|(_, &degrees)| {
                    if degrees == 0 {
                        0.
                    }
                    else {
                        1./scale_fit(&scale_params, degrees as f64)
                    }
                }).collect()
            },
            _ => network.degrees.iter().map(|&degrees| ((degrees + 1) as f64)).collect()
        };

        // weighted index of each individual
        let dist = WeightedIndex::new(probabilities).unwrap();
        let selected: Vec<usize> = (0..number_of_infecteds).map(|_| dist.sample(&mut rng)).collect();

        // infect selected individuals
        for &i in selected.iter() {
            self.nodal_states[i] = State::Infected(0);
            // self.nodal_states[i] = State::Infected(poisson_infectious_period.sample(&mut rng).round() as usize);
            self.generation[i] = 1;
        }
    }

    pub fn initialize_infection_degree(&mut self, network: &NetworkStructure, proportion_of_population: f64, inv_gamma: f64, scaling: &str) {

        let number_of_infecteds: usize = match proportion_of_population as usize {
            0..=1 => {
                ((self.nodal_states.len() as f64) * proportion_of_population) as usize
            },
            _ => {
                println!("The proportion infected must be between 0 and 1");
                0
            }
        };

        // define infectious period sampler
        // let poisson_infectious_period = Poisson::new(inv_gamma).unwrap();
        let geom_infectious_period = Geometric::new(1./inv_gamma).unwrap();

        // define random number generator
        let mut rng = rand::thread_rng();

        
        //we want a weighted sampling of the population
        let probabilities: Vec<f64> = match scaling {
            "log" => {
                network.degrees.iter().map(|&degrees| ((degrees + 1) as f64).ln()).collect()
            },
            "sqrt" => {
                network.degrees.iter().map(|&degrees| ((degrees + 1) as f64).sqrt()).collect()
            },
            "linear" => {
                network.degrees.iter().map(|&degrees| ((degrees + 1) as f64)).collect()
            },
            // duration = A*(k^2)*exp(-Bk) + C/(k^D) + E/k 
            // max at 1, when k=1 we want f(k) = 1
            "fit1" => {
                let scale_params = ScaleParams::new(1.92943985e-01, 2.59700437e-01,4.55889377e04,9.99839680e-01,-4.55800575e04);
                // ScaleParams::new(
                    // vec![7.94597576e-02, 1.86501075e-02, 0.27331857, 2.71714397e-01, 1.50843120e-01, 0.22501698, 0.36498229,0.3402533,0.16861257],
                    // vec![1.86727109e-01, 1.57216691e-01, 0.26578535, 2.93698607e-01, 2.33321449e-01, 0.3025751, 0.32910101,0.32746589,0.23098462],
                    // vec![6.26714288e+04, 1.15274663e+05, 55.08602862, 1.01006951e+04, 2.61115659e+02, 123.57028328, 8.26235973,6.75177633,1.15683718], 
                    // vec![9.99749380e-01, 9.99844176e-01, 0.84128853, 9.99305162e-01, 9.71511671e-01, 0.93912153, 0.62576043,0.73053706,0.28046687],
                    // vec![-6.26598777e+04, -1.15266709e+05, -46.86310399, -1.00927696e+04, -2.51956809e+02, -115.13632869, 0.72628637,2.23407002,8.0201658]);
                network.degrees.iter().enumerate().map(|(_, &degrees)| {
                    if degrees == 0 {
                        0.
                    }
                    else {
                        1./scale_fit(&scale_params, degrees as f64)
                    }
                }).collect()
            },
            "fit2" => {
                let scale_params = ScaleParams::new(5.93853399e-02,1.81040353e-01,  1.08985503e+05,  9.99930465e-01, -1.08976101e+05);
                // ScaleParams::new(
                //     vec![6.87326840e-01, 7.38784056e-02, 5.77862944e-01, 2.13415641e-01, 3.29687844e-01, 3.85160330e-01, 3.24887201e-01,0.25216752,5.27970340e-02],
                //     vec![4.04847685e-01, 1.55839474e-01, 4.98656357e-01, 2.80893647e-01, 2.70297825e-01, 2.97786924e-01, 3.61132503e-01,0.35245878,1.71913746e-01],
                //     vec![9.50900768e+04, 1.94907963e+05, 4.96158310e+04, 5.52982066e+04, 2.69952333e+04, 2.34848788e+04, 7.52264235e+03,4.50828294,7.44934579e+03], 
                //     vec![9.99885451e-01, 9.99928224e-01, 9.99822099e-01, 9.99878834e-01, 9.99780538e-01, 9.99786125e-01, 9.99440989e-01,0.62107256,9.99746592e-01],
                //     vec![-9.50788242e+04, -1.94898582e+05, -4.96091268e+04, -5.52894549e+04, -2.69858020e+04, -2.34759408e+04, -7.51374782e+03,5.22464991,-7.43959415e+03]);        
                network.degrees.iter().enumerate().map(|(_, &degrees)| {
                    if degrees == 0 {
                        0.
                    }
                    else {
                        1./scale_fit(&scale_params, degrees as f64)
                    }
                }).collect()
            },
            _ => network.degrees.iter().map(|&degrees| ((degrees + 1) as f64)).collect()
        };

        // weighted index of each individual
        let dist = WeightedIndex::new(probabilities).unwrap();
        let selected: Vec<usize> = (0..number_of_infecteds).map(|_| dist.sample(&mut rng)).collect();

        // infect selected individuals
        for &i in selected.iter() {
            self.nodal_states[i] = State::Infected(geom_infectious_period.sample(&mut rng) as usize);
            // self.nodal_states[i] = State::Infected(poisson_infectious_period.sample(&mut rng).round() as usize);
            self.generation[i] = 1;
        }
    }

    // pub fn initialize_infection_random(&mut self, proportion_of_population: f64, inv_gamma: f64) {

    //     let number_of_infecteds: usize = match proportion_of_population as usize {
    //         0..=1 => {
    //             ((self.nodal_states.len() as f64) * proportion_of_population) as usize
    //         },
    //         _ => {
    //             println!("The proportion infected must be between 0 and 1");
    //             0
    //         }
    //     };

    //     // define infectious period sampler
    //     let poisson_infectious_period = Poisson::new(inv_gamma).unwrap();
    //     // define random number generator
    //     let mut rng = rand::thread_rng();
    //     // shuffle indices and choose
    //     let mut indices: Vec<usize> = (0..self.nodal_states.len()).collect();
    //     indices.shuffle(&mut rng);
    //     for i in 0..number_of_infecteds {
    //         self.nodal_states[indices[i]] = State::Infected(poisson_infectious_period.sample(&mut rng).round() as usize);
    //         self.generation[indices[i]] = 1;
    //     }
    // }

    pub fn count_states(&self) -> Vec<usize> {
        let mut result: Vec<usize> = vec![0; 3];
        for state in self.nodal_states.iter() {
            match state {
                State::Susceptible => result[0] += 1,
                State::Infected(_) => result[1] += 1,
                State::Recovered => result[2] += 1
            }
        }
        result
    }
    pub fn count_states_age(&self, network: &NetworkStructure) -> Vec<Vec<usize>> {
        let mut result: Vec<Vec<usize>> = vec![vec![0; 3]; network.ages.iter().max().unwrap().to_owned()+1];
        for (i, state) in self.nodal_states.iter().enumerate() {
            match state {
                &State::Susceptible => result[network.ages[i]][0] += 1,
                &State::Infected(_) => result[network.ages[i]][1] += 1,
                &State::Recovered => result[network.ages[i]][2] += 1
            }
        }
        result
    }
}

