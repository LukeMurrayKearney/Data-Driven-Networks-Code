use std::collections::{HashMap, HashSet, VecDeque};
use rand::Rng;
use rand::rngs::ThreadRng;
use crate::distributions::*;
// use crate::read_in::read_rates_mat;
use crate::connecting_stubs::*;
// use serde::Serialize;
use pyo3::prelude::*;



// #[derive(Debug, Clone, Serialize)]
#[derive(Debug, Clone)]
#[pyclass]
pub struct NetworkStructure {
    pub adjacency_matrix: Vec<Vec<(usize, usize)>>,
    pub degrees: Vec<usize>,
    pub ages: Vec<usize>,
    pub frequency_distribution: Vec<Vec<usize>>,
    pub partitions: Vec<usize>,
}

#[pyclass]
pub struct NetworkStructureDuration {
    pub adjacency_matrix: Vec<Vec<(usize, usize, usize)>>,
    pub degrees: Vec<Vec<usize>>,
    pub ages: Vec<usize>,
    pub frequency_distribution: Vec<Vec<Vec<usize>>>,
    pub partitions: Vec<usize>,
}

impl NetworkStructureDuration {

    pub fn transform(&mut self, props: &Vec<f64>) {

        let mut new_degrees = vec![vec![0; 5]; self.degrees.len()];
        let mut new_freq_dist: Vec<Vec<Vec<usize>>> = vec![vec![vec![0; 5]; self.ages.last().unwrap() + 1]; self.ages.len()];
        let mut new_adj_matrix: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); self.ages.len()];
        let mut rng = rand::thread_rng();
        for i in 0..self.adjacency_matrix.len() {
            for (_, j, d) in self.adjacency_matrix[i].iter() {
                let mut tmp_d = *d + 2;
                if *d == 0 {
                    let x = rng.gen::<f64>();
                    if x < props[0] {
                        tmp_d = 0;
                    }
                    else if x < props[0] + props[1] {
                        tmp_d = 1;
                    }
                    else {
                        tmp_d = 2;
                    }
                }
                new_degrees[*j][tmp_d] += 1;
                new_freq_dist[i][self.ages[*j]][tmp_d] += 1;
                new_adj_matrix[i].push((i, *j, tmp_d));
            }
        }
        self.degrees = new_degrees;
        self.frequency_distribution = new_freq_dist;
        self.adjacency_matrix = new_adj_matrix;
    }

    pub fn new_sbm_dur(n: usize, partitions: &Vec<usize>, contact_matrix: &Vec<Vec<Vec<f64>>>, num_durs: usize) -> NetworkStructureDuration {
        

        let mut rng: ThreadRng = rand::thread_rng();
        let mut edge_list: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); n];
        let mut degrees = vec![vec![0; num_durs];n];
        let prob_mat: Vec<Vec<Vec<f64>>> = contact_matrix.iter().map(|cm| rates_to_probabilities(cm.clone(), partitions)).collect();
        for i in 0..n {
            for j in 0..i {
                // find which block we are in
                let part_i = partitions
                    .iter()
                    .position(|&x| i<x)
                    .unwrap();
                let part_j = partitions
                    .iter()
                    .position(|&x| j<x)
                    .unwrap();
                // randomly generate edges with probability prob_mat
                let total_prob: f64 = (0..num_durs)
                    .map(|d| prob_mat[d][part_i][part_j])
                    .sum();

                if rng.gen::<f64>() < total_prob {
                    // edge exists
                    let u1 = rng.gen::<f64>()*total_prob;
                    let mut cumulative_prob = 0.0;
                    for duration in 0..num_durs {
                        cumulative_prob += prob_mat[duration][part_i][part_j];
                        if u1 < cumulative_prob {
                            edge_list[i].push((i, j, duration));
                            edge_list[j].push((j, i, duration));
                            degrees[i][duration] += 1;
                            degrees[j][duration] += 1;
                            break;
                        }
                    }
                }
            }
        }
        let mut last_idx = 0;
        let ages: Vec<usize> = partitions  
            .iter()
            .enumerate()
            .flat_map(|(i,x)| {
                let answer = vec![i; *x - last_idx];
                last_idx = *x;
                answer
            })
            .collect();
        let frequency_distribution: Vec<Vec<Vec<usize>>> = create_frequency_distribution_dur(&edge_list, &ages, num_durs);
        NetworkStructureDuration {
            adjacency_matrix: edge_list,
            degrees: degrees,
            ages: ages,
            frequency_distribution: frequency_distribution,
            partitions: partitions.clone(),
        }
    }

    
    pub fn new_from_dur_dist(partitions: &Vec<usize>, degree_age_breakdown: &Vec<Vec<usize>>, num_durs: usize) -> NetworkStructureDuration {

        let n_people = degree_age_breakdown.len();
        let mut degrees = vec![vec![0; num_durs];n_people];
        // let mut unconnected_stubs_breakdown: Vec<Vec<Vec<(usize,usize)>>> = vec![vec![Vec::new(); partitions.len()*num_durs]; partitions.len()*num_durs];
        let mut rng: ThreadRng = rand::thread_rng();
        let mut edge_list: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); n_people];
        let mut group_sizes: Vec<usize> = partitions
            .windows(2)
            .map(|pair| {
                pair[1] - pair[0]
            })
            .collect();
        group_sizes.insert(0,partitions[0]);

        // start connecting stubs
        let mut start_i: usize = 0;
        for (part_i, &part_i_end) in partitions.iter().enumerate() {
            let mut start_j:usize = 0;
            // go through partitions again only lower triangular 
            for (part_j, &part_j_end) in partitions.iter().enumerate().take(part_i+1) {
                // all degrees of partition i with partition j and vice versa
                let nodes_i: Vec<(usize, Vec<usize>)> = degree_age_breakdown
                    .iter()
                    .enumerate()
                    .skip(start_i)
                    .take(group_sizes[part_i])
                    .map(|(i, vec)| (i, (0..num_durs).map(|d| vec[part_j*num_durs + d]).collect()))
                    .collect();
                let nodes_j: Vec<(usize, Vec<usize>)> = degree_age_breakdown
                    .iter()
                    .enumerate()
                    .skip(start_j)
                    .take(group_sizes[part_j])
                    .map(|(j, vec)| (j, (0..num_durs).map(|d| vec[part_i*num_durs + d]).collect()))
                    .collect();
                // connect stubs one partition at a time
                let tmp_edges: Vec<(usize,usize,usize)>;
                // let stubs_remaining: (Vec<(usize,usize)>, Vec<(usize,usize)>); 
                if part_i == part_j {
                    tmp_edges = connect_stubs_diagonal_dur(&nodes_i, &mut rng);
                }
                else {
                    let (nodes_i, nodes_j) = balance_stubs_dur(&nodes_i, &nodes_j, &mut rng);
                    tmp_edges = connect_stubs_dur(&nodes_i, &nodes_j, &mut rng);
                }
                // add edges to sparse matrix
                for edge in tmp_edges.iter() {
                    edge_list[edge.0].push((edge.0, edge.1, edge.2));
                    edge_list[edge.1].push((edge.1, edge.0, edge.2));
                    degrees[edge.0][edge.2] += 1;
                    degrees[edge.1][edge.2] += 1;
                }
                start_j = part_j_end;
            }
            start_i = part_i_end;
        }

        // define age brackets
        let mut last_idx = 0;
        let ages: Vec<usize> = partitions  
            .iter()
            .enumerate()
            .flat_map(|(i,x)| {
                let answer = vec![i; *x - last_idx];
                last_idx = *x;
                answer
            })
            .collect();
        let frequency_distribution: Vec<Vec<Vec<usize>>> = create_frequency_distribution_dur(&edge_list, &ages, num_durs);

        NetworkStructureDuration {
            adjacency_matrix: edge_list,
            degrees: degrees,
            ages: ages,
            frequency_distribution: frequency_distribution,
            partitions: partitions.clone(),
        }
    }
}

impl NetworkStructure {

    pub fn new_dcsbm(partitions: &Vec<usize>, degree_correction: &Vec<f64>, contact_matrix: &Vec<Vec<f64>>) -> NetworkStructure {
        
        // transform contact matrix to a matrix of probabilities
        let n = *partitions.last().unwrap();
        let prob_mat: Vec<Vec<f64>> = rates_to_probabilities(contact_matrix.clone(), partitions);
        let mut rng: ThreadRng = rand::thread_rng();
        let mut edge_list: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        let mut degrees: Vec<usize> = vec![0; n];
        for i in 0..n {
            for j in 0..i {
                // find which block we are in
                let part_i = partitions
                    .iter()
                    .position(|&x| (i < x))
                    .unwrap();
                let part_j = partitions
                    .iter()
                    .position(|&x| (j < x))
                    .unwrap();
                // randomly generate edges with probability prob_mat
                if rng.gen::<f64>() < prob_mat[part_i][part_j]*degree_correction[i]*degree_correction[j] {
                    edge_list[i].push((i, j));
                    edge_list[j].push((j, i));
                    degrees[i] += 1;
                    degrees[j] += 1;
                }
            }
        }
        let mut last_idx = 0;
        let ages: Vec<usize> = partitions  
            .iter()
            .enumerate()
            .flat_map(|(i,x)| {
                let answer = vec![i; *x - last_idx];
                last_idx = *x;
                answer
            })
            .collect();
        let frequency_distribution: Vec<Vec<usize>> = create_frequency_distribution(&edge_list, &ages);

        NetworkStructure {
            adjacency_matrix: edge_list,
            degrees: degrees,
            ages: ages,
            frequency_distribution: frequency_distribution,
            partitions: partitions.clone(),
        }
    }

    pub fn new_er(partitions: &Vec<usize>, mean_degree: f64) -> NetworkStructure {

        let n = *partitions.last().unwrap();
        let mut rng: ThreadRng = rand::thread_rng();
        let mut edge_list: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        let mut degrees = vec![0;n];
        for i in 0..n {
            for j in 0..i {
                let prob = mean_degree / (n as f64);
                // randomly generate edges with probability prob_mat
                if rng.gen::<f64>() < prob {
                    edge_list[i].push((i, j));
                    edge_list[j].push((j, i));
                    degrees[i] += 1;
                    degrees[j] += 1;
                }
            }
        }
        // define age brackets
        let mut last_idx = 0;
        let ages: Vec<usize> = partitions  
            .iter()
            .enumerate()
            .flat_map(|(i,x)| {
                let answer = vec![i; *x - last_idx];
                last_idx = *x;
                answer
            })
            .collect();
        let frequency_distribution: Vec<Vec<usize>> = create_frequency_distribution(&edge_list, &ages);

        NetworkStructure {
            adjacency_matrix: edge_list,
            degrees: degrees,
            ages: ages,
            frequency_distribution: frequency_distribution,
            partitions: partitions.clone(),
        }
    }

    pub fn new_from_degree_dist(partitions: &Vec<usize>, degree_age_breakdown: &Vec<Vec<usize>>) -> NetworkStructure {

        let n = degree_age_breakdown.len();
        let mut degrees = vec![0;n];
        let mut start_i: usize = 0;
        let mut unconnected_stubs_breakdown: Vec<Vec<Vec<(usize,usize)>>> = vec![vec![Vec::new(); partitions.len()]; partitions.len()];
        let mut rng: ThreadRng = rand::thread_rng();
        let mut edge_list: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        let mut group_sizes: Vec<usize> = partitions
            .windows(2)
            .map(|pair| {
                pair[1] - pair[0]
            })
            .collect();
        
        group_sizes.insert(0,partitions[0]);
        
        // start connecting stubs
        for (part_i, &part_i_end) in partitions.iter().enumerate() {
            let mut start_j:usize = 0;
            // go through partitions again only lower triangular 
            for (part_j, &part_j_end) in partitions.iter().enumerate().take(part_i+1) {
                // all degrees of partition i with partition j and vice versa
                let nodes_i: Vec<(usize, usize)> = degree_age_breakdown
                    .iter()
                    .enumerate()
                    .skip(start_i)
                    .take(group_sizes[part_i])
                    .map(|(i, vec)| (i, vec[part_j]))
                    .collect();
                let nodes_j: Vec<(usize, usize)> = degree_age_breakdown
                    .iter()
                    .enumerate()
                    .skip(start_j)
                    .take(group_sizes[part_j])
                    .map(|(j, vec)| (j, vec[part_i]))
                    .collect();
                // connect stubs one partition at a time
                let tmp_edges: Vec<(usize,usize)>;
                let stubs_remaining: (Vec<(usize,usize)>, Vec<(usize,usize)>); 
                if part_i == part_j {
                    (tmp_edges, stubs_remaining) = connect_stubs_diagonal(&nodes_i, &mut rng);
                }
                else {
                    let (nodes_i, nodes_j) = balance_stubs(&nodes_i, &nodes_j, &mut rng);
                    (tmp_edges, stubs_remaining) = connect_stubs(&nodes_i, &nodes_j, &mut rng);
                }
                // save lists of unconnected stubs
                unconnected_stubs_breakdown[part_i][part_j] = stubs_remaining.0;
                unconnected_stubs_breakdown[part_j][part_i] = stubs_remaining.1;
                // add edges to sparse matrix
                for pair in tmp_edges.iter() {
                    edge_list[pair.0].push((pair.0, pair.1));
                    edge_list[pair.1].push((pair.1, pair.0));
                    degrees[pair.0] += 1;
                    degrees[pair.1] += 1;
                }
                start_j = part_j_end;
            }
            start_i = part_i_end;
        }
        // // attempt to connect remaining with neighbours of target
        // let mut tmp_edges: Vec<(usize, usize)> = Vec::new();
        // let mut source: Vec<(usize, usize)> = Vec::new();
        // let mut target1: Vec<(usize, usize)> = Vec::new();
        // let mut target2: Vec<(usize, usize)> = Vec::new();
        // // create a vector for iterating through partitions
        // let mut parts_iterable: Vec<usize> = vec![0]; 
        // parts_iterable.extend_from_slice(&partitions);

        // let num_groups: usize = partitions.len();
        // for i in 0..num_groups {
        //     // don't try to connect neighbours if there is less than 3 age groups 
        //     if num_groups < 3 {continue}
        //     for j in 0..num_groups {
        //         let mut old_edge_list: Vec<(usize, usize)> = Vec::new();
        //         for index in parts_iterable[i]..parts_iterable[i+1] {
        //             old_edge_list.append(&mut edge_list[index].clone());
        //         }
        //         match j {
        //             0 => {
        //                 for index in parts_iterable[j+1]..parts_iterable[j+2] {
        //                     old_edge_list.append(&mut edge_list[index].clone());
        //                 }
        //                 (tmp_edges, source, target1) = cleanup_single(&unconnected_stubs_breakdown[i][j], &unconnected_stubs_breakdown[j+1][i], &old_edge_list, &mut rng);
        //                 unconnected_stubs_breakdown[i][j] = source;
        //                 unconnected_stubs_breakdown[j+1][i] = target1;
        //             },
        //             // a rust thing
        //             temporary if temporary == (num_groups-1) => {
        //                 for index in parts_iterable[j-1]..parts_iterable[j] {
        //                     old_edge_list.append(&mut edge_list[index].clone());
        //                 }
        //                 (tmp_edges, source, target1) = cleanup_single(&unconnected_stubs_breakdown[i][j], &unconnected_stubs_breakdown[j-1][i], &old_edge_list, &mut rng);
        //                 unconnected_stubs_breakdown[i][j] = source;
        //                 unconnected_stubs_breakdown[j-1][i] = target1;
        //             },
        //             _ => {
        //                 for index in parts_iterable[j-1]..parts_iterable[j] {
        //                     old_edge_list.append(&mut edge_list[index].clone());
        //                 }
        //                 for index in parts_iterable[j+1]..parts_iterable[j+2] {
        //                     old_edge_list.append(&mut edge_list[index].clone());
        //                 }
                        
        //                 // ADD OLD EDGES TO THIS CALCULATION TO MAKE SURE THERE ARE NO DOUBLE EDGES
        //                 (tmp_edges, source, target1, target2) = cleanup_double(&unconnected_stubs_breakdown[i][j], &unconnected_stubs_breakdown[j-1][i], &unconnected_stubs_breakdown[j+1][i], &mut rng);
        //                 unconnected_stubs_breakdown[i][j] = source;
        //                 unconnected_stubs_breakdown[j-1][i] = target1;
        //                 unconnected_stubs_breakdown[j+1][i] = target2;
        //             }
        //         }
                
        //         for pair in tmp_edges.iter() {
        //             edge_list[pair.0].push((pair.0, pair.1));
        //             edge_list[pair.1].push((pair.1, pair.0));
        //             degrees[pair.0] += 1;
        //             degrees[pair.1] += 1;
        //         }
        //     }
        // }
    
        // define age brackets
        let mut last_idx = 0;
        let ages: Vec<usize> = partitions  
            .iter()
            .enumerate()
            .flat_map(|(i,x)| {
                let answer = vec![i; *x - last_idx];
                last_idx = *x;
                answer
            })
            .collect();
        let frequency_distribution: Vec<Vec<usize>> = create_frequency_distribution(&edge_list, &ages);

        NetworkStructure {
            adjacency_matrix: edge_list,
            degrees: degrees,
            ages: ages,
            frequency_distribution: frequency_distribution,
            partitions: partitions.clone(),
        }
    }

    pub fn new_mult_from_input(n:usize, partitions: &Vec<usize>, dist_type: &str, params: &Vec<Vec<f64>>, contact_matrix: &Vec<Vec<f64>>) -> NetworkStructure {
        
        let mut rng: ThreadRng = rand::thread_rng();
        let mut edge_list: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        let mut group_sizes: Vec<usize> = partitions
            .windows(2)
            .map(|pair| {
                pair[1] - pair[0]
            })
            .collect();
        
        group_sizes.insert(0,partitions[0]);
        // transform contact matrix to a matrix of probabilities
        let probs: Vec<Vec<f64>> = rates_to_row_probabilities(contact_matrix.clone());
        // sample degrees from age degrees distributions
        let mut degrees = degrees_from_params(&partitions, &group_sizes, dist_type, &params, &mut rng);
        // println!("{:?}", (degrees.iter().sum::<usize>() as f64)/(degrees.len() as f64));
        // assigning all stubs to age groups
        let mut start: usize = 0;
        let mut degree_age_breakdown: Vec<Vec<usize>> = Vec::new();
        for (i, x) in partitions.iter().enumerate() {
            for j in start..*x {
                degree_age_breakdown.push(multinomial_sample(degrees[j], &probs[i], &mut rng));
            }
            start = *x;
        }
        //reset degrees
        degrees = vec![0;n];
        let mut start_i: usize = 0;
        let mut unconnected_stubs_breakdown: Vec<Vec<Vec<(usize,usize)>>> = vec![vec![Vec::new(); partitions.len()]; partitions.len()];
        
        // start connecting stubs
        for (part_i, &part_i_end) in partitions.iter().enumerate() {
            let mut start_j:usize = 0;
            // go through partitions again only lower triangular 
            for (part_j, &part_j_end) in partitions.iter().enumerate().take(part_i+1) {
                // all degrees of partition i with partition j and vice versa
                let nodes_i: Vec<(usize, usize)> = degree_age_breakdown
                    .iter()
                    .enumerate()
                    .skip(start_i)
                    .take(group_sizes[part_i])
                    .map(|(i, vec)| (i, vec[part_j]))
                    .collect();
                let nodes_j: Vec<(usize, usize)> = degree_age_breakdown
                    .iter()
                    .enumerate()
                    .skip(start_j)
                    .take(group_sizes[part_j])
                    .map(|(j, vec)| (j, vec[part_i]))
                    .collect();
                // connect stubs one partition at a time
                let tmp_edges: Vec<(usize,usize)>;
                let stubs_remaining: (Vec<(usize,usize)>, Vec<(usize,usize)>); 
                if part_i == part_j {
                    (tmp_edges, stubs_remaining) = connect_stubs_diagonal(&nodes_i, &mut rng);
                }
                else {
                    (tmp_edges, stubs_remaining) = connect_stubs(&nodes_i, &nodes_j, &mut rng);
                }
                // save lists of unconnected stubs
                unconnected_stubs_breakdown[part_i][part_j] = stubs_remaining.0;
                unconnected_stubs_breakdown[part_j][part_i] = stubs_remaining.1;
                // add edges to sparse matrix
                for pair in tmp_edges.iter() {
                    edge_list[pair.0].push((pair.0, pair.1));
                    edge_list[pair.1].push((pair.1, pair.0));
                    degrees[pair.0] += 1;
                    degrees[pair.1] += 1;
                }
                start_j = part_j_end;
            }
            start_i = part_i_end;
        }
        // // attempt to connect remaining with neighbours of target
        // let mut tmp_edges: Vec<(usize, usize)> = Vec::new();
        // let mut source: Vec<(usize, usize)> = Vec::new();
        // let mut target1: Vec<(usize, usize)> = Vec::new();
        // let mut target2: Vec<(usize, usize)> = Vec::new();
        // // create a vector for iterating through partitions
        // let mut parts_iterable: Vec<usize> = vec![0]; 
        // parts_iterable.extend_from_slice(&partitions);

        // let num_groups: usize = partitions.len();
        // for i in 0..num_groups {
        //     // don't try to connect neighbours if there is less than 3 age groups 
        //     if num_groups < 3 {continue}
        //     for j in 0..num_groups {
        //         let mut old_edge_list: Vec<(usize, usize)> = Vec::new();
        //         for index in parts_iterable[i]..parts_iterable[i+1] {
        //             old_edge_list.append(&mut edge_list[index].clone());
        //         }
        //         match j {
        //             0 => {
        //                 for index in parts_iterable[j+1]..parts_iterable[j+2] {
        //                     old_edge_list.append(&mut edge_list[index].clone());
        //                 }
        //                 (tmp_edges, source, target1) = cleanup_single(&unconnected_stubs_breakdown[i][j], &unconnected_stubs_breakdown[j+1][i], &old_edge_list, &mut rng);
        //                 unconnected_stubs_breakdown[i][j] = source;
        //                 unconnected_stubs_breakdown[j+1][i] = target1;
        //             },
        //             // a rust thing
        //             temporary if temporary == (num_groups-1) => {
        //                 for index in parts_iterable[j-1]..parts_iterable[j] {
        //                     old_edge_list.append(&mut edge_list[index].clone());
        //                 }
        //                 (tmp_edges, source, target1) = cleanup_single(&unconnected_stubs_breakdown[i][j], &unconnected_stubs_breakdown[j-1][i], &old_edge_list, &mut rng);
        //                 unconnected_stubs_breakdown[i][j] = source;
        //                 unconnected_stubs_breakdown[j-1][i] = target1;
        //             },
        //             _ => {
        //                 for index in parts_iterable[j-1]..parts_iterable[j] {
        //                     old_edge_list.append(&mut edge_list[index].clone());
        //                 }
        //                 for index in parts_iterable[j+1]..parts_iterable[j+2] {
        //                     old_edge_list.append(&mut edge_list[index].clone());
        //                 }
                        
        //                 // ADD OLD EDGES TO THIS CALCULATION TO MAKE SURE THERE ARE NO DOUBLE EDGES
        //                 (tmp_edges, source, target1, target2) = cleanup_double(&unconnected_stubs_breakdown[i][j], &unconnected_stubs_breakdown[j-1][i], &unconnected_stubs_breakdown[j+1][i], &mut rng);
        //                 unconnected_stubs_breakdown[i][j] = source;
        //                 unconnected_stubs_breakdown[j-1][i] = target1;
        //                 unconnected_stubs_breakdown[j+1][i] = target2;
        //             }
        //         }
                
        //         for pair in tmp_edges.iter() {
        //             edge_list[pair.0].push((pair.0, pair.1));
        //             edge_list[pair.1].push((pair.1, pair.0));
        //             degrees[pair.0] += 1;
        //             degrees[pair.1] += 1;
        //         }
        //     }
        // }
    
        // define age brackets
        let mut last_idx = 0;
        let ages: Vec<usize> = partitions  
            .iter()
            .enumerate()
            .flat_map(|(i,x)| {
                let answer = vec![i; *x - last_idx];
                last_idx = *x;
                answer
            })
            .collect();
        let frequency_distribution: Vec<Vec<usize>> = create_frequency_distribution(&edge_list, &ages);

        NetworkStructure {
            adjacency_matrix: edge_list,
            degrees: degrees,
            ages: ages,
            frequency_distribution: frequency_distribution,
            partitions: partitions.clone(),
        }
    }


    pub fn new_sbm_from_vars(n: usize, partitions: &Vec<usize>, contact_matrix: &Vec<Vec<f64>>) -> NetworkStructure {
        
        // transform contact matrix to a matrix of probabilities
        let prob_mat: Vec<Vec<f64>> = rates_to_probabilities(contact_matrix.clone(), partitions);
        let mut rng: ThreadRng = rand::thread_rng();
        let mut edge_list: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        let mut degrees: Vec<usize> = vec![0; n];
        for i in 0..n {
            for j in 0..i {
                // find which block we are in
                let part_i = partitions
                    .iter()
                    .position(|&x| (i/x) < 1)
                    .unwrap();
                let part_j = partitions
                    .iter()
                    .position(|&x| (j/x) < 1)
                    .unwrap();
                // randomly generate edges with probability prob_mat
                if rng.gen::<f64>() < prob_mat[part_i][part_j] {
                    edge_list[i].push((i, j));
                    edge_list[j].push((j, i));
                    degrees[i] += 1;
                    degrees[j] += 1;
                }
            }
        }
        let mut last_idx = 0;
        let ages: Vec<usize> = partitions  
            .iter()
            .enumerate()
            .flat_map(|(i,x)| {
                let answer = vec![i; *x - last_idx];
                last_idx = *x;
                answer
            })
            .collect();
        let frequency_distribution: Vec<Vec<usize>> = create_frequency_distribution(&edge_list, &ages);

        NetworkStructure {
            adjacency_matrix: edge_list,
            degrees: degrees,
            ages: ages,
            frequency_distribution: frequency_distribution,
            partitions: partitions.clone(),
        }
    }
}

pub fn balance_stubs_dur(nodes_i: &Vec<(usize, Vec<usize>)>, nodes_j: &Vec<(usize, Vec<usize>)>, rng: &mut ThreadRng) -> (Vec<(usize, Vec<usize>)>, Vec<(usize, Vec<usize>)>) {
    
    let mut nodes_i = nodes_i.clone();
    let mut nodes_j = nodes_j.clone();
    
    for dur in 0..nodes_i[0].1.len() {
        let total_i = nodes_i.iter().map(|(_, vec)| vec[dur]).sum::<usize>() as f64;
        let total_j = nodes_j.iter().map(|(_, vec)| vec[dur]).sum::<usize>() as f64;
        if total_i < total_j {
            let scale = (total_i + total_j)/(2.0*total_i);
            for idx in 0..nodes_i.len() {
                let new_val = (nodes_i[idx].1[dur] as f64)*scale;
                nodes_i[idx].1[dur] = stochastic_round(new_val, rng);
            }
        }
        else {
            let scale = (total_i + total_j)/(2.0*total_j);
            for idx in 0..nodes_j.len() {
                let new_val = (nodes_j[idx].1[dur] as f64)*scale;
                nodes_j[idx].1[dur] = stochastic_round(new_val, rng);
            }
        }
    }
    (nodes_i, nodes_j)
}

pub fn balance_stubs(nodes_i: &Vec<(usize, usize)>, nodes_j: &Vec<(usize, usize)>, rng: &mut ThreadRng) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    
    let mut nodes_i = nodes_i.clone();
    let mut nodes_j = nodes_j.clone();
    
    let total_i = nodes_i.iter().map(|(_, val)| val).sum::<usize>() as f64;
    let total_j = nodes_j.iter().map(|(_, val)| val).sum::<usize>() as f64;
    if total_i < total_j {
        let scale = (total_i + total_j)/(2.0*total_i);
        for idx in 0..nodes_i.len() {
            let new_val = (nodes_i[idx].1 as f64)*scale;
            nodes_i[idx].1 = stochastic_round(new_val, rng);
        }
    }
    else {
        let scale = (total_i + total_j)/(2.0*total_j);
        for idx in 0..nodes_j.len() {
            let new_val = (nodes_j[idx].1 as f64)*scale;
            nodes_j[idx].1 = stochastic_round(new_val, rng);
        }
    }
    (nodes_i, nodes_j)
}

fn stochastic_round(x: f64, rng: &mut ThreadRng) -> usize {
    let int_part = x.floor() as usize;
    let frac_part = x - (int_part as f64);
    if rng.gen::<f64>() < frac_part {
        int_part + 1
    }
    else {
        int_part
    }
}

pub fn largest_cc(edges: Vec<Vec<(usize, usize)>>) -> usize {
    // Build adjacency list
    let mut graph: HashMap<usize, Vec<usize>> = HashMap::new();
    for (u, row) in edges.iter().enumerate() {
        for (_,v) in row.iter() {
            if u < *v {
                graph.entry(u).or_default().push(*v);
                graph.entry(*v).or_default().push(u);
            }
        }
    }

    let mut visited = HashSet::new();
    let mut max_size = 0;

    // BFS/DFS from each unvisited node
    for &node in graph.keys() {
        if !visited.contains(&node) {
            let size = bfs_component_size(node, &graph, &mut visited);
            max_size = max_size.max(size);
        }
    }

    max_size
}

/// Helper: BFS to compute component size
fn bfs_component_size(
    start: usize,
    graph: &HashMap<usize, Vec<usize>>,
    visited: &mut HashSet<usize>,
) -> usize {
    let mut queue = VecDeque::new();
    queue.push_back(start);
    visited.insert(start);

    let mut count = 1;

    while let Some(node) = queue.pop_front() {
        if let Some(neighbors) = graph.get(&node) {
            for &nbr in neighbors {
                if !visited.contains(&nbr) {
                    visited.insert(nbr);
                    queue.push_back(nbr);
                    count += 1;
                }
            }
        }
    }

    count
}