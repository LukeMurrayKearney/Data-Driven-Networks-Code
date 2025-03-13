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
    pub partitions: Vec<usize>
}

impl NetworkStructure {

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
        // attempt to connect remaining with neighbours of target
        let mut tmp_edges: Vec<(usize, usize)> = Vec::new();
        let mut source: Vec<(usize, usize)> = Vec::new();
        let mut target1: Vec<(usize, usize)> = Vec::new();
        let mut target2: Vec<(usize, usize)> = Vec::new();
        // create a vector for iterating through partitions
        let mut parts_iterable: Vec<usize> = vec![0]; 
        parts_iterable.extend_from_slice(&partitions);

        let num_groups: usize = partitions.len();
        for i in 0..num_groups {
            // don't try to connect neighbours if there is less than 3 age groups 
            if num_groups < 3 {continue}
            for j in 0..num_groups {
                let mut old_edge_list: Vec<(usize, usize)> = Vec::new();
                for index in parts_iterable[i]..parts_iterable[i+1] {
                    old_edge_list.append(&mut edge_list[index].clone());
                }
                match j {
                    0 => {
                        for index in parts_iterable[j+1]..parts_iterable[j+2] {
                            old_edge_list.append(&mut edge_list[index].clone());
                        }
                        (tmp_edges, source, target1) = cleanup_single(&unconnected_stubs_breakdown[i][j], &unconnected_stubs_breakdown[j+1][i], &old_edge_list, &mut rng);
                        unconnected_stubs_breakdown[i][j] = source;
                        unconnected_stubs_breakdown[j+1][i] = target1;
                    },
                    // a rust thing
                    temporary if temporary == (num_groups-1) => {
                        for index in parts_iterable[j-1]..parts_iterable[j] {
                            old_edge_list.append(&mut edge_list[index].clone());
                        }
                        (tmp_edges, source, target1) = cleanup_single(&unconnected_stubs_breakdown[i][j], &unconnected_stubs_breakdown[j-1][i], &old_edge_list, &mut rng);
                        unconnected_stubs_breakdown[i][j] = source;
                        unconnected_stubs_breakdown[j-1][i] = target1;
                    },
                    _ => {
                        for index in parts_iterable[j-1]..parts_iterable[j] {
                            old_edge_list.append(&mut edge_list[index].clone());
                        }
                        for index in parts_iterable[j+1]..parts_iterable[j+2] {
                            old_edge_list.append(&mut edge_list[index].clone());
                        }
                        
                        // ADD OLD EDGES TO THIS CALCULATION TO MAKE SURE THERE ARE NO DOUBLE EDGES
                        (tmp_edges, source, target1, target2) = cleanup_double(&unconnected_stubs_breakdown[i][j], &unconnected_stubs_breakdown[j-1][i], &unconnected_stubs_breakdown[j+1][i], &mut rng);
                        unconnected_stubs_breakdown[i][j] = source;
                        unconnected_stubs_breakdown[j-1][i] = target1;
                        unconnected_stubs_breakdown[j+1][i] = target2;
                    }
                }
                
                for pair in tmp_edges.iter() {
                    edge_list[pair.0].push((pair.0, pair.1));
                    edge_list[pair.1].push((pair.1, pair.0));
                    degrees[pair.0] += 1;
                    degrees[pair.1] += 1;
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
            partitions: partitions.clone()
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
            partitions: partitions.clone()
        }
    }
}
