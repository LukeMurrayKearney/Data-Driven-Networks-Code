use crate::dpln::sample;
use statrs::distribution::{Poisson, Gamma};
use rand::{distributions::Distribution, rngs::ThreadRng};
use rand_distr::Binomial;



pub fn degrees_from_params(partitions: &Vec<usize>, group_sizes: &Vec<usize>, dist_type: &str, params: &Vec<Vec<f64>>, rng: &mut ThreadRng) -> Vec<usize> {
    let mut degrees: Vec<usize> = Vec::new();
    match dist_type {
        "dpln" => {
            for (i,_) in partitions.iter().enumerate() {
                let alpha = params[0][i];
                let beta = params[1][i];
                let nu = params[2][i];
                let tau = params[3][i];
                degrees.append(&mut sample(vec![alpha, beta, nu, tau], group_sizes[i]).iter().map(|x| 
                    if x.round() < 0. {
                        0
                    }
                    else {
                        x.round() as usize
                    }
                ).collect::<Vec<usize>>());
            }
        }
        _ => {
            for (i, _) in partitions.iter().enumerate() {
                degrees.append(&mut (0..group_sizes[i])
                    .map(|_| {
                        nbinom_sample(params[0][i], params[1][i], rng)
                    })
                    .collect());
            }
        }
    }
    degrees
}

pub fn create_frequency_distribution(adjacency_matrix: &Vec<Vec<(usize, usize)>>, ages: &Vec<usize>) -> Vec<Vec<usize>> {
    let mut frequency_distribution: Vec<Vec<usize>> = vec![vec![0; ages.last().unwrap() + 1]; ages.len()];
    for (i, ego) in adjacency_matrix.iter().enumerate() {
        for (_, contact_idx) in ego.iter() {
            frequency_distribution[i][ages[*contact_idx]] += 1;
        }
    }
    return frequency_distribution
}

pub fn nbinom_sample(r: f64, p: f64, rng: &mut ThreadRng) -> usize {
    // Sample from the Gamma Distribution 
    let gamma = Gamma::new(r, p/(1.0-p)).unwrap();
    let lambda = gamma.sample(rng);
    // Use sampled value as a parameter for the Poisson distribution
    let poisson = Poisson::new(lambda).unwrap();
    poisson.sample(rng) as usize
}

pub fn multinomial_sample(n: usize, ps: &Vec<f64>, rng: &mut ThreadRng) -> Vec<usize> {
    let mut x_sum: usize = 0;
    let mut p_sum: f64 = 0.0;
    ps.iter()
        .map(|&p| {
            // check if p or n equal 0
            if n - x_sum == 0 || p == 0.0 || 1.0 - p_sum <= f64::EPSILON {
                return 0
            }
            let mut p_cond: f64 = p / (1.0 - p_sum);
            if p_cond > 1.0 {
                p_cond = 1.0;
            }
            // make each binomial sample conditional on the last
            if Binomial::new((n-x_sum) as u64, p_cond).is_err() {
                return 0
            }
            let bin = Binomial::new((n - x_sum) as u64, p_cond).unwrap();
            p_sum += p;
            let x: usize = bin.sample(rng) as usize;
            x_sum += x;
            x
        })
        .collect()
}

pub fn rates_to_row_probabilities(rates_mat: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    rates_mat
        .iter()
        .map(|row| {
            let sum_row: f64 = row.iter().sum();
            row.iter().map(|&val| val/sum_row).collect()
        })
        .collect()
}

pub fn rates_to_probabilities(rates_mat: Vec<Vec<f64>>, partitions: &Vec<usize>) -> Vec<Vec<f64>> {
    
    // find consecutive group sizes to turn rates to probabilities
    let mut group_sizes: Vec<usize> = partitions
        .windows(2)
        .map(|pair| {
            pair[1] - pair[0]
        })
        .collect();
    group_sizes.insert(0,partitions[0]);
    
    // transform rates matrix to probability matrix 
    rates_mat
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter().enumerate().map(|(_, rate)| {
                rate / (group_sizes[i] as f64)
            })
            .collect()
        })
        .collect()
}

pub fn median(vals: &mut Vec<f64>) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = vals.len();
    if len % 2 == 0 {
        (vals[len / 2 - 1] + vals[len / 2]) / 2.0
    } else {
        vals[len / 2]
    }
}