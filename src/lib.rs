use core::num;
use std::os::linux::net;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::{iter, prelude::*, vec};
use crate::dpln::{pdf, sample};
use crate::network_structure::NetworkStructure;

mod network_structure;
mod distributions;
mod dpln;
mod connecting_stubs;
mod network_properties;
mod run_model;

////////////////////////////////////////////// Network Creation ////////////////////////////////////////

#[pyfunction]
fn network_dur(degree_age_breakdown: Vec<Vec<usize>>, partitions: Vec<usize>, num_dur: usize, props: Vec<f64>) -> PyResult<Py<PyDict>> {

    let mut network: network_structure::NetworkStructureDuration = network_structure::NetworkStructureDuration::new_from_dur_dist(&partitions, &degree_age_breakdown, num_dur);
    if props.len() > 0 {
        network.transform(&props);
    }
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);

        dict.set_item("adjacency_matrix", network.adjacency_matrix.to_object(py))?;
        dict.set_item("degrees", network.degrees.to_object(py))?;
        dict.set_item("ages", network.ages.to_object(py))?;
        dict.set_item("frequency_distribution", network.frequency_distribution.to_object(py))?;
        dict.set_item("partitions", network.partitions.to_object(py))?;

        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

//  Creates a netwrok from given source and targets
#[pyfunction]
fn network_from_source_and_targets(partitions: Vec<usize>, degree_dist: Vec<Vec<usize>>) -> PyResult<Py<PyDict>>  {
    
    // WRITE FUNCTION THAT TAKES IN DD AND PARTITIONS TO MAKE A NETWORK - JUST COPY END OF CURRENT METHOD
    let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_from_degree_dist(&partitions, &degree_dist);

    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("adjacency_matrix", network.adjacency_matrix.to_object(py))?;
        dict.set_item("degrees", network.degrees.to_object(py))?;
        dict.set_item("ages", network.ages.to_object(py))?;
        dict.set_item("frequency_distribution", network.frequency_distribution.to_object(py))?;
        dict.set_item("partitions", network.partitions.to_object(py))?;

        Ok(dict.into())
    })
}

//  Creates a network from given variables
#[pyfunction]
fn network_from_vars(n: usize, partitions: Vec<usize>, dist_type: &str, network_params: Vec<Vec<f64>>, contact_matrix: Vec<Vec<f64>>) -> PyResult<Py<PyDict>>  {
    
    let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_mult_from_input(n, &partitions, dist_type, &network_params, &contact_matrix);
    
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("adjacency_matrix", network.adjacency_matrix.to_object(py))?;
        dict.set_item("degrees", network.degrees.to_object(py))?;
        dict.set_item("ages", network.ages.to_object(py))?;
        dict.set_item("frequency_distribution", network.frequency_distribution.to_object(py))?;
        dict.set_item("partitions", network.partitions.to_object(py))?;

        Ok(dict.into())
    })
}

// Creates a SBM network
#[pyfunction]
fn sbm_from_vars(n: usize, partitions: Vec<usize>, contact_matrix: Vec<Vec<f64>>) -> PyResult<Py<PyDict>>  {
    
    let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_sbm_from_vars(n, &partitions, &contact_matrix);
    
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("adjacency_matrix", network.adjacency_matrix.to_object(py))?;
        dict.set_item("degrees", network.degrees.to_object(py))?;
        dict.set_item("ages", network.ages.to_object(py))?;
        dict.set_item("frequency_distribution", network.frequency_distribution.to_object(py))?;
        dict.set_item("partitions", network.partitions.to_object(py))?;

        Ok(dict.into())
    })
}

// Creates a SBM duration network
#[pyfunction]
fn sbm_duration(n: usize, partitions: Vec<usize>, contact_matrix: Vec<Vec<Vec<f64>>>, num_dur: usize, props: Vec<f64>) -> PyResult<Py<PyDict>>  {
    
    let mut network: network_structure::NetworkStructureDuration = network_structure::NetworkStructureDuration::new_sbm_dur(partitions.last().cloned().unwrap(), &partitions, &contact_matrix, num_dur);
    if props.len() > 0 {
        network.transform(&props);
    }
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("adjacency_matrix", network.adjacency_matrix.to_object(py))?;
        dict.set_item("degrees", network.degrees.to_object(py))?;
        dict.set_item("ages", network.ages.to_object(py))?;
        dict.set_item("frequency_distribution", network.frequency_distribution.to_object(py))?;
        dict.set_item("partitions", network.partitions.to_object(py))?;

        Ok(dict.into())
    })
}

// Creates an ER network
#[pyfunction]
fn build_ER(partitions: Vec<usize>, mean_degree: f64) -> PyResult<Py<PyDict>>  {
    
    let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_er(&partitions, mean_degree);
    
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("adjacency_matrix", network.adjacency_matrix.to_object(py))?;
        dict.set_item("degrees", network.degrees.to_object(py))?;
        dict.set_item("ages", network.ages.to_object(py))?;
        dict.set_item("frequency_distribution", network.frequency_distribution.to_object(py))?;
        dict.set_item("partitions", network.partitions.to_object(py))?;

        Ok(dict.into())
    })
}

// Creates an DCSBM network
#[pyfunction]
fn build_DCSBM(partitions: Vec<usize>, degree_correction: Vec<f64>, contact_matrix: Vec<Vec<f64>>) -> PyResult<Py<PyDict>>  {
    
    let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_dcsbm(&partitions, &degree_correction, &contact_matrix);
    
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("adjacency_matrix", network.adjacency_matrix.to_object(py))?;
        dict.set_item("degrees", network.degrees.to_object(py))?;
        dict.set_item("ages", network.ages.to_object(py))?;
        dict.set_item("frequency_distribution", network.frequency_distribution.to_object(py))?;
        dict.set_item("partitions", network.partitions.to_object(py))?;

        Ok(dict.into())
    })
}


//////////////////////////////////////////// outbreak simulation //////////////////////////////////////

#[pyfunction]
fn small_gillespie_dur(degree_age_breakdown: Vec<Vec<usize>>, tau: f64, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize, num_dur: usize, props: Vec<f64>) -> PyResult<Py<PyDict>> {
    
    let mut tmp_num_dur = num_dur;
    let mut cur_params = outbreak_params.clone();
    cur_params[0] = tau;
    let mut network: network_structure::NetworkStructureDuration = network_structure::NetworkStructureDuration::new_from_dur_dist(&partitions, &degree_age_breakdown, num_dur);
    if props.len() > 0 {
        network.transform(&props);
        tmp_num_dur = 5;
    }
    let mut properties = network_properties::NetworkProperties::new_dur(&network, &cur_params);

    let (sir, e_events, i_events, r_events, ts) = run_model::small_dur_g(&network, &mut properties, num_infec, tmp_num_dur);
               
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("adjacency_matrix", network.adjacency_matrix.to_object(py))?;
        dict.set_item("degrees", network.degrees.to_object(py))?;
        dict.set_item("frequency_distribution", network.frequency_distribution.to_object(py))?;
        dict.set_item("partitions", network.partitions.to_object(py))?;
        dict.set_item("ages", network.ages.to_object(py))?;
        dict.set_item("disease_from", properties.disease_from.to_object(py))?;
        dict.set_item("generation", properties.generation.to_object(py))?;
        dict.set_item("parameters", properties.parameters.to_object(py))?;
        dict.set_item("secondary_cases", properties.secondary_cases.to_object(py))?;
        dict.set_item("sir", sir.to_object(py))?;
        dict.set_item("i_events", i_events.to_object(py))?;
        dict.set_item("r_events", r_events.to_object(py))?;
        dict.set_item("e_events", e_events.to_object(py))?;
        dict.set_item("ts", ts.to_object(py))?;
    
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

#[pyfunction]
fn small_sbm_dur(contact_matrix: Vec<Vec<Vec<f64>>>, tau: f64, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize, num_dur: usize, props: Vec<f64>) -> PyResult<Py<PyDict>> {
    
    let mut tmp_num_dur = num_dur;
    let mut cur_params = outbreak_params.clone();
    cur_params[0] = tau;
    let mut network: network_structure::NetworkStructureDuration = network_structure::NetworkStructureDuration::new_sbm_dur(partitions.last().unwrap().to_owned(), &partitions, &contact_matrix, num_dur);
    if props.len() > 0 {
        network.transform(&props);
        tmp_num_dur = 5;
    }
    let mut properties = network_properties::NetworkProperties::new_dur(&network, &cur_params);

    let (sir, e_events, i_events, r_events, ts) = run_model::small_dur_g(&network, &mut properties, num_infec, tmp_num_dur);
               
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("adjacency_matrix", network.adjacency_matrix.to_object(py))?;
        dict.set_item("degrees", network.degrees.to_object(py))?;
        dict.set_item("frequency_distribution", network.frequency_distribution.to_object(py))?;
        dict.set_item("partitions", network.partitions.to_object(py))?;
        dict.set_item("ages", network.ages.to_object(py))?;
        dict.set_item("disease_from", properties.disease_from.to_object(py))?;
        dict.set_item("generation", properties.generation.to_object(py))?;
        dict.set_item("parameters", properties.parameters.to_object(py))?;
        dict.set_item("secondary_cases", properties.secondary_cases.to_object(py))?;
        dict.set_item("sir", sir.to_object(py))?;
        dict.set_item("i_events", i_events.to_object(py))?;
        dict.set_item("r_events", r_events.to_object(py))?;
        dict.set_item("e_events", e_events.to_object(py))?;
        dict.set_item("ts", ts.to_object(py))?;
    
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

#[pyfunction]
fn small_gillespie(degree_age_breakdown: Vec<Vec<usize>>, tau: f64, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize) -> PyResult<Py<PyDict>> {
    
    let mut cur_params = outbreak_params.clone();
    cur_params[0] = tau;
    let network = network_structure::NetworkStructure::new_from_degree_dist(&partitions, &degree_age_breakdown);
    let mut properties = network_properties::NetworkProperties::new(&network, &cur_params);

    let (sir, e_events, i_events, r_events, ts) = run_model::small_g(&network, &mut properties, num_infec);
               
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("adjacency_matrix", network.adjacency_matrix.to_object(py))?;
        dict.set_item("degrees", network.degrees.to_object(py))?;
        dict.set_item("frequency_distribution", network.frequency_distribution.to_object(py))?;
        dict.set_item("partitions", network.partitions.to_object(py))?;
        dict.set_item("ages", network.ages.to_object(py))?;
        dict.set_item("disease_from", properties.disease_from.to_object(py))?;
        dict.set_item("generation", properties.generation.to_object(py))?;
        dict.set_item("parameters", properties.parameters.to_object(py))?;
        dict.set_item("secondary_cases", properties.secondary_cases.to_object(py))?;
        dict.set_item("sir", sir.to_object(py))?;
        dict.set_item("i_events", i_events.to_object(py))?;
        dict.set_item("r_events", r_events.to_object(py))?;
        dict.set_item("e_events", e_events.to_object(py))?;
        dict.set_item("ts", ts.to_object(py))?;
    
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}


#[pyfunction]
fn gillesp_dur(degree_age_breakdown: Vec<Vec<usize>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize, num_dur: usize, props: Vec<f64>) -> PyResult<Py<PyDict>> {
    
    let mut I1 = vec![vec![0; iterations];taus.len()];
    let mut I2 = vec![vec![0; iterations];taus.len()];
    let mut I3 = vec![vec![0; iterations];taus.len()];
    let mut I4 = vec![vec![0; iterations];taus.len()];
    let mut fs = vec![vec![0.; iterations];taus.len()];
    let mut peak_heights = vec![vec![0; iterations];taus.len()];
    let mut peak_times = vec![vec![0.; iterations];taus.len()];
    let mut initial_infected = vec![vec![Vec::new(); iterations];taus.len()];
    let mut age_dur_breakdown = vec![vec![Vec::new(); iterations];taus.len()];
    let mut max_gen = vec![vec![0usize; iterations];taus.len()];
    let mut tmp_num_dur = num_dur;

    let mut network: network_structure::NetworkStructureDuration = network_structure::NetworkStructureDuration::new_from_dur_dist(&partitions, &degree_age_breakdown, num_dur);
    if props.len() > 0 {
        network.transform(&props);
        tmp_num_dur = 5;
    }
    let avg_d = network.degrees.iter().map(|x| (x.iter().sum::<usize>() as f64)).sum::<f64>() / (network.degrees.len() as f64);
    
    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        
        let properties = network_properties::NetworkProperties::new_dur(&network, &cur_params);

        let results: Vec<(f64, usize, usize, usize, usize, usize, f64, Vec<Vec<Vec<usize>>>,Vec<Vec<Vec<usize>>>,usize)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::dur_gillesp(&network, &mut properties.clone(), num_infec, tmp_num_dur)
                })
                .collect();
        for (k, sim) in results.iter().enumerate() {
            fs[i][k] = sim.0; I1[i][k] = sim.1; I2[i][k] = sim.2; I3[i][k] = sim.3; I4[i][k] = sim.4; peak_heights[i][k] = sim.5; peak_times[i][k] = sim.6; initial_infected[i][k] = sim.7.clone(); age_dur_breakdown[i][k] = sim.8.clone(); max_gen[i][k] = sim.9;
        }
    }
    let tmp_edge_list = network.adjacency_matrix.iter().map(|x| x.iter().map(|y| (y.0, y.1)).collect::<Vec<(usize, usize)>>()).collect::<Vec<Vec<(usize, usize)>>>();
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("fs", fs.to_object(py))?;
        dict.set_item("I1", I1.to_object(py))?;
        dict.set_item("I2", I2.to_object(py))?;
        dict.set_item("I3", I3.to_object(py))?;
        dict.set_item("I4", I4.to_object(py))?;
        dict.set_item("peak_heights", peak_heights.to_object(py))?;
        dict.set_item("peak_times", peak_times.to_object(py))?;
        dict.set_item("initial_infected", initial_infected.to_object(py))?;
        dict.set_item("max_gen", max_gen.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;
        dict.set_item("age_dur_sc", age_dur_breakdown.to_object(py))?;
        dict.set_item("avg_d_network", avg_d.to_object(py))?;
        dict.set_item("avg_d_input", degree_age_breakdown.iter().map(|x| x.iter().sum::<usize>() as f64).sum::<f64>().to_object(py))?;
        dict.set_item("largest_connected_component", network_structure::largest_cc(tmp_edge_list).to_object(py))?;
    
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

#[pyfunction]
fn gillesp_dur_sc(degree_age_breakdown: Vec<Vec<usize>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize, num_dur: usize, props: Vec<f64>) -> PyResult<Py<PyDict>> {
    
    let mut r0 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut r02 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut r03 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut age_dur_sc = vec![vec![Vec::new(); iterations];taus.len()];
    let mut tmp_num_dur = num_dur;

    let mut network: network_structure::NetworkStructureDuration = network_structure::NetworkStructureDuration::new_from_dur_dist(&partitions, &degree_age_breakdown, num_dur);
    if props.len() > 0 {
        network.transform(&props);
        tmp_num_dur = 5;
    }
    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        
        let properties = network_properties::NetworkProperties::new_dur(&network, &cur_params);

        let results: Vec<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<Vec<Vec<usize>>>)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::dur_gillesp_sc(&network, &mut properties.clone(), num_infec, tmp_num_dur)
                })
                .collect();
        for (k, sim) in results.iter().enumerate() {
            for val in sim.0.iter() {
                r0[i][k].push(*val);
            }
            for val in sim.1.iter() {
                r02[i][k].push(*val);
            }
            for val in sim.2.iter() {
                r03[i][k].push(*val);
            }
            age_dur_sc[i][k] = sim.3.clone();
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("sc", r0.to_object(py))?;
        dict.set_item("sc2", r02.to_object(py))?;
        dict.set_item("sc3", r03.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?; 
        dict.set_item("age_dur_sc", age_dur_sc.to_object(py))?;
        Ok(dict.into())
    })
}

#[pyfunction]
fn gillesp_sbmdur_sc(contact_matrix: Vec<Vec<Vec<f64>>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize, num_dur: usize, props: Vec<f64>) -> PyResult<Py<PyDict>> {
    
    let mut r0 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut r02 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut r03 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut age_dur_sc = vec![vec![Vec::new(); iterations];taus.len()];

    let mut tmp_num_dur = num_dur;

    let mut network: network_structure::NetworkStructureDuration = network_structure::NetworkStructureDuration::new_sbm_dur(partitions.last().cloned().unwrap(), &partitions, &contact_matrix, num_dur);
    if props.len() > 0 {
        network.transform(&props);
        tmp_num_dur = 5;
    }
    for (i, &tau) in taus.iter().enumerate() {
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        
        let properties = network_properties::NetworkProperties::new_dur(&network, &cur_params);

        let results: Vec<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<Vec<Vec<usize>>>)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::dur_gillesp_sc(&network, &mut properties.clone(), num_infec, tmp_num_dur)
                })
                .collect();
        for (k, sim) in results.iter().enumerate() {
            for val in sim.0.iter() {
                r0[i][k].push(*val);
            }
            for val in sim.1.iter() {
                r02[i][k].push(*val);
            }
            for val in sim.2.iter() {
                r03[i][k].push(*val);
            }
            age_dur_sc[i][k] = sim.3.clone();
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("sc", r0.to_object(py))?;
        dict.set_item("sc2", r02.to_object(py))?;
        dict.set_item("sc3", r03.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;  
        dict.set_item("age_dur_sc", age_dur_sc.to_object(py))?;  
        Ok(dict.into())
    })
}


#[pyfunction]
fn gillesp_dur_gr(degree_age_breakdown: Vec<Vec<usize>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize, num_dur: usize, props: Vec<f64>) -> PyResult<Py<PyDict>> {
    
    let mut r02 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut gr = vec![vec![0.; iterations]; taus.len()];
    let mut tmp_num_dur = num_dur;

    let mut network: network_structure::NetworkStructureDuration = network_structure::NetworkStructureDuration::new_from_dur_dist(&partitions, &degree_age_breakdown, num_dur);
    if props.len() > 0 {
        network.transform(&props);
        tmp_num_dur = 5;
    }
    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        
        let properties = network_properties::NetworkProperties::new_dur(&network, &cur_params);

        let results: Vec<(Vec<usize>, f64)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::dur_gillesp_gr(&network, &mut properties.clone(), num_infec, tmp_num_dur)
                })
                .collect();
        for (k, sim) in results.iter().enumerate() {
            for val in sim.0.iter() {
                r02[i][k].push(*val);
            }
            gr[i][k] = sim.1;
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("sc2", r02.to_object(py))?;
        dict.set_item("growth_rate", gr.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;    
        Ok(dict.into())
    })
}

#[pyfunction]
fn gillespie_gmm(degree_age_breakdown: Vec<Vec<usize>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize) -> PyResult<Py<PyDict>> {
    
    let mut I1 = vec![vec![0; iterations];taus.len()];
    let mut I2 = vec![vec![0; iterations];taus.len()];
    let mut I3 = vec![vec![0; iterations];taus.len()];
    let mut I4 = vec![vec![0; iterations];taus.len()];
    let mut fs = vec![vec![0.; iterations];taus.len()];
    let mut peak_heights = vec![vec![0; iterations];taus.len()];
    let mut peak_times = vec![vec![0.; iterations];taus.len()];
    let mut initial_infected = vec![vec![Vec::new(); iterations];taus.len()];
    let mut age_dur_breakdown = vec![vec![Vec::new(); iterations];taus.len()];
    let mut max_gen = vec![vec![0usize; iterations];taus.len()];

    let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_from_degree_dist(&partitions, &degree_age_breakdown);
    let avg_d = (network.degrees.iter().sum::<usize>() as f64) / (network.degrees.len() as f64);
    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        let properties = network_properties::NetworkProperties::new(&network, &cur_params);

        let results: Vec<(f64, usize, usize, usize, usize, usize, f64, Vec<Vec<usize>>, Vec<Vec<usize>>, usize)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::gillesp(&network, &mut properties.clone(), num_infec)
                })
                .collect();
        for (k, sim) in results.iter().enumerate() {
            fs[i][k] = sim.0; I1[i][k] = sim.1; I2[i][k] = sim.2; I3[i][k] = sim.3; I4[i][k] = sim.4; peak_heights[i][k] = sim.5; peak_times[i][k] = sim.6; initial_infected[i][k] = sim.7.clone(); age_dur_breakdown[i][k] = sim.8.clone(); max_gen[i][k] = sim.9;
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
         
        dict.set_item("fs", fs.to_object(py))?;
        dict.set_item("I1", I1.to_object(py))?;
        dict.set_item("I2", I2.to_object(py))?;
        dict.set_item("I3", I3.to_object(py))?;
        dict.set_item("I4", I4.to_object(py))?;
        dict.set_item("peak_heights", peak_heights.to_object(py))?;
        dict.set_item("peak_times", peak_times.to_object(py))?;
        dict.set_item("initial_infected", initial_infected.to_object(py))?;
        dict.set_item("max_gen", max_gen.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;
        dict.set_item("age_dur_sc", age_dur_breakdown.to_object(py))?;
        dict.set_item("avg_d_network", avg_d.to_object(py))?;
        dict.set_item("avg_d_input", degree_age_breakdown.iter().map(|x| x.iter().sum::<usize>() as f64).sum::<f64>().to_object(py))?;
        dict.set_item("largest_connected_component", network_structure::largest_cc(network.adjacency_matrix.clone()).to_object(py))?;
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

#[pyfunction]
fn gillesp_dur_sbm(contact_matrix: Vec<Vec<Vec<f64>>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize, num_dur: usize, props: Vec<f64>) -> PyResult<Py<PyDict>> {
    
    let mut I1 = vec![vec![0; iterations];taus.len()];
    let mut I2 = vec![vec![0; iterations];taus.len()];
    let mut I3 = vec![vec![0; iterations];taus.len()];
    let mut I4 = vec![vec![0; iterations];taus.len()];
    let mut fs = vec![vec![0.; iterations];taus.len()];
    let mut peak_heights = vec![vec![0; iterations];taus.len()];
    let mut peak_times = vec![vec![0.; iterations];taus.len()];
    let mut initial_infected = vec![vec![Vec::new(); iterations];taus.len()];
    let mut age_dur_breakdown = vec![vec![Vec::new(); iterations];taus.len()];
    let mut max_gen = vec![vec![0usize; iterations];taus.len()];
    let mut tmp_num_dur = num_dur;

    let mut network: network_structure::NetworkStructureDuration = network_structure::NetworkStructureDuration::new_sbm_dur(partitions.last().cloned().unwrap(), &partitions, &contact_matrix, num_dur);
    if props.len() > 0 {
        network.transform(&props);
        tmp_num_dur = 5;
    }
    let avg_d = network.degrees.iter().map(|x| (x.iter().sum::<usize>() as f64)).sum::<f64>() / (network.degrees.len() as f64);
    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        
        let properties = network_properties::NetworkProperties::new_dur(&network, &cur_params);

        let results: Vec<(f64, usize, usize, usize, usize, usize, f64, Vec<Vec<Vec<usize>>>,Vec<Vec<Vec<usize>>>,usize)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::dur_gillesp(&network, &mut properties.clone(), num_infec, tmp_num_dur)
                })
                .collect();
        for (k, sim) in results.iter().enumerate() {
            fs[i][k] = sim.0; I1[i][k] = sim.1; I2[i][k] = sim.2; I3[i][k] = sim.3; I4[i][k] = sim.4; peak_heights[i][k] = sim.5; peak_times[i][k] = sim.6; initial_infected[i][k] = sim.7.clone(); age_dur_breakdown[i][k] = sim.8.clone(); max_gen[i][k] = sim.9;
        }
    }
    let tmp_edge_list = network.adjacency_matrix.iter().map(|x| x.iter().map(|y| (y.0, y.1)).collect::<Vec<(usize, usize)>>()).collect::<Vec<Vec<(usize, usize)>>>();
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("fs", fs.to_object(py))?;
        dict.set_item("I1", I1.to_object(py))?;
        dict.set_item("I2", I2.to_object(py))?;
        dict.set_item("I3", I3.to_object(py))?;
        dict.set_item("I4", I4.to_object(py))?;
        dict.set_item("peak_heights", peak_heights.to_object(py))?;
        dict.set_item("peak_times", peak_times.to_object(py))?;
        dict.set_item("initial_infected", initial_infected.to_object(py))?;
        dict.set_item("max_gen", max_gen.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;
        dict.set_item("age_dur_sc", age_dur_breakdown.to_object(py))?;
        dict.set_item("avg_d_network", avg_d.to_object(py))?;
        dict.set_item("largest_connected_component", network_structure::largest_cc(tmp_edge_list).to_object(py))?;
    
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

#[pyfunction]
fn gillespie_sbm(contact_matrix: Vec<Vec<f64>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize) -> PyResult<Py<PyDict>> {
    
    let mut I1 = vec![vec![0; iterations];taus.len()];
    let mut I2 = vec![vec![0; iterations];taus.len()];
    let mut I3 = vec![vec![0; iterations];taus.len()];
    let mut I4 = vec![vec![0; iterations];taus.len()];
    let mut fs = vec![vec![0.; iterations];taus.len()];
    let mut peak_heights = vec![vec![0; iterations];taus.len()];
    let mut peak_times = vec![vec![0.; iterations];taus.len()];
    let mut initial_infected = vec![vec![Vec::new(); iterations];taus.len()];
    let mut age_dur_breakdown = vec![vec![Vec::new(); iterations];taus.len()];
    let mut max_gen = vec![vec![0usize; iterations];taus.len()];

    let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_sbm_from_vars(partitions.last().unwrap().to_owned(), &partitions, &contact_matrix);
    let avg_d = (network.degrees.iter().sum::<usize>() as f64) / (network.degrees.len() as f64);
    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        let properties = network_properties::NetworkProperties::new(&network, &cur_params);

        let results: Vec<(f64,usize,usize,usize,usize,usize, f64, Vec<Vec<usize>>, Vec<Vec<usize>>, usize)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::gillesp(&network, &mut properties.clone(), num_infec)
                })
                .collect();
        for (k, sim) in results.iter().enumerate() {
            fs[i][k] = sim.0; I1[i][k] = sim.1; I2[i][k] = sim.2; I3[i][k] = sim.3; I4[i][k] = sim.4; peak_heights[i][k] = sim.5; peak_times[i][k] = sim.6; initial_infected[i][k] = sim.7.clone(); age_dur_breakdown[i][k] = sim.8.clone(); max_gen[i][k] = sim.9;
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("fs", fs.to_object(py))?;
        dict.set_item("I1", I1.to_object(py))?;
        dict.set_item("I2", I2.to_object(py))?;
        dict.set_item("I3", I3.to_object(py))?;
        dict.set_item("I4", I4.to_object(py))?;
        dict.set_item("peak_heights", peak_heights.to_object(py))?;
        dict.set_item("peak_times", peak_times.to_object(py))?;
        dict.set_item("initial_infected", initial_infected.to_object(py))?;
        dict.set_item("max_gen", max_gen.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;
        dict.set_item("age_dur_sc", age_dur_breakdown.to_object(py))?;
        dict.set_item("avg_d_network", avg_d.to_object(py))?;
        dict.set_item("max_gen", max_gen.to_object(py))?;
        dict.set_item("largest_connected_component", network_structure::largest_cc(network.adjacency_matrix.clone()).to_object(py))?;
    
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

#[pyfunction]
fn gillesp_gmm_sc(degree_age_breakdown: Vec<Vec<usize>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize) -> PyResult<Py<PyDict>> {
    
    let mut r0 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut r02 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut r03 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut age_dur_sc = vec![vec![Vec::new(); iterations];taus.len()];

    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_from_degree_dist(&partitions, &degree_age_breakdown);

        let properties = network_properties::NetworkProperties::new(&network, &cur_params);

        let results: Vec<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<Vec<Vec<usize>>>)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::gillesp_sc(&network, &mut properties.clone(), num_infec)
                })
                .collect();
        for (k, sim) in results.iter().enumerate() {
            for val in sim.0.iter() {
                r0[i][k].push(*val);
            }
            for val in sim.1.iter() {
                r02[i][k].push(*val);
            }
            for val in sim.2.iter() {
                r03[i][k].push(*val);
            }
            age_dur_sc[i][k] = sim.3.clone();
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("sc", r0.to_object(py))?;
        dict.set_item("sc2", r02.to_object(py))?;
        dict.set_item("sc3", r03.to_object(py))?;
        dict.set_item("age_dur_sc", age_dur_sc.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;    
        Ok(dict.into())
    })
}

#[pyfunction]
fn gillesp_sbm_sc(contact_matrix: Vec<Vec<f64>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize) -> PyResult<Py<PyDict>> {
    
    let mut r0 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut r02 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut r03 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut age_dur_sc = vec![vec![Vec::new(); iterations];taus.len()];

    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_sbm_from_vars(partitions.last().unwrap().to_owned(), &partitions, &contact_matrix);

        let properties = network_properties::NetworkProperties::new(&network, &cur_params);

        let results: Vec<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<Vec<Vec<usize>>>)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::gillesp_sc(&network, &mut properties.clone(), num_infec)
                })
                .collect();
        for (k, sim) in results.iter().enumerate() {
            for val in sim.0.iter() {
                r0[i][k].push(*val);
            }
            for val in sim.1.iter() {
                r02[i][k].push(*val);
            }
            for val in sim.2.iter() {
                r03[i][k].push(*val);
            }
            age_dur_sc[i][k] = sim.3.clone();
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("sc", r0.to_object(py))?;
        dict.set_item("sc2", r02.to_object(py))?;
        dict.set_item("sc3", r03.to_object(py))?;
        dict.set_item("age_dur_sc", age_dur_sc.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;    
        Ok(dict.into())
    })
}


#[pyfunction]
fn gillesp_sbm_gr(contact_matrix: Vec<Vec<f64>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, num_infec: usize) -> PyResult<Py<PyDict>> {
    
    let mut r02 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut gr = vec![vec![0.; iterations]; taus.len()];

    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_sbm_from_vars(partitions.last().unwrap().to_owned(), &partitions, &contact_matrix);

        let properties = network_properties::NetworkProperties::new(&network, &cur_params);

        let results: Vec<(Vec<usize>, f64)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::gillesp_gr(&network, &mut properties.clone(), num_infec)
                })
                .collect();
        for (k, sim) in results.iter().enumerate() {
            for val in sim.0.iter() {
                r02[i][k].push(*val);
            }
            gr[i][k] = sim.1;
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("sc2", r02.to_object(py))?;
        dict.set_item("gr", gr.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;    
        Ok(dict.into())
    })
}



#[pyfunction]
fn sellke_dur(degree_age_breakdown: Vec<Vec<usize>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, prop_infec: f64, num_dur: usize, props: Vec<f64>) -> PyResult<Py<PyDict>> {
    
    let mut r0 = vec![vec![0.; iterations];taus.len()]; 
    let mut r02 = vec![vec![0.; iterations];taus.len()]; 
    let mut r03 = vec![vec![0.; iterations];taus.len()]; 
    let mut r04 = vec![vec![0.; iterations];taus.len()]; 
    let mut r05 = vec![vec![0.; iterations];taus.len()]; 
    let mut r06 = vec![vec![0.; iterations];taus.len()]; 
    let mut r07 = vec![vec![0.; iterations];taus.len()];
    let mut r08 = vec![vec![0.; iterations];taus.len()];
    let mut fs = vec![vec![0.; iterations];taus.len()]; 
    let mut avg_d = vec![0.;taus.len()]; 
    let mut age_dur_breakdown = vec![vec![Vec::new(); iterations];taus.len()];
    let mut tmp_num_dur = num_dur;

    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        let mut network: network_structure::NetworkStructureDuration = network_structure::NetworkStructureDuration::new_from_dur_dist(&partitions, &degree_age_breakdown, num_dur);
        if props.len() > 0 {
            network.transform(&props);
            tmp_num_dur = 5;
        }
        let properties = network_properties::NetworkProperties::new_dur(&network, &cur_params);

        avg_d[i] = network.degrees.iter().map(|x| (x.iter().sum::<usize>() as f64)).sum::<f64>() / (network.degrees.len() as f64);

        let results: Vec<(f64,f64,f64,f64,f64,f64,f64,f64,f64,Vec<Vec<Vec<usize>>>,f64)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::dur_sellke(&network, &mut properties.clone(), prop_infec, tmp_num_dur)
                })
                .collect();
        for (k, sim) in results.iter().enumerate() {
            fs[i][k] = sim.0; r0[i][k] = sim.1; r02[i][k] = sim.2; r03[i][k] = sim.3; r04[i][k] = sim.4; r05[i][k] = sim.5; r06[i][k] = sim.6; r07[i][k] = sim.7; r08[i][k] = sim.8; age_dur_breakdown[i][k] = sim.9.clone();
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("fs", fs.to_object(py))?;
        dict.set_item("r0", r0.to_object(py))?;
        dict.set_item("r02", r02.to_object(py))?;
        dict.set_item("r03", r03.to_object(py))?;
        dict.set_item("r04", r04.to_object(py))?;
        dict.set_item("r05", r05.to_object(py))?;
        dict.set_item("r06", r06.to_object(py))?;
        dict.set_item("r07", r07.to_object(py))?;
        dict.set_item("r08", r08.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;
        dict.set_item("age_dur_sc", age_dur_breakdown.to_object(py))?;
        dict.set_item("avg_d_network", avg_d.to_object(py))?;
        dict.set_item("avg_d_input", degree_age_breakdown.iter().map(|x| x.iter().sum::<usize>() as f64).sum::<f64>().to_object(py))?;
    
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

#[pyfunction]
fn dur_r0(degree_age_breakdown: Vec<Vec<usize>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, prop_infec: f64, num_dur: usize, props: Vec<f64>) -> PyResult<Py<PyDict>> {
    
    let mut r0 = vec![vec![Vec::new(); iterations];taus.len()]; 
    let mut avg_d = vec![0.;taus.len()];
    let mut tmp_num_dur = num_dur; 

    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        let mut network: network_structure::NetworkStructureDuration = network_structure::NetworkStructureDuration::new_from_dur_dist(&partitions, &degree_age_breakdown, num_dur);
        if props.len() > 0 {
            network.transform(&props);
            tmp_num_dur = 5;
        }

        let properties = network_properties::NetworkProperties::new_dur(&network, &cur_params);

        avg_d[i] = network.degrees.iter().map(|x| (x.iter().sum::<usize>() as f64)).sum::<f64>() / (network.degrees.len() as f64);

        let results: Vec<(Vec<i64>,f64)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::dur_r0(&network, &mut properties.clone(), prop_infec, tmp_num_dur, props.clone())
                })
                .collect();
        for k in 0..results.len() {
            let sim = &results[k];
            r0[i][k] = sim.0.to_owned();
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        dict.set_item("r0s", r0.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;
        dict.set_item("avg_d_network", avg_d.to_object(py))?;
        dict.set_item("avg_d_input", degree_age_breakdown.iter().map(|x| x.iter().sum::<usize>() as f64).sum::<f64>().to_object(py))?;
    
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}



#[pyfunction]
fn get_r0(degree_age_breakdown: Vec<Vec<usize>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, prop_infec: f64, scaling: &str) -> PyResult<Py<PyDict>> {

    let mut r0 = vec![vec![0.; iterations];taus.len()]; 
    // let (mut ts, mut sirs) = (Vec::new(), Vec::new());
    // parallel simulations

    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_from_degree_dist(&partitions, &degree_age_breakdown);
        let properties = network_properties::NetworkProperties::new(&network, &cur_params);

        let results: Vec<f64>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::r0_sellke(&network, &mut properties.clone(), prop_infec, scaling)
                })
                .collect();
        for (k, &sim) in results.iter().enumerate() {
            r0[i][k] = sim;
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("r0", r0.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;
        // dict.set_item("t", ts.to_object(py))?;
        // dict.set_item("sir", sirs.to_object(py))?;
        
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

#[pyfunction]
fn get_fs(degree_age_breakdown: Vec<Vec<usize>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, prop_infec: f64, scaling: &str) -> PyResult<Py<PyDict>> {

    let mut fs = vec![vec![0.; iterations];taus.len()]; 
    // let (mut ts, mut sirs) = (Vec::new(), Vec::new());
    // parallel simulations

    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_from_degree_dist(&partitions, &degree_age_breakdown);
        let properties = network_properties::NetworkProperties::new(&network, &cur_params);

        let results: Vec<f64>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    run_model::fs_sellke(&network, &mut properties.clone(), prop_infec, scaling)
                })
                .collect();
        for (k, &sim) in results.iter().enumerate() {
            fs[i][k] = sim;
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("final_size", fs.to_object(py))?;
        dict.set_item("taus", taus.to_object(py))?;
        // dict.set_item("t", ts.to_object(py))?;
        // dict.set_item("sir", sirs.to_object(py))?;
        
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}


#[pyfunction]
fn big_sellke_sec_cases(taus: Vec<f64>, networks: usize, iterations: usize, n: usize, partitions: Vec<usize>, dist_type: &str, network_params: Vec<Vec<f64>>, contact_matrix: Vec<Vec<f64>>, outbreak_params: Vec<f64>, prop_infec: f64, scaling: &str) -> PyResult<Py<PyDict>> {

    let (mut r01, mut sc1, mut sc2, mut sc3, mut sc4, mut sc5, mut sc6, mut sc7, mut sc8, mut sc9, mut sc10) = (vec![vec![0.; networks*iterations]; taus.len()], vec![Vec::new(); taus.len()], vec![Vec::new(); taus.len()], 
        vec![Vec::new(); taus.len()], vec![Vec::new(); taus.len()], vec![Vec::new(); taus.len()], vec![Vec::new(); taus.len()], 
        vec![Vec::new(); taus.len()], vec![Vec::new(); taus.len()], vec![Vec::new(); taus.len()], vec![Vec::new(); taus.len()]); 
    // let (mut ts, mut sirs) = (Vec::new(), Vec::new());
    // parallel simulations

    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        for j in 0..networks {
            let network: network_structure::NetworkStructure = match dist_type { 
                "sbm" => {
                    network_structure::NetworkStructure::new_sbm_from_vars(n, &partitions, &contact_matrix)
                },
                _ => network_structure::NetworkStructure::new_mult_from_input(n, &partitions, dist_type, &network_params, &contact_matrix)
            };
            let properties = network_properties::NetworkProperties::new(&network, &cur_params);

            let results: Vec<(f64, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>)>
                = (0..iterations)
                    .into_par_iter()
                    .map(|_| {
                        let (t,_,_,sir,sec_cases,geners, _) = run_model::run_sellke(&network, &mut properties.clone(), prop_infec, scaling);
                        if geners.iter().max().unwrap().to_owned() <= 3 {
                            (-1.,Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new())
                        }
                        else {
                            let gen1 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 1).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen2 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 2).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen3 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 3).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen4 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 4).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen5 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 5).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen6 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 6).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen7 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 7).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen8 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 8).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen9 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 9).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen10 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 10).map(|(_,&x)| x).collect::<Vec<usize>>();
                            
                            // let gen23 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 2 || geners[i.to_owned()] == 3).map(|(_,&x)| x).collect::<Vec<usize>>();
                            ((gen1.iter().sum::<usize>() as f64) / (gen1.len() as f64), gen1, gen2, gen3, gen4, gen5, gen6, gen7, gen8, gen9, gen10)
                        }
                    })
                    .collect();
            for (k, sim) in results.iter().enumerate() {
                r01[i][j*iterations + k] = sim.0; 
                for val in sim.1.iter() {sc1[i].push(val.to_owned());}
                for val in sim.2.iter() {sc2[i].push(val.to_owned());}
                for val in sim.3.iter() {sc3[i].push(val.to_owned());}
                for val in sim.4.iter() {sc4[i].push(val.to_owned());}
                for val in sim.5.iter() {sc5[i].push(val.to_owned());}
                for val in sim.6.iter() {sc6[i].push(val.to_owned());}
                for val in sim.7.iter() {sc7[i].push(val.to_owned());}
                for val in sim.8.iter() {sc8[i].push(val.to_owned());}
                for val in sim.9.iter() {sc9[i].push(val.to_owned());}
                for val in sim.10.iter() {sc10[i].push(val.to_owned());}
            }
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("r0_1", r01.to_object(py))?;
        dict.set_item("secondary_cases1", sc1.to_object(py))?;
        dict.set_item("secondary_cases2", sc2.to_object(py))?;
        dict.set_item("secondary_cases3", sc3.to_object(py))?;
        dict.set_item("secondary_cases4", sc4.to_object(py))?;
        dict.set_item("secondary_cases5", sc5.to_object(py))?;
        dict.set_item("secondary_cases6", sc6.to_object(py))?;
        dict.set_item("secondary_cases7", sc7.to_object(py))?;
        dict.set_item("secondary_cases8", sc8.to_object(py))?;
        dict.set_item("secondary_cases9", sc9.to_object(py))?;
        dict.set_item("secondary_cases10", sc10.to_object(py))?;

        
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}


#[pyfunction]
fn gmm_sims(degree_age_breakdown: Vec<Vec<usize>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, prop_infec: f64, scaling: &str) -> PyResult<Py<PyDict>> {

    let (mut r01, mut r023, mut final_size, mut peak_height) = (vec![vec![0.; iterations]; taus.len()], vec![vec![0.; iterations]; taus.len()], vec![vec![0; iterations]; taus.len()], vec![vec![0; iterations]; taus.len()]); 
    // let (mut ts, mut sirs) = (Vec::new(), Vec::new());
    // parallel simulations

    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_from_degree_dist(&partitions, &degree_age_breakdown);
        let properties = network_properties::NetworkProperties::new(&network, &cur_params);

        let results: Vec<(f64, f64, i64, i64, Vec<f64>, Vec<Vec<usize>>)>
            = (0..iterations)
                .into_par_iter()
                .map(|_| {
                    let (t,_,_,sir,sec_cases,geners, _) = run_model::run_sellke(&network, &mut properties.clone(), prop_infec, scaling);
                    if geners.iter().max().unwrap().to_owned() < 3 {
                        (-1.,-1.,-1,-1, t,sir)
                    }
                    else {
                        let gen1 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 1).map(|(_,&x)| x).collect::<Vec<usize>>();
                        let gen23 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 2 || geners[i.to_owned()] == 3).map(|(_,&x)| x).collect::<Vec<usize>>();
                        ((gen1.iter().sum::<usize>() as f64) / (gen1.len() as f64),(gen23.iter().sum::<usize>() as f64) / (gen23.len() as f64), sir.last().unwrap()[2] as i64, sir.iter().filter_map(|x| x.get(1)).max().unwrap().to_owned() as i64, t, sir)
                    }
                })
                .collect();
        for (k, sim) in results.iter().enumerate() {
            r01[i][k] = sim.0; r023[i][k] = sim.1; final_size[i][k] = sim.2; peak_height[i][k] = sim.3;
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("taus", taus.to_object(py))?;
        dict.set_item("r0_1", r01.to_object(py))?;
        dict.set_item("r0_23", r023.to_object(py))?;
        dict.set_item("final_size", final_size.to_object(py))?;
        dict.set_item("peak_height", peak_height.to_object(py))?;
        // dict.set_item("t", ts.to_object(py))?;
        // dict.set_item("sir", sirs.to_object(py))?;
        
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

#[pyfunction]
fn gmm_sims_sc(degree_age_breakdown: Vec<Vec<usize>>, taus: Vec<f64>, iterations: usize, partitions: Vec<usize>, outbreak_params: Vec<f64>, prop_infec: f64, scaling: &str) -> PyResult<Py<PyDict>> {

    let (mut r01, mut sc1, mut sc2, mut sc3) = (vec![vec![0.; iterations]; taus.len()], vec![Vec::new(); taus.len()], vec![Vec::new(); taus.len()], vec![Vec::new(); taus.len()]); 
    // let (mut ts, mut sirs) = (Vec::new(), Vec::new());
    // parallel simulations

    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        let network: network_structure::NetworkStructure = network_structure::NetworkStructure::new_from_degree_dist(&partitions, &degree_age_breakdown);
        let properties = network_properties::NetworkProperties::new(&network, &cur_params);

        let results: Vec<(f64, Vec<usize>, Vec<usize>, Vec<usize>)>
                = (0..iterations)
                    .into_par_iter()
                    .map(|_| {
                        let (t,_,_,sir,sec_cases,geners, _) = run_model::run_sellke(&network, &mut properties.clone(), prop_infec, scaling);
                        if geners.iter().max().unwrap().to_owned() <= 3 {
                            (-1.,Vec::new(),Vec::new(),Vec::new())
                        }
                        else {
                            let gen1 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 1).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen2 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 2).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen3 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 3).map(|(_,&x)| x).collect::<Vec<usize>>();
                            // let gen23 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 2 || geners[i.to_owned()] == 3).map(|(_,&x)| x).collect::<Vec<usize>>();
                            ((gen1.iter().sum::<usize>() as f64) / (gen1.len() as f64), gen1, gen2, gen3)
                        }
                    })
                    .collect();
            for (k, sim) in results.iter().enumerate() {
                r01[i][k] = sim.0; 
                for val in sim.1.iter() {sc1[i].push(val.to_owned());}
                for val in sim.2.iter() {sc2[i].push(val.to_owned());}
                for val in sim.3.iter() {sc3[i].push(val.to_owned());}
            }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("r0_1", r01.to_object(py))?;
        dict.set_item("secondary_cases", sc1.to_object(py))?;
        // dict.set_item("t", ts.to_object(py))?;
        // dict.set_item("sir", sirs.to_object(py))?;
        
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}



#[pyfunction]
fn big_sellke(taus: Vec<f64>, networks: usize, iterations: usize, n: usize, partitions: Vec<usize>, dist_type: &str, network_params: Vec<Vec<f64>>, contact_matrix: Vec<Vec<f64>>, outbreak_params: Vec<f64>, prop_infec: f64, scaling: &str) -> PyResult<Py<PyDict>> {

    let (mut r01, mut r023, mut final_size, mut peak_height) = (vec![vec![0.; networks*iterations]; taus.len()], vec![vec![0.; networks*iterations]; taus.len()], vec![vec![0; networks*iterations]; taus.len()], vec![vec![0; networks*iterations]; taus.len()]); 
    // let (mut ts, mut sirs) = (Vec::new(), Vec::new());
    // parallel simulations

    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        for j in 0..networks {
            let network: network_structure::NetworkStructure = match dist_type { 
                "sbm" => {
                    network_structure::NetworkStructure::new_sbm_from_vars(n, &partitions, &contact_matrix)
                },
                _ => network_structure::NetworkStructure::new_mult_from_input(n, &partitions, dist_type, &network_params, &contact_matrix)
            };
            let properties = network_properties::NetworkProperties::new(&network, &cur_params);

            let results: Vec<(f64, f64, i64, i64, Vec<f64>, Vec<Vec<usize>>)>
                = (0..iterations)
                    .into_par_iter()
                    .map(|_| {
                        let (t,_,_,sir,sec_cases,geners, _) = run_model::run_sellke(&network, &mut properties.clone(), prop_infec, scaling);
                        if geners.iter().max().unwrap().to_owned() < 3 {
                            (-1.,-1.,-1,-1, t,sir)
                        }
                        else {
                            let gen1 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 1).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen23 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 2 || geners[i.to_owned()] == 3).map(|(_,&x)| x).collect::<Vec<usize>>();
                            ((gen1.iter().sum::<usize>() as f64) / (gen1.len() as f64),(gen23.iter().sum::<usize>() as f64) / (gen23.len() as f64), sir.last().unwrap()[2] as i64, sir.iter().filter_map(|x| x.get(1)).max().unwrap().to_owned() as i64, t, sir)
                            // let gen23 = geners.iter().filter(|&&x| x == 2 || x == 3).collect::<Vec<&usize>>().len();
                            // let gen34 = geners.iter().filter(|&&x| x == 3 || x == 4).collect::<Vec<&usize>>().len();
                            // ((gen34 as f64)/(gen23 as f64), sir.last().unwrap()[2] as i64, sir.iter().filter_map(|x| x.get(1)).max().unwrap().to_owned() as i64)
                        }
                    })
                    .collect();
            for (k, sim) in results.iter().enumerate() {
                r01[i][j*iterations + k] = sim.0; r023[i][j*iterations + k] = sim.1; final_size[i][j*iterations + k] = sim.2; peak_height[i][j*iterations + k] = sim.3;
                // ts.push(sim.4.clone()); sirs.push(sim.5.iter().map(|sir| sir[1]).collect::<Vec<usize>>());
            }
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("r0_1", r01.to_object(py))?;
        dict.set_item("r0_23", r023.to_object(py))?;
        dict.set_item("final_size", final_size.to_object(py))?;
        dict.set_item("peak_height", peak_height.to_object(py))?;
        // dict.set_item("t", ts.to_object(py))?;
        // dict.set_item("sir", sirs.to_object(py))?;
        
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

#[pyfunction]
fn big_sellke_growth_rate(taus: Vec<f64>, networks: usize, iterations: usize, n: usize, partitions: Vec<usize>, dist_type: &str, network_params: Vec<Vec<f64>>, contact_matrix: Vec<Vec<f64>>, outbreak_params: Vec<f64>, prop_infec: f64, scaling: &str) -> PyResult<Py<PyDict>> {

    let (mut r01, mut r023, mut final_size, mut peak_height) = (vec![vec![0.; networks*iterations]; taus.len()], vec![vec![0.; networks*iterations]; taus.len()], vec![vec![0; networks*iterations]; taus.len()], vec![vec![0; networks*iterations]; taus.len()]); 
    let (mut ts, mut sirs) = (Vec::new(), Vec::new());
    // parallel simulations

    for (i, &tau) in taus.iter().enumerate() {
        println!("{i}");
        let mut cur_params = outbreak_params.clone();
        cur_params[0] = tau;
        for j in 0..networks {
            let network: network_structure::NetworkStructure = match dist_type { 
                "sbm" => {
                    network_structure::NetworkStructure::new_sbm_from_vars(n, &partitions, &contact_matrix)
                },
                _ => network_structure::NetworkStructure::new_mult_from_input(n, &partitions, dist_type, &network_params, &contact_matrix)
            };
            let properties = network_properties::NetworkProperties::new(&network, &cur_params);

            let results: Vec<(f64, f64, i64, i64, Vec<f64>, Vec<Vec<usize>>)>
                = (0..iterations)
                    .into_par_iter()
                    .map(|_| {
                        let (t,_,_,sir,sec_cases,geners, _) = run_model::run_sellke(&network, &mut properties.clone(), prop_infec, scaling);
                        if geners.iter().max().unwrap().to_owned() < 3 {
                            (-1.,-1.,-1,-1, t,sir)
                        }
                        else {
                            let gen1 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 1).map(|(_,&x)| x).collect::<Vec<usize>>();
                            let gen23 = sec_cases.iter().enumerate().filter(|(i,_)| geners[i.to_owned()] == 2 || geners[i.to_owned()] == 3).map(|(_,&x)| x).collect::<Vec<usize>>();
                            ((gen1.iter().sum::<usize>() as f64) / (gen1.len() as f64),(gen23.iter().sum::<usize>() as f64) / (gen23.len() as f64), sir.last().unwrap()[2] as i64, sir.iter().filter_map(|x| x.get(1)).max().unwrap().to_owned() as i64, t, sir)
                            // let gen23 = geners.iter().filter(|&&x| x == 2 || x == 3).collect::<Vec<&usize>>().len();
                            // let gen34 = geners.iter().filter(|&&x| x == 3 || x == 4).collect::<Vec<&usize>>().len();
                            // ((gen34 as f64)/(gen23 as f64), sir.last().unwrap()[2] as i64, sir.iter().filter_map(|x| x.get(1)).max().unwrap().to_owned() as i64)
                        }
                    })
                    .collect();
            for (k, sim) in results.iter().enumerate() {
                r01[i][j*iterations + k] = sim.0; r023[i][j*iterations + k] = sim.1; final_size[i][j*iterations + k] = sim.2; peak_height[i][j*iterations + k] = sim.3;
                ts.push(sim.4.clone()); sirs.push(sim.5.iter().enumerate().filter(|(index, _)| sim.4[index.to_owned()] < 14.).map(|(_, sir)| sir[1]).collect::<Vec<usize>>());
            }
        }
    }
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("r0_1", r01.to_object(py))?;
        dict.set_item("r0_23", r023.to_object(py))?;
        dict.set_item("final_size", final_size.to_object(py))?;
        dict.set_item("peak_height", peak_height.to_object(py))?;
        dict.set_item("t", ts.to_object(py))?;
        dict.set_item("sir", sirs.to_object(py))?;
        
        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

#[pyfunction]
fn small_sellke(n: usize, adjacency_matrix: Vec<Vec<(usize,usize)>>, ages: Vec<usize>, outbreak_params: Vec<f64>, prop_infec: f64,scaling: &str) -> PyResult<Py<PyDict>> {

    let mut partitions = vec![0; ages.iter().max().unwrap().to_owned()+1];
    for &age in ages.iter() {
        partitions[age] += 1;
    }
    partitions.iter_mut().fold(0usize, |acc, x| {
        *x + acc
    });
    let network = NetworkStructure{
        adjacency_matrix: adjacency_matrix.clone(),
        degrees: adjacency_matrix.iter().map(|x| x.len()).collect(),
        ages: ages,
        frequency_distribution: Vec::new(),
        partitions: partitions,
    };
    let mut properties = network_properties::NetworkProperties::new(&network, &outbreak_params);
    let (t, I_events, R_events, sir, secondary_cases, generations, infected_by) = run_model::run_sellke(&network, &mut properties.clone(), prop_infec, scaling);

    // Initialize the Python interpreter
    Python::with_gil(|py| {
        // Create output PyDict
        let dict = PyDict::new_bound(py);
        
        dict.set_item("t", t.to_object(py))?;
        dict.set_item("I_events", I_events.to_object(py))?;
        dict.set_item("R_events", R_events.to_object(py))?;
        dict.set_item("SIR", sir.to_object(py))?;
        dict.set_item("secondary_cases", secondary_cases.to_object(py))?;
        dict.set_item("generations", generations.to_object(py))?;
        dict.set_item("infected_by", infected_by.to_object(py))?;
        

        // Convert dict to PyObject and return
        Ok(dict.into())
    })
}

//////////////////////////////// Double Pareto Log-Normal functions /////////////////////////////////////////

#[pyfunction]
pub fn fit_dpln(data: Vec<f64>, iters: usize, prior_params: Vec<f64>) -> PyResult<Py<PyDict>> {
    
    // Initialize the Python interpreter
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        // Attempt to run the optimization
        match dpln::fit_dpln(data, iters, prior_params) {
            Ok(network_params) => {
                dict.set_item("alpha", network_params.alpha.to_object(py))?;
                dict.set_item("beta", network_params.beta.to_object(py))?;
                dict.set_item("nu", network_params.nu.to_object(py))?;
                dict.set_item("tau", network_params.tau.to_object(py))?;
                return Ok(dict.into());
            }, // If everything is okay, return Ok(())
            Err(ArgminError) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                ArgminError.to_string()
            )),
        }
    })
}

#[pyfunction]
pub fn dpln_sample(network_params: Vec<f64>, n: usize) -> Vec<f64> {

    sample(network_params, n)
}

#[pyfunction]
pub fn dpln_pdf(xs: Vec<f64>, network_params: Vec<f64>) -> Vec<f64> {

    pdf(xs, network_params)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn nd_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(network_dur, m)?)?;
    m.add_function(wrap_pyfunction!(network_from_source_and_targets, m)?)?;
    m.add_function(wrap_pyfunction!(network_from_vars, m)?)?;
    m.add_function(wrap_pyfunction!(sbm_from_vars, m)?)?;
    m.add_function(wrap_pyfunction!(sbm_duration, m)?)?;
    m.add_function(wrap_pyfunction!(build_ER, m)?)?;
    m.add_function(wrap_pyfunction!(build_DCSBM, m)?)?;
    m.add_function(wrap_pyfunction!(dpln_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(dpln_sample, m)?)?;
    m.add_function(wrap_pyfunction!(fit_dpln, m)?)?;
    m.add_function(wrap_pyfunction!(small_sellke, m)?)?;
    m.add_function(wrap_pyfunction!(big_sellke, m)?)?;
    m.add_function(wrap_pyfunction!(get_r0, m)?)?;
    m.add_function(wrap_pyfunction!(get_fs, m)?)?;
    m.add_function(wrap_pyfunction!(sellke_dur, m)?)?;
    m.add_function(wrap_pyfunction!(gillesp_dur_gr, m)?)?;
    m.add_function(wrap_pyfunction!(gillesp_sbm_gr, m)?)?;
    m.add_function(wrap_pyfunction!(gillesp_dur, m)?)?;
    m.add_function(wrap_pyfunction!(gillesp_dur_sbm, m)?)?;
    m.add_function(wrap_pyfunction!(gillespie_sbm, m)?)?;
    m.add_function(wrap_pyfunction!(gillespie_gmm, m)?)?;
    m.add_function(wrap_pyfunction!(gillesp_dur_sc, m)?)?;
    m.add_function(wrap_pyfunction!(gillesp_gmm_sc, m)?)?;
    m.add_function(wrap_pyfunction!(gillesp_sbm_sc, m)?)?;
    m.add_function(wrap_pyfunction!(gillesp_sbmdur_sc, m)?)?;
    m.add_function(wrap_pyfunction!(small_sbm_dur, m)?)?;
    m.add_function(wrap_pyfunction!(small_gillespie_dur, m)?)?;
    m.add_function(wrap_pyfunction!(small_gillespie, m)?)?;
    m.add_function(wrap_pyfunction!(dur_r0, m)?)?;
    m.add_function(wrap_pyfunction!(gmm_sims, m)?)?;
    m.add_function(wrap_pyfunction!(gmm_sims_sc, m)?)?;
    m.add_function(wrap_pyfunction!(big_sellke_growth_rate, m)?)?;
    m.add_function(wrap_pyfunction!(big_sellke_sec_cases, m)?)?;
    Ok(())
}