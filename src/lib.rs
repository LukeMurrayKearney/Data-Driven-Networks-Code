use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use crate::dpln::{pdf, sample};
use crate::network_structure::NetworkStructure;

mod network_structure;
mod distributions;
mod dpln;
mod connecting_stubs;
mod network_properties;
mod run_model;

////////////////////////////////////////////// Network Creation ////////////////////////////////////////

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


//////////////////////////////////////////// outbreak simulation //////////////////////////////////////

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
    let network = NetworkStructure{
        adjacency_matrix: adjacency_matrix.clone(),
        degrees: adjacency_matrix.iter().map(|x| x.len()).collect(),
        ages: ages,
        frequency_distribution: Vec::new(),
        partitions: partitions
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
    m.add_function(wrap_pyfunction!(network_from_vars, m)?)?;
    m.add_function(wrap_pyfunction!(sbm_from_vars, m)?)?;
    m.add_function(wrap_pyfunction!(dpln_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(dpln_sample, m)?)?;
    m.add_function(wrap_pyfunction!(fit_dpln, m)?)?;
    m.add_function(wrap_pyfunction!(small_sellke, m)?)?;
    m.add_function(wrap_pyfunction!(big_sellke, m)?)?;
    m.add_function(wrap_pyfunction!(big_sellke_growth_rate, m)?)?;
    m.add_function(wrap_pyfunction!(big_sellke_sec_cases, m)?)?;
    Ok(())
}