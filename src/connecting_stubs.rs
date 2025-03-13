use rand::{rngs::ThreadRng, Rng};

pub fn cleanup_single(source: &Vec<(usize, usize)>, target: &Vec<(usize, usize)>, old_edges: &Vec<(usize, usize)>, rng: &mut ThreadRng) -> 
    (Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<(usize, usize)>) {
    
    // order and remove zeros
    let mut edge_list: Vec<(usize, usize)> = Vec::new();
    let mut source: Vec<(usize, usize)> = source
        .iter()
        .filter(|(_, x)| *x != 0)
        .map(|(i,x)| (*i, *x))
        .collect();
    source.sort_by(|a,b| b.1.cmp(&a.1));

    let mut target: Vec<(usize, usize)> = target
        .iter()
        .filter(|(_, x)| *x != 0)
        .map(|(i,x)| (*i, *x))
        .collect();


    while source.len() > 0 && target.len() > 0  {
        edge_list.push(connect_stub(&mut source, &mut target, rng));
    } 

    // new code
    while source.len() > 0 && target.len() > 0  {
        // find edges which are already connected to source
        let remove_new: Vec<usize> = edge_list.clone()
            .into_iter()
            .filter(|&(a, b)| a == source[0].0 || b == source[0].0)
            .flat_map(|(a, b)| vec![a, b])
            .filter(|&x| x != source[0].0)
            .collect();
        // remove already connected edges from contention
        let tmp_target1: Vec<(usize, usize)> = target
            .iter()
            .filter(|(i, _)| !remove_new.contains(i))
            .map(|x| x.to_owned())
            .collect();
        let remove_old: Vec<usize> = old_edges.clone()
            .into_iter()
            .filter(|&(a, b)| a == source[0].0 || b == source[0].0)
            .flat_map(|(a, b)| vec![a, b])
            .filter(|&x| x != source[0].0)
            .collect();
        // remove already connected edges from contention
        let tmp_target: Vec<(usize, usize)> = tmp_target1
            .iter()
            .filter(|(i, _)| !remove_old.contains(i))
            .map(|x| x.to_owned())
            .collect();


        match tmp_target.is_empty() {
            true => {
                // remove source if no target
                source.remove(0);
            },
            false => {
                let i = rng.gen_range(0..tmp_target.len());
                edge_list.push((source[0].0, tmp_target[i].0));
                // index of target in original list
                let index = tmp_target
                    .iter()
                    .position(|x| x.0 == tmp_target[i].0)
                    .unwrap();
                //reduce or delete target
                match tmp_target[i].1 {
                    0..=1 => {
                        target.remove(index);
                    },
                    _ => {
                        target[index].1 -= 1;
                    }
                }
                //reduce or delete index node
                match source[0].1 {
                    0..=1 => {
                        source.remove(0);
                    },
                    _ => {
                        source[0].1 -= 1;
                    }
                }
                move_element(&mut source);
            }
        }
    }

    (edge_list, source, target)
    }


pub fn cleanup_double(source: &Vec<(usize, usize)>, target1: &Vec<(usize, usize)>, target2: &Vec<(usize, usize)>, rng: &mut ThreadRng) -> 
    (Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<(usize, usize)>) {
        // order and remove zeros
        let mut edge_list: Vec<(usize, usize)> = Vec::new();
        let mut source: Vec<(usize, usize)> = source
            .iter()
            .filter(|(_, x)| *x != 0)
            .map(|(i,x)| (*i, *x))
            .collect();
        source.sort_by(|a,b| b.1.cmp(&a.1));

        let mut target1: Vec<(usize, usize)> = target1
            .iter()
            .filter(|(_, x)| *x != 0)
            .map(|(i,x)| (*i, *x))
            .collect();
        // target1.sort_by(|a,b| b.1.cmp(&a.1));

        let mut target2: Vec<(usize, usize)> = target2
            .iter()
            .filter(|(_, x)| *x != 0)
            .map(|(i,x)| (*i, *x))
            .collect();
        // target2.sort_by(|a,b| b.1.cmp(&a.1));

        while source.len() > 0 && (target1.len() > 0 || target2.len() > 0) {
            if target1.len() > 0 && target2.len() > 0 {
                let neighbour: usize = rng.gen_range(0..=1);
                match neighbour {
                    0 => {
                        edge_list.push(connect_stub(&mut source, &mut target1, rng));
                    },
                    1 => {
                        edge_list.push(connect_stub(&mut source, &mut target2, rng));
                    },
                    _ => {println!("oh no")}
                }
            }
            else {
                match target1.len() {
                    0 => {
                        edge_list.push(connect_stub(&mut source, &mut target2, rng));
                    },
                    _ => {
                        edge_list.push(connect_stub(&mut source, &mut target1, rng));
                    }
                }
            }
        } 

    (edge_list, source, target1, target2)
    }

pub fn connect_stubs_diagonal(degrees: &Vec<(usize, usize)>, rng: &mut ThreadRng) -> 
    (Vec<(usize,usize)>, (Vec<(usize,usize)>, Vec<(usize,usize)>)) {

    // order and remove zeros
    let mut edge_list: Vec<(usize, usize)> = Vec::new();
    let mut nodes: Vec<(usize, usize)> = degrees
        .iter()
        .filter(|(_, x)| *x != 0)
        .map(|(i,x)| (*i, *x))
        .collect();
    nodes.sort_by(|a,b| b.1.cmp(&a.1));

    // go through node list from largest degree to smallest connecting and removing at each step
    while nodes.len() > 1 {
        // find edges which are already connected
        let remove: Vec<usize> = edge_list.clone()
            .into_iter()
            .filter(|&(a, b)| a == nodes[0].0 || b == nodes[0].0)
            .flat_map(|(a, b)| vec![a, b])
            .filter(|&x| x != nodes[0].0)
            .collect();
        // remove already connected edges from contention
        let tmp_nodes: Vec<(usize, usize)> = nodes
            .iter()
            .filter(|(i, _)| !remove.contains(i))
            .map(|x| x.to_owned())
            .collect();
        
        // check if there are possible target nodes
        match tmp_nodes.len() {
            0..=1 => {
                // remove source if no target
                nodes.remove(0);
            },
            _ => {
                //pick target
                let i = rng.gen_range(1..tmp_nodes.len());
                edge_list.push((nodes[0].0,tmp_nodes[i].0));
                // index of target in original list
                let index = nodes
                    .iter()
                    .position(|x| x.0 == tmp_nodes[i].0)
                    .unwrap();
                //reduce or delete target
                match tmp_nodes[i].1 {
                    0..=1 => {
                        nodes.remove(index);
                    },
                    _ => {
                        nodes[index].1 -= 1;
                    }
                }
                //reduce or delete index node
                match nodes[0].1 {
                    0..=1 => {
                        nodes.remove(0);
                    },
                    _ => {
                        nodes[0].1 -= 1;
                    }
                }
                move_element(&mut nodes);
            }
        }
    }

    (edge_list, (nodes.clone(),nodes))
}

// do the non-diagonal case next
pub fn connect_stubs(degrees1: &Vec<(usize, usize)>, degrees2: &Vec<(usize, usize)>, rng: &mut ThreadRng) -> 
    (Vec<(usize,usize)>, (Vec<(usize,usize)>, Vec<(usize,usize)>)) {

    // order and remove zeros
    let mut edge_list: Vec<(usize, usize)> = Vec::new();
    let mut degrees_a: Vec<(usize, usize)> = degrees1
        .iter()
        .filter(|(_, x)| *x != 0)
        .map(|(i,x)| (*i, *x))
        .collect();
    degrees_a.sort_by(|a,b| b.1.cmp(&a.1));
    // order and remove zeros
    let mut degrees_b: Vec<(usize, usize)> = degrees2
        .iter()
        .filter(|(_, x)| *x != 0)
        .map(|(i,x)| (*i, *x))
        .collect();
    

    // loop through a and b
    while degrees_a.len() > 0 && degrees_b.len() > 0 {

        // find edges which are already connected to source
        let remove: Vec<usize> = edge_list.clone()
            .into_iter()
            .filter(|&(a, b)| a == degrees_a[0].0 || b == degrees_a[0].0)
            .flat_map(|(a, b)| vec![a, b])
            .filter(|&x| x != degrees_a[0].0)
            .collect();
        // remove already connected edges from contention
        let tmp_degrees_b: Vec<(usize, usize)> = degrees_b
            .iter()
            .filter(|(i, _)| !remove.contains(i))
            .map(|x| x.to_owned())
            .collect();
        
        match tmp_degrees_b.is_empty() {
            true => {
                // remove source if no target
                degrees_a.remove(0);
            },
            false => {
                let i = rng.gen_range(0..tmp_degrees_b.len());
                edge_list.push((degrees_a[0].0, tmp_degrees_b[i].0));
                // index of target in original list
                let index = degrees_b
                    .iter()
                    .position(|x| x.0 == tmp_degrees_b[i].0)
                    .unwrap();
                //reduce or delete target
                match tmp_degrees_b[i].1 {
                    0..=1 => {
                        degrees_b.remove(index);
                    },
                    _ => {
                        degrees_b[index].1 -= 1;
                    }
                }
                //reduce or delete index node
                match degrees_a[0].1 {
                    0..=1 => {
                        degrees_a.remove(0);
                    },
                    _ => {
                        degrees_a[0].1 -= 1;
                    }
                }
                move_element(&mut degrees_a);
            }
        }
    }


    (edge_list, (degrees_a, degrees_b))
}

// worker functions 

fn move_element(vec: &mut Vec<(usize,usize)>) {
    if vec.is_empty() {
        return 
    }

    let first = vec[0].1;
    let mut i = 1;
    while i < vec.len() && vec[i].1 > first {
        vec.swap(i - 1, i);
        i += 1;
    }
}

// fn move_second_element(vec: &mut Vec<(usize, usize)>) {
//     if vec.len() < 2 {
//         return 
//     }

//     let second = vec[1].1;
//     let mut i = 2;
//     while i < vec.len() && vec[i].1 > second {
//         vec.swap(i - 1, i);
//         i += 1;
//     }
// }

fn connect_stub(source: &mut Vec<(usize, usize)>, target: &mut Vec<(usize, usize)>, rng: &mut ThreadRng) -> (usize,usize) {
    let i = rng.gen_range(0..target.len());
    let link = (source[0].0,target[i].0);
    //reduce or delete target
    match target[i].1 {
        0..=1 => {
            target.remove(i);
        },
        _ => {
            target[i].1 -= 1;
        }
    }
    //reduce or delete index node
    match source[0].1 {
        0..=1 => {
            source.remove(0);
        },
        _ => {
            source[0].1 -= 1;
        }
    }
    move_element(source);

    link
}