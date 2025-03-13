use argmin::{
    solver::neldermead::NelderMead,
    core::{CostFunction, Error, Executor}
};
use statrs::{distribution::{Normal, ContinuousCDF, Continuous, Exp, Gamma}, statistics::Statistics};
// use statrs::statistics::Statistics;
// use rand_distr::Exp;
use rand::{distributions::Distribution, rngs::ThreadRng};
use rand::{thread_rng, Rng};

#[derive(Debug)]
pub struct Parameters {
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    pub nu: Vec<f64>,
    pub tau: Vec<f64>
}

impl Parameters {
    fn new(iters: usize) -> Parameters {
        Parameters {
            alpha: vec![0.0; iters+1],
            beta: vec![0.0; iters+1],
            nu: vec![0.0; iters+1],
            tau: vec![0.0; iters+1],
        }
    }
    fn iterate(self: &mut Self, alpha: f64, beta: f64, nu: f64, tau: f64, index: usize) {
        self.alpha[index] = alpha;
        self.beta[index] = beta;
        self.nu[index] = nu; 
        self.tau[index] = tau;
    }
}

// Define a cost function
struct Likelihood {
    data: Vec<f64>
}

impl Likelihood {
    pub fn new(data: Vec<f64>) -> Self {
        Likelihood { data }
    }
}

impl CostFunction for Likelihood {
    type Param = Vec<f64>;
    type Output = f64;

    // Function that computes the cost
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(dpln_likelihood(&self.data, p))
    }
}

pub fn sample(params: Vec<f64>, n: usize) -> Vec<f64> {

    let alpha: f64 = params[0];
    let beta: f64 = params[1];
    let nu: f64 = params[2];
    let tau: f64 = params[3];
    let mut rng: ThreadRng = thread_rng();

    let normal: Normal = Normal::new(0., 1.).unwrap();
    let exp = Exp::new(1.).unwrap();

    (0..n).map(|_| (nu + tau*normal.sample(&mut rng) + (1./alpha)*exp.sample(&mut rng) - (1./beta)*exp.sample(&mut rng)).exp() - 1.).collect::<Vec<f64>>()
}

pub fn pdf(xs: Vec<f64>, params: Vec<f64>) -> Vec<f64> {

    // let params = fit_dpln(data, iters, prior_params).unwrap();
    
    let alpha: f64 = params[0];
    let beta: f64 = params[1];
    let nu: f64 = params[2];
    let tau: f64 = params[3];

    let normal: Normal = Normal::new(0., 1.).unwrap();
    
    // // pdf of dpln
    // let f = xs
    //     .iter()
    //     .map(|&x| {
    //         (alpha*beta / (alpha + beta)) * (calc_a(alpha, nu, tau) * x.powf(-alpha - 1.) * normal.cdf((x.ln() - nu - alpha*tau.powi(2)) / tau) + 
    //         x.powf(beta-1.) * calc_a(-beta, nu, tau) * (1. - normal.cdf((x.ln() - nu + beta*tau.powi(2)) / tau)))
    //     })
    //     .collect();


    // Ramirez approch 'Bayesian Inference for double pareto lognormal queues'
    let f = xs
        .iter()
        .map(|&x| {
            (alpha*beta / (alpha + beta)) * x.powi(-1) * normal.pdf((x.ln() - nu)/tau) * 
            (mills_ratio(alpha*tau - (x.ln() - nu)/tau) + mills_ratio(beta*tau + (x.ln() - nu)/tau))
        })
        .collect();

    // // Reed and Jorgensen mixed approach, cited in the same paper.
    // let f = xs
    //     .iter()
    //     .map(|&x| {
    //         (beta/(alpha+beta)) * 
    //         alpha*x.powf(- alpha - 1.) *
    //         (alpha*nu + alpha.powi(2)*tau.powi(2)/2.).exp() *
    //         normal.cdf((x.ln() - nu - alpha*tau.powi(2))/tau) +
    //         (alpha/(alpha + beta)) * 
    //         beta * x .powf(- beta - 1.) *
    //         (-beta*nu + beta.powi(2)*tau.powi(2)/2.).exp() *
    //         normal.cdf((x.ln() - nu + beta*tau.powi(2))/tau)
    //     })
    //     .collect();

    // // grbac note solution
    // let f = xs
    //     .iter()
    //     .map(|&x| {
    //         (alpha*beta)/(alpha+beta) * (
    //             calc_a(alpha, nu, tau) * x.powf(-alpha-1.) *normal.cdf((x.ln() - nu - alpha*tau.powi(2)) / tau) +
    //             calc_a(-beta, nu, tau) * x.powf(beta-1.) * (1. - normal.cdf((x.ln() - nu + beta*tau.powi(2))/tau))
    //         )
    //     })
    //     .collect();

    // scipy things 
    // let f = xs
    //     .iter()
    //     .map(|&x| {
    //         let a_plus_b = alpha + beta;
    //         let log_f1 = alpha.ln() + x.ln() * (-1. - alpha) + (alpha*nu + alpha.powi(2) * tau / 2.) +
    //             normal.cdf((x.ln() - nu - alpha * tau)/tau.sqrt()).ln();
    //         let log_f2 = beta.ln() + x.ln() * (beta - 1.) + (-beta * nu + beta.powi(2) * tau / 2.) +
    //             normal.cdf(-((x.ln() - nu + beta * tau)/tau.sqrt())).ln();

    //         beta/a_plus_b * log_f1.exp() + alpha/a_plus_b * log_f2.exp()
    //     })
    //     .collect();
    f
}

fn mills_ratio(x: f64) -> f64 {
    let normal = Normal::new(0., 1.).unwrap();
    (1.-normal.cdf(x))/normal.pdf(x)
}

// fn calc_a(theta: f64, nu: f64, tau: f64) -> f64{
//     (theta*nu + tau.powi(2)*tau.powi(2)/2.0).exp()
// }

pub fn fit_dpln(data: Vec<f64>, iters: usize, prior_params: Vec<f64>) -> Result<Parameters, Error> {

    // use gibbs sampling to fit to the dpln distribution

    // define parameter object and data to be optimised against
    let mut params: Parameters = Parameters::new(iters);
    let cost = Likelihood::new(data.clone());
    let n: usize = data.len();
    // set up random number generator
    let mut rng = thread_rng();

    // Set up solver, based on fit from danon et al 2012, (alpha, beta, tau) 
    let solver = NelderMead::<Vec<f64>, f64>::new(vec![
        vec![0.1,5.5,0.5],
        vec![1.0,7.0,1.0],
        vec![0.5,4.0,2.0],
    ])
        .with_sd_tolerance(0.00001)?;

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(30_000))
        // .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;
    
    // Save MLEs to first iteration of the Gibbs sampler
    let mle_params: Vec<f64> = res.state.best_param.unwrap();
    params.alpha[0] = mle_params[0];
    params.beta[0] = mle_params[1];
    params.nu[0] = data.iter().sum::<f64>() / (data.len() as f64) - 1.0/params.alpha[0] - 1.0/params.beta[0];
    params.tau[0] = mle_params[2];


    // Begin Gibbs sampling
    for i in 1..(iters+1) {
        // Generate W and Z
        let alpha = params.alpha[i-1]; 
        let beta = params.beta[i-1];
        let nu = params.nu[i-1];
        let tau = params.tau[i-1];
        // create a standard normal distribution for cdf and pdf pulls
        let normal = Normal::new(0.0, 1.0).unwrap();

        let probs: Vec<f64> = data.iter().map(|&x| {
            normal.cdf( - ( x - nu + beta*(tau.powi(2)) ) / tau ) / (
            normal.cdf( - ( x - nu + beta*(tau.powi(2)) ) / tau ) + 
            (1.0 - normal.cdf( - ( x - nu - alpha*(tau.powi(2)) ) / tau) ) * 
            (( 0.5/(tau.powi(2)) ) * (( x - nu - alpha*(tau.powi(2)) ).powi(2) -
             ( x - nu + beta*(tau.powi(2))).powi(2) )).exp()
            )
        })
        .collect();

        
        // generate random numbers to compare to probability
        let random_nums: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
        let u: Vec<bool> = random_nums
            .iter()
            .enumerate()
            .map(|(idx, &x)| {
                if x < probs[idx] {
                    true
                } 
                else {
                    false
                }
            })
            .collect();
        
        let mu1: Vec<f64> = data
            .iter()
            .enumerate()
            .filter(|(idx, _)| u[idx.to_owned()])
            .map(|(_, &x)| {
                - ( x -  nu + beta*(tau.powi(2)))
            })
            .collect();
        
        let mu2: Vec<f64> = data
            .iter()
            .enumerate()
            .filter(|(idx, _)| !u[idx.to_owned()])
            .map(|(_, &x)| {
                x -  nu - alpha*(tau.powi(2))
            })
            .collect();

        let w1: Vec<f64> = truncate_norm_vector(&mu1, tau, &mut rng);
        let w2: Vec<f64> = truncate_norm_vector(&mu2, tau, &mut rng);
        
        let mut idx_w1: usize = 0; let mut idx_w2: usize = 0; 
        let w: Vec<f64> = u
            .iter()
            .map(|&x| {
                match x {
                    true => {
                        idx_w1 += 1;
                        -w1[idx_w1 - 1]
                    },
                    false => {
                        idx_w2 += 1;
                        w2[idx_w2 - 1]
                    }
                }
            })
            .collect();

        // Generation of E1 and E2
        let e1: Vec<f64> = w.iter().map(|&x| {
            // again exp is upside down??? weird
            let exp = Exp::new(alpha + beta).unwrap();
            if x < 0.0 {
                exp.sample(&mut rng)
            }
            else {
                exp.sample(&mut rng) + x
            }
        })
        .collect();
        let e2: Vec<f64> = e1.iter().enumerate().map(|(idx,&x)| x - w[idx]).collect();

        // Generate nu and tau
        let z: Vec<f64> = w.iter().enumerate().map(|(idx, &x)| data[idx] - x).collect();
        let (nu, tau) = gen_nu_tau(z, &prior_params, tau, &mut rng);

        // Generate alpha and beta
        let mut alpha = 0.;
        let mut beta = 0.;
        if Gamma::new(prior_params[4] + (n as f64), prior_params[5] + (n as f64)*(e1.clone().mean())).is_err() {
            let gamma = Gamma::new(prior_params[4] + (n as f64), prior_params[5]).unwrap();
            alpha = gamma.sample(&mut rng);
        }
        else {
            let gamma = Gamma::new(prior_params[4] + (n as f64), prior_params[5] + (n as f64)*(e1.mean())).unwrap();
            alpha = gamma.sample(&mut rng);
        }
        
        if Gamma::new(prior_params[4] + (n as f64), prior_params[5] + (n as f64)*(e2.clone().mean())).is_err() {
            let gamma = Gamma::new(prior_params[4] + (n as f64), prior_params[5]).unwrap();
            beta = gamma.sample(&mut rng);
        }
        else {
            let gamma = Gamma::new(prior_params[4] + (n as f64), prior_params[5] + (n as f64)*(e2.mean())).unwrap();
            beta = gamma.sample(&mut rng);
        }
        
        
        // assign step values
        params.iterate(alpha, beta, nu, tau, i);
    }
    Ok(params)
}

fn dpln_likelihood(data: &Vec<f64>, params: &Vec<f64>) -> f64 {
    
    // make sure params stay positive
    for &x in params.iter() {
        if x <= 0.0 {return f64::INFINITY;}
    }
    // Initital values by calculating MLE from Reed and Jorgensen, 2004
    let mean: f64 = data.iter().sum::<f64>() / (data.len() as f64);
    let n: f64 = data.len() as f64;
    // create a standard normal distribution for cdf and pdf pulls
    let normal = Normal::new(0.0, 1.0).unwrap();
    // iterate through data points
    let mut m: Vec<f64> = Vec::new();
    for &x in data.iter() {
        let p: f64 = params[0]*params[2] - (x - mean + 1.0/params[0] - 1.0/params[1])/params[2];
        let q: f64 = params[1]*params[2] + (x - mean + 1.0/params[0] - 1.0/params[1])/params[2];
        m.push(
            ((1.0 - normal.cdf(p))*normal.pdf(q) + (1.0 - normal.cdf(q))*normal.pdf(p)).ln() 
            + 0.5*((2.0*std::f64::consts::PI).ln() + p.powi(2)) 
            + 0.5*((2.0*std::f64::consts::PI).ln() + q.powi(2))
        );
    }
    // now calculate likelihood
    let f: f64 = 
        n*params[0].ln() + n*params[1].ln() - n*(params[0] + params[1]).ln() 
        + data.iter().map(|&y| {
            -0.5*((2.0*std::f64::consts::PI).ln() + ((y - mean + 1.0/params[0] - 1.0/params[1]) / params[2]).powi(2))
        }).sum::<f64>()
        + m.iter().sum::<f64>();

    if f.is_nan() || f.is_infinite() {
        f64::INFINITY
    }
    else {
        -f
    }
}

fn truncate_norm_vector(mu: &Vec<f64>, sigma: f64, rng: &mut ThreadRng) -> Vec<f64> {
    
    // generating k values of w , w~N(mu, sigma) where w > wmin
    let k = mu.len();
    let mut w: Vec<f64> = vec![0.0; k];
    let wmin = f64::EPSILON;
    let mulim: Vec<f64> = mu.iter().map(|&x| (wmin - x) / sigma).collect();

    // seperate the above and the below 0.45 mulim values
    let mut z: Vec<usize> = mulim.iter().map(|&x| {
        if x < 0.45 {
            1
        }
        else {
            0
        }
    })
    .collect();

    // number of values less than 0.45
    let mut sumz: usize = z.iter().sum();

    // values greater than 0.45
    let mut zz: Vec<usize> = mulim.iter().map(|&x| {
        if x >= 0.45 {
            1
        }
        else {
            0
        }
    })
    .collect();
    let mut sumzz: usize = zz.iter().sum::<usize>();

    // set w values using normal random below, only for positive samples (mulim > 0.45)
    while sumz > 0 {
        for (idx, _) in z.iter().enumerate().filter(|(_, &x)| x == 1) {
            let normal = Normal::new(mu[idx], sigma).unwrap();
            w[idx] = normal.sample(rng);
        }
        for (i, _) in w.iter().enumerate().filter(|(_, &x)| x > wmin) {
            z[i] = 0;
        }
        sumz = z.iter().sum::<usize>(); 
    }


    //mulim > 0.45
    while sumzz > 0 {
        let random_nums: Vec<f64> = (0..sumzz).map(|_| rng.gen()).collect();
        let mut rand_index: usize = 0;
        for (idx, _) in zz.clone().iter().enumerate().filter(|(_,&x)| x == 1) {
            // rate of exponential distribution is upside down for some reason?
            let exp = Exp::new(mulim[idx]).unwrap();
            let ww: f64 = exp.sample(rng) + mulim[idx];
            let p = (-0.5*(ww.powi(2) + mulim[idx].powi(2)) + mulim[idx]*ww).exp();
            if random_nums[rand_index] < p {
                w[idx] = ww*sigma + mu[idx];
                zz[idx] = 0;
            }
            else {
                w[idx] = ww;
            }
            rand_index += 1;
        }

        sumzz = zz.iter().sum::<usize>();
    }
    w
}

fn gen_nu_tau(z: Vec<f64>, params: &Vec<f64>, tau: f64, rng: &mut ThreadRng) -> (f64, f64) {
    
    // generate nu, tau given a sample z_i ~ N(nu, tau)
    // nu|tau ~ N(m, tau/sqrt(k))
    // invtau ~ Gamma(a/2, b/2) where inverse tau is 1/tau^2

    let a = params[0]; let b = params[1]; let m = params[2]; let k = params[3];
    
    // mean of z
    let n: f64 = z.len() as f64;
    let zbar: f64 = z.clone().mean(); 
    // variance of z
    let s2: f64 = z.clone().variance();
    
    let apost: f64 = if a.abs() < 0.0001 { a + n - 1.0 } else { a + n };
    let bpost = b + (n - 1.0)*s2 + (k*n/(k + n))*((zbar - m).powi(2));
    if Gamma::new(apost/2.0, bpost/2.0).is_err() {
        let gamma = Gamma::new(a/2.0, b/2.0).unwrap();
        let normal = Normal::new(0.0, tau/((k+n).sqrt())).unwrap();
        return (normal.sample(rng), 1.0/(gamma.sample(rng).sqrt()));
    }
    
    let gamma = Gamma::new(apost/2.0, bpost/2.0).unwrap();
    let normal = Normal::new((k*m + n*zbar)/(k + n), tau/((k+n).sqrt())).unwrap();

    (normal.sample(rng), 1.0/(gamma.sample(rng).sqrt()))
}