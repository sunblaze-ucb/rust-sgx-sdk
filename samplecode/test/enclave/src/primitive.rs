use sgx_types::*;
use std::vec::*;
use std::ptr;
use std::slice;
use std::fmt::Debug;

use rand::{Rng, thread_rng};
use rand::distributions::{Sample, Normal, IndependentSample};

use linalg::{Matrix, BaseMatrix, BaseMatrixMut};
use linalg::Vector;

//use rusty_machine::linalg::{Matrix, BaseMatrix, Vector};
use rusty_machine::learning::SupModel;
use rusty_machine::learning::logistic_reg::LogisticRegressor;
use rusty_machine::learning::dp_logistic_reg::DPLogisticRegressor;
use rusty_machine::datasets::iris;

/// Logarithm for applying within cost function.
fn ln(x: f64) -> f64 {
    x.ln()
}

/// Trait for activation functions in models.
trait ActivationFunc: Clone + Debug {
    /// The activation function.
    fn func(x: f64) -> f64;

    /// The gradient of the activation function.
    fn func_grad(x: f64) -> f64;

    /// The gradient of the activation function calculated using the output of the function.
    /// Calculates f'(x) given f(x) as an input
    fn func_grad_from_output(y: f64) -> f64;

    /// The inverse of the activation function.
    fn func_inv(x: f64) -> f64;
}

/// Sigmoid activation function.
#[derive(Clone, Copy, Debug)]
struct Sigmoid;

impl ActivationFunc for Sigmoid {
    /// Sigmoid function.
    ///
    /// Returns 1 / ( 1 + e^-t).
    fn func(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Gradient of sigmoid function.
    ///
    /// Evaluates to (1 - e^-t) / (1 + e^-t)^2
    fn func_grad(x: f64) -> f64 {
        Self::func(x) * (1f64 - Self::func(x))
    }

    fn func_grad_from_output(y: f64) -> f64 {
        y * (1f64 - y)
    }

    fn func_inv(x: f64) -> f64 {
        (x / (1f64 - x)).ln()
    }
}

/// Trait for cost functions in models.
trait CostFunc<T> {
    /// The cost function.
    fn cost(outputs: &T, targets: &T) -> f64;

    /// The gradient of the cost function.
    fn grad_cost(outputs: &T, targets: &T) -> T;
}

/// The cross entropy error cost function.
#[derive(Clone, Copy, Debug)]
struct CrossEntropyError;

impl CostFunc<Matrix<f64>> for CrossEntropyError {
    fn cost(outputs: &Matrix<f64>, targets: &Matrix<f64>) -> f64 {
        // The cost for a single
        let log_inv_output = (-outputs + 1f64).apply(&ln);
        let log_output = outputs.clone().apply(&ln);

        let mat_cost = targets.elemul(&log_output) + (-targets + 1f64).elemul(&log_inv_output);

        let n = outputs.rows();

        -(mat_cost.sum()) / (n as f64)
    }

    fn grad_cost(outputs: &Matrix<f64>, targets: &Matrix<f64>) -> Matrix<f64> {
        (outputs - targets).elediv(&(outputs.elemul(&(-outputs + 1f64))))
    }
}

impl CostFunc<Vector<f64>> for CrossEntropyError {
    fn cost(outputs: &Vector<f64>, targets: &Vector<f64>) -> f64 {
        // The cost for a single
        let log_inv_output = (-outputs + 1f64).apply(&ln);
        let log_output = outputs.clone().apply(&ln);

        let mat_cost = targets.elemul(&log_output) + (-targets + 1f64).elemul(&log_inv_output);

        let n = outputs.size();

        -(mat_cost.sum()) / (n as f64)
    }

    fn grad_cost(outputs: &Vector<f64>, targets: &Vector<f64>) -> Vector<f64> {
        (outputs - targets).elediv(&(outputs.elemul(&(-outputs + 1f64))))
    }
}

/// Generate a vector obeying normal distribution.
///
/// # Examples
///
/// ```
/// use rusty_machine::learning::toolkit::rand_utils;
///
/// // Generate a noise vector
/// let mut noise = normal_vector(0.1, 10)
/// ```
fn normal_vector(std_dev: f64, len: usize) -> Vector<f64> {
    let mut rng = thread_rng();
    let mut normal = Normal::new(0.0, std_dev);
    let mut res = Vec::with_capacity(1000);

    for _ in 0..len {
        res.push(normal.sample(&mut rng));
    }

    Vector::from(res)
}

// We need to change f64 to u8 after adding crypto things.
// Can we merge in_ptr and out_ptr? leave this later.
#[no_mangle]
pub extern "C"
fn add_normal_noise(std_dev: f64, 
                    in_ptr: *mut f64, 
                    len: usize, 
                    out_ptr: *mut f64) 
                    -> sgx_status_t {
    println!("Adding normal noise...");
    let in_slice = unsafe { slice::from_raw_parts(in_ptr, len)};
    let mut in_vector = Vector::new(in_slice)+normal_vector(std_dev, len); 
    unsafe{
        ptr::copy_nonoverlapping(in_vector.mut_data().as_ptr(),
                                 out_ptr,
                                 len);
    }
    sgx_status_t::SGX_SUCCESS    
}

#[no_mangle]
pub extern "C"
fn compute_grad(in_params: *const f64,
                params_len: usize,
                inputs: *const f64,
                inputs_len: usize,
                targets: *const f64,
                targets_len: usize,
                out_params: *mut f64)
                -> sgx_status_t {
    println!("Computing Gradient");
    let params_slice = unsafe { slice::from_raw_parts(in_params, params_len)};
    let beta_vec = Vector::new(params_slice.to_vec());
    let inputs_slice = unsafe { slice::from_raw_parts(inputs, inputs_len)};
    println!("{} {} {}", targets_len, params_len, inputs_slice.len());
    let inputs_mat = &Matrix::new(targets_len, params_len, inputs_slice);
    let targets_slice = unsafe { slice::from_raw_parts(targets, targets_len)};
    let targets_vec = &Vector::new(targets_slice.to_vec());
    let outputs = (inputs_mat * beta_vec).apply(&Sigmoid::func);

    let cost = CrossEntropyError::cost(&outputs, targets_vec);
    let mut grad = (inputs_mat.transpose() * (outputs - targets_vec)) / (inputs_mat.rows() as f64);
    
    /*let sample_num = targets_vec.size() as f64;
    // This is hard-coded in gradient descent codes.
    let iters = 100.0;
    // We do not have regularization term here so this is 1 for LR.
    let L = 1.0;
    // We now hard-code the eps and delta here, will change this to the arguments later.
    let eps = 1.0;
    let delta = 0.00001;
    let std_dev = 4.0*L*(iters*((1.0/delta) as f64).log2()).sqrt()/(sample_num*eps);
    let noise = Vector::new(normal_vector(std_dev, params.len()));
    println!("{:?}", grad);
    println!("{:?}", noise);
    let noisy_grad = noise+grad;
    println!("{:?}", noisy_grad);

    (cost, noisy_grad.into_vec())*/

    unsafe{
        ptr::copy_nonoverlapping(grad.mut_data().as_ptr(),
                                 out_params,
                                 params_len);
    }

    sgx_status_t::SGX_SUCCESS
}

#[no_mangle]
pub extern "C"
fn sample_main() -> sgx_status_t {
    println!("DP regression example on Iris dataset:");

    println!("Load Iris dataset:");
    let dataset = iris::load();
    let mut samples = dataset.data();
    let samples_size = samples.data().len()*8;
    let targets = dataset.target();
    let targets_size = targets.data().len()*8;

    /*println!("Encrypt Iris dataset:");
    let serializable_samples: *const u8 = unsafe{
        mem::transmute::<*const f64, *const u8>(samples.data().as_ptr())
    };
    //let serialized_targets = ;
    //let aes_gcm_plaintext: [u8;16] = [0; 16];
    //let plaintext_ptr = &aes_gcm_plaintext as *const u8;
    let aes_gcm_key: [u8;16] = [0; 16];
    let aes_gcm_iv: [u8;12] = [0; 12];
    let mut aes_gcm_ciphertext: [u8;3200] = [0;3200];
    let ciphertext_ptr = unsafe{
        mem::transmute::<&[u8;3200], *mut u8>(&aes_gcm_ciphertext)
    };
    let mut aes_gcm_mac: [u8;16] = [0;16];
    println!("[+] aes-gcm-128 args prepared!");
    println!("[+] aes-gcm-128 expected ciphertext: {}", "0388dace60b6a392f328c2b971b2fe78");
    let sgx_ret = unsafe{
        aes_gcm_128_encrypt(&aes_gcm_key,
                           serializable_samples,
                           samples_size,
                           &aes_gcm_iv,
                           ciphertext_ptr,
                           &mut aes_gcm_mac)
    };
    println!("[+] aes-gcm-128 returned from enclave!");
    println!("{:?}", &aes_gcm_ciphertext[0..16]);

    println!("Decrypt Iris dataset:");
    let mut decrypted_samples: [f64;400] = [0.0;400];
    let decrypt_ptr = unsafe{
        mem::transmute::<&[f64;400], *mut u8>(&decrypted_samples)
    };
    let sgx_ret = unsafe{
        aes_gcm_128_decrypt(&aes_gcm_key,
                           ciphertext_ptr,
                           samples_size,
                           &aes_gcm_iv,
                           &aes_gcm_mac,
                           decrypt_ptr)
    };
    println!("{:?}", &decrypted_samples[0..16]);
    println!("{:?}", &decrypted_samples[384..400]);
    */
    let mut model = DPLogisticRegressor::default();

    // Train the model
    println!("Training the model...");
    model.train(&samples, &targets).unwrap();

    // Predict the classes and partition into
    println!("Classifying the samples...");
    //let classes = model.predict(&samples).unwrap().into_iter().collect::<Vec<_>>();
    let classes = model.predict(&samples).unwrap().into_iter().map(|x|if x > 0.5 {return 1.0;} else {return 0.0;}).collect::<Vec<_>>();
    println!("{:?}", classes);
    println!("{:?}", targets);
    println!("{:?}", model.parameters().unwrap());
    let matching = classes.into_iter().zip(targets.into_iter()).filter(|(a, b)| a==*b ).count();
    println!("{}", matching);

    sgx_status_t::SGX_SUCCESS
}
