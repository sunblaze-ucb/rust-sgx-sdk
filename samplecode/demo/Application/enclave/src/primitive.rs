use sgx_types::*;
use sgx_tcrypto::*;
use std::vec::*;
use std::ptr;
use std::slice;
use std::fmt::Debug;
use std::mem;

use rand::{Rng, thread_rng, StdRng};
use rand::distributions::{Sample, Normal, IndependentSample};

use linalg::{Matrix, BaseMatrix, BaseMatrixMut};
use linalg::Vector;

//use rusty_machine::linalg::{Matrix, BaseMatrix, Vector};
use rusty_machine::learning::SupModel;
use rusty_machine::learning::logistic_reg::LogisticRegressor;
use rusty_machine::learning::dp_logistic_reg::DPLogisticRegressor;
use rusty_machine::datasets::iris;

use crypto::{aes_gcm_128_encrypt, aes_gcm_128_decrypt};

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
    let mut normal = Normal::new(0.0, std_dev);
    let mut res = Vec::with_capacity(1000);
    unsafe {
        let mut rng = StdRng::new().unwrap();
        for _ in 0..len {
            res.push(normal.ind_sample(&mut rng));
        }
    }
    Vector::from(res)
}

// We need to change f64 to u8 after adding crypto things.
// Can we merge in_ptr and out_ptr? leave this later.
#[no_mangle]
pub extern "C"
fn add_normal_noise(std_dev: f64, 
                    gradient: *mut u8, 
                    len: usize, 
                    key: &[u8;16],
                    iv: &[u8;12],
                    gradient_mac: &mut [u8;16])
                    -> sgx_status_t {

    // println!("Adding normal noise...");

    // println!("[+] Decrypting data...");
    let ciphertext_slice = unsafe { slice::from_raw_parts(gradient, len*8) };
    let mut plaintext_vec: Vec<f64> = vec![0.0;len];
    let mut plaintext_ptr = unsafe {
        mem::transmute::<*const f64, *mut u8>(plaintext_vec[..].as_ptr())
    };
    let mut plaintext_slice = unsafe { Vec::from_raw_parts(plaintext_ptr, len*8, len*8) };
    let aad_array: [u8; 0] = [0; 0];
    let d_result = rsgx_rijndael128GCM_decrypt(key,
                                               &ciphertext_slice,
                                               iv,
                                               &aad_array,
                                               gradient_mac,
                                               &mut plaintext_slice);

    // println!("[+] Adding noise...");
    // println!("[+] plain text length {} {}", plaintext_vec.len(), len);
    let mut result = Vector::new(&plaintext_vec[..])+normal_vector(std_dev, len);
    // println!("[+] After adding noise...");
    let result_ptr = unsafe{
        mem::transmute::<*const f64, *const u8>(result.data().as_ptr())
    };
    let result_slice = unsafe { slice::from_raw_parts(result_ptr, len*8) };

    // println!("[+] Encrypting Data...");
    let mut ciphertext_vec: Vec<u8> = vec![0; len*8];
    let ciphertext_slice = &mut ciphertext_vec[..];
    let mut mac_array: [u8; 16] = [0; 16];
    let e_result = rsgx_rijndael128GCM_encrypt(key,
                                               result_slice,
                                               iv,
                                               &aad_array,
                                               ciphertext_slice,
                                               &mut mac_array);
    unsafe{
        ptr::copy_nonoverlapping(ciphertext_slice.as_ptr(),
                                 gradient,
                                 len*8);
    }
    *gradient_mac = mac_array;
    mem::forget(plaintext_slice);
    sgx_status_t::SGX_SUCCESS    
}

#[no_mangle]
pub extern "C"
fn compute_grad(params: *const u8,
                params_len: usize,
                inputs: *const u8,
                inputs_len: usize,
                targets: *const u8,
                targets_len: usize,
                gradient: *mut u8,
                key: &[u8;16],
                iv: &[u8;12],
                model_mac: &[u8;16],
                inputs_mac: &[u8;16],
                targets_mac: &[u8;16],
                gradient_mac: &mut [u8;16])
                -> sgx_status_t {

    // println!("Computing Gradient");

    // println!("[+] Decrypting Data...");
    // println!("[++] Decrypting Params...");
    let e_params_slice = unsafe { slice::from_raw_parts(params, params_len*8) };
    let mut params_vec: Vec<f64> = vec![0.0;params_len];
    let mut params_ptr = unsafe {
        mem::transmute::<*const f64, *mut u8>(params_vec[..].as_ptr())
    };
    let mut params_slice = unsafe { Vec::from_raw_parts(params_ptr, params_len*8, params_len*8)};
    let aad_array: [u8; 0] = [0; 0];
    let params_result = rsgx_rijndael128GCM_decrypt(key,
                                                    &e_params_slice,
                                                    iv,
                                                    &aad_array,
                                                    model_mac,
                                                    &mut params_slice);

    //println!("[++] Decrypting Inputs...");
    let e_inputs_slice = unsafe { slice::from_raw_parts(inputs, inputs_len*8) };
    let mut inputs_vec: Vec<f64> = vec![0.0;inputs_len];
    let mut inputs_ptr = unsafe {
        mem::transmute::<*const f64, *mut u8>(inputs_vec[..].as_ptr())
    };
    let mut inputs_slice = unsafe { Vec::from_raw_parts(inputs_ptr, inputs_len*8, inputs_len*8)};
    let inputs_result = rsgx_rijndael128GCM_decrypt(key,
                                                    &e_inputs_slice,
                                                    iv,
                                                    &aad_array,
                                                    inputs_mac,
                                                    &mut inputs_slice);

    //println!("[++] Decrypting Targets...");
    let e_targets_slice = unsafe { slice::from_raw_parts(targets, targets_len*8) };
    let mut targets_vec: Vec<f64> = vec![0.0;targets_len];
    let mut targets_ptr = unsafe {
        mem::transmute::<*const f64, *mut u8>(targets_vec[..].as_ptr())
    };
    let mut targets_slice = unsafe { Vec::from_raw_parts(targets_ptr, targets_len*8, targets_len*8)};
    let targets_result = rsgx_rijndael128GCM_decrypt(key,
                                                     &e_targets_slice,
                                                     iv,
                                                     &aad_array,
                                                     targets_mac,
                                                     &mut targets_slice);


    //println!("[++] Computing Gradient...");
    let beta_vec = Vector::new(params_vec);
    // println!("{} {} {}", targets_len, params_len, inputs_vec.len());
    let inputs_mat = &Matrix::new(targets_len, params_len, inputs_vec);
    let targets_vec = &Vector::new(targets_vec);
    let outputs = (inputs_mat * beta_vec).apply(&Sigmoid::func);
    let cost = CrossEntropyError::cost(&outputs, targets_vec);
    let mut result = (inputs_mat.transpose() * (outputs - targets_vec)) / (inputs_mat.rows() as f64);
    // println!("computed gradient:{:?}", result.data());

    //println!("[+] Encrypting Data...");
    let result_ptr = unsafe{
        mem::transmute::<*const f64, *const u8>(result.data().as_ptr())
    };
    let result_slice = unsafe { slice::from_raw_parts(result_ptr, params_len*8) };
    let mut ciphertext_vec: Vec<u8> = vec![0; params_len*8];
    let ciphertext_slice = &mut ciphertext_vec[..];
    let mut mac_array: [u8; 16] = [0; 16];
    let e_result = rsgx_rijndael128GCM_encrypt(key,
                                               result_slice,
                                               iv,
                                               &aad_array,
                                               ciphertext_slice,
                                               &mut mac_array);
    unsafe{
        ptr::copy_nonoverlapping(ciphertext_slice.as_ptr(),
                                 gradient,
                                 params_len*8);
    }
    *gradient_mac = mac_array;
    mem::forget(params_slice);
    mem::forget(inputs_slice);
    mem::forget(targets_slice);
    sgx_status_t::SGX_SUCCESS
}

#[no_mangle]
pub extern "C"
fn update_model(model: *mut u8,
                gradient: *const u8,
                model_len: usize,
                alpha: f64,
                key: &[u8;16],
                iv: &[u8;12],
                model_mac: &mut [u8;16],
                gradient_mac: &[u8;16])
                -> sgx_status_t {

    // println!("Updating Model...");

    // println!("[++] Decrypting Model...");
    let e_model_slice = unsafe { slice::from_raw_parts(model, model_len*8) };
    let mut model_vec: Vec<f64> = vec![0.0;model_len];
    let mut model_ptr = unsafe {
        mem::transmute::<*const f64, *mut u8>(model_vec[..].as_ptr())
    };
    let mut model_slice = unsafe { Vec::from_raw_parts(model_ptr, model_len*8, model_len*8)};
    let aad_array: [u8; 0] = [0; 0];
    let model_result = rsgx_rijndael128GCM_decrypt(key,
                                                    &e_model_slice,
                                                    iv,
                                                    &aad_array,
                                                    model_mac,
                                                    &mut model_slice);
    // println!("model before updated: {:?}", model_vec);

    // println!("[++] Decrypting Gradient...");
    let e_gradient_slice = unsafe { slice::from_raw_parts(gradient, model_len*8) };
    let mut gradient_vec: Vec<f64> = vec![0.0;model_len];
    let mut gradient_ptr = unsafe {
        mem::transmute::<*const f64, *mut u8>(gradient_vec[..].as_ptr())
    };
    let mut gradient_slice = unsafe { Vec::from_raw_parts(gradient_ptr, model_len*8, model_len*8)};
    let aad_array: [u8; 0] = [0; 0];
    let gradient_result = rsgx_rijndael128GCM_decrypt(key,
                                                      &e_gradient_slice,
                                                      iv,
                                                      &aad_array,
                                                      gradient_mac,
                                                      &mut gradient_slice);
    // println!("gradient used to update the model:{:?}", gradient_vec);

    // println!("[++] Updating Model...");
    let mut result = Vector::new(model_vec) - Vector::new(gradient_vec) * alpha;
    // println!("updated_model: {:?}", result.data());
    /*for i in 0..20 {
       println!("{}", e_gradient_slice[i]);
    }*/
    // println!("{:?}", result.data());

    // println!("[++] Encrypting Data...");
    let result_ptr = unsafe{
        mem::transmute::<*const f64, *const u8>(result.data().as_ptr())
    };
    let result_slice = unsafe { slice::from_raw_parts(result_ptr, model_len*8) };
    let mut ciphertext_vec: Vec<u8> = vec![0; model_len*8];
    let ciphertext_slice = &mut ciphertext_vec[..];
    let mut mac_array: [u8; 16] = [0; 16];
    let e_result = rsgx_rijndael128GCM_encrypt(key,
                                               result_slice,
                                               iv,
                                               &aad_array,
                                               ciphertext_slice,
                                               &mut mac_array);
    unsafe{
        ptr::copy_nonoverlapping(ciphertext_slice.as_ptr(),
                                 model,
                                 model_len*8);
    }
    *model_mac = mac_array;
    mem::forget(model_slice);
    mem::forget(gradient_slice);
    sgx_status_t::SGX_SUCCESS
}

#[no_mangle]
pub extern "C"
fn sgx_predict(model: *const f64, 
           model_len: usize, 
           samples: *const f64, 
           samples_len: usize, 
           prediction: *mut f64, 
           prediction_len: usize) -> sgx_status_t {
    // println!("I am in the enclave");
    let model_slice = unsafe { slice::from_raw_parts(model, model_len)};
    let model_vec = &Vector::new(model_slice.to_vec());
    /* for i in 0..20 {
        println!("{}", model_slice[i]);
    }*/
    let samples_slice = unsafe { slice::from_raw_parts(samples, samples_len)};
    /*for i in 0..20 {
        println!("{}", samples_slice[i]);
    }*/
    let samples_mat = &Matrix::new(prediction_len, model_len, samples_slice);
    let mut result = (samples_mat * model_vec).apply(&Sigmoid::func);
    /* for i in 0..20 {
        println!("#{}", result[i]);
    }*/
    unsafe{
        ptr::copy_nonoverlapping(result.mut_data().as_ptr(),
                                 prediction,
                                 prediction_len);
    };
    sgx_status_t::SGX_SUCCESS
}
