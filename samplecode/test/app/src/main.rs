// Copyright (C) 2017-2018 Baidu, Inc. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in
//    the documentation and/or other materials provided with the
//    distribution.
//  * Neither the name of Baidu, Inc., nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

extern crate sgx_types;
extern crate sgx_urts;
extern crate dirs;
#[macro_use]
extern crate rulinalg;

use sgx_types::*;
use sgx_urts::SgxEnclave;

use std::io::{Read, Write};
use std::fs;
use std::path;
use std::mem;
use std::fmt::Debug;
use std::vec::Vec;

use rulinalg::matrix::{Axes, Matrix, MatrixSlice, MatrixSliceMut, BaseMatrix, BaseMatrixMut};
use rulinalg::vector::Vector;
use rulinalg::norm;

/// Dataset container
#[derive(Clone, Debug)]
pub struct Dataset<D, T> where D: Clone + Debug, T: Clone + Debug {

    data: D,
    target: T
}

impl<D, T> Dataset<D, T> where D: Clone + Debug, T: Clone + Debug {

    /// Returns explanatory variable (features)
    pub fn data(&self) -> &D {
        &self.data
    }

    /// Returns objective variable (target)
    pub fn target(&self) -> &T {
        &self.target
    }
}

pub fn load() -> Dataset<Matrix<f64>, Vec<f64>> {
    let data: Matrix<f64> = matrix![5.1, 3.5, 1.4, 0.2;
                                    4.9, 3.0, 1.4, 0.2;
                                    4.7, 3.2, 1.3, 0.2;
                                    4.6, 3.1, 1.5, 0.2;
                                    5.0, 3.6, 1.4, 0.2;
                                    5.4, 3.9, 1.7, 0.4;
                                    4.6, 3.4, 1.4, 0.3;
                                    5.0, 3.4, 1.5, 0.2;
                                    4.4, 2.9, 1.4, 0.2;
                                    4.9, 3.1, 1.5, 0.1;
                                    5.4, 3.7, 1.5, 0.2;
                                    4.8, 3.4, 1.6, 0.2;
                                    4.8, 3.0, 1.4, 0.1;
                                    4.3, 3.0, 1.1, 0.1;
                                    5.8, 4.0, 1.2, 0.2;
                                    5.7, 4.4, 1.5, 0.4;
                                    5.4, 3.9, 1.3, 0.4;
                                    5.1, 3.5, 1.4, 0.3;
                                    5.7, 3.8, 1.7, 0.3;
                                    5.1, 3.8, 1.5, 0.3;
                                    5.4, 3.4, 1.7, 0.2;
                                    5.1, 3.7, 1.5, 0.4;
                                    4.6, 3.6, 1.0, 0.2;
                                    5.1, 3.3, 1.7, 0.5;
                                    4.8, 3.4, 1.9, 0.2;
                                    5.0, 3.0, 1.6, 0.2;
                                    5.0, 3.4, 1.6, 0.4;
                                    5.2, 3.5, 1.5, 0.2;
                                    5.2, 3.4, 1.4, 0.2;
                                    4.7, 3.2, 1.6, 0.2;
                                    4.8, 3.1, 1.6, 0.2;
                                    5.4, 3.4, 1.5, 0.4;
                                    5.2, 4.1, 1.5, 0.1;
                                    5.5, 4.2, 1.4, 0.2;
                                    4.9, 3.1, 1.5, 0.1;
                                    5.0, 3.2, 1.2, 0.2;
                                    5.5, 3.5, 1.3, 0.2;
                                    4.9, 3.1, 1.5, 0.1;
                                    4.4, 3.0, 1.3, 0.2;
                                    5.1, 3.4, 1.5, 0.2;
                                    5.0, 3.5, 1.3, 0.3;
                                    4.5, 2.3, 1.3, 0.3;
                                    4.4, 3.2, 1.3, 0.2;
                                    5.0, 3.5, 1.6, 0.6;
                                    5.1, 3.8, 1.9, 0.4;
                                    4.8, 3.0, 1.4, 0.3;
                                    5.1, 3.8, 1.6, 0.2;
                                    4.6, 3.2, 1.4, 0.2;
                                    5.3, 3.7, 1.5, 0.2;
                                    5.0, 3.3, 1.4, 0.2;
                                    7.0, 3.2, 4.7, 1.4;
                                    6.4, 3.2, 4.5, 1.5;
                                    6.9, 3.1, 4.9, 1.5;
                                    5.5, 2.3, 4.0, 1.3;
                                    6.5, 2.8, 4.6, 1.5;
                                    5.7, 2.8, 4.5, 1.3;
                                    6.3, 3.3, 4.7, 1.6;
                                    4.9, 2.4, 3.3, 1.0;
                                    6.6, 2.9, 4.6, 1.3;
                                    5.2, 2.7, 3.9, 1.4;
                                    5.0, 2.0, 3.5, 1.0;
                                    5.9, 3.0, 4.2, 1.5;
                                    6.0, 2.2, 4.0, 1.0;
                                    6.1, 2.9, 4.7, 1.4;
                                    5.6, 2.9, 3.6, 1.3;
                                    6.7, 3.1, 4.4, 1.4;
                                    5.6, 3.0, 4.5, 1.5;
                                    5.8, 2.7, 4.1, 1.0;
                                    6.2, 2.2, 4.5, 1.5;
                                    5.6, 2.5, 3.9, 1.1;
                                    5.9, 3.2, 4.8, 1.8;
                                    6.1, 2.8, 4.0, 1.3;
                                    6.3, 2.5, 4.9, 1.5;
                                    6.1, 2.8, 4.7, 1.2;
                                    6.4, 2.9, 4.3, 1.3;
                                    6.6, 3.0, 4.4, 1.4;
                                    6.8, 2.8, 4.8, 1.4;
                                    6.7, 3.0, 5.0, 1.7;
                                    6.0, 2.9, 4.5, 1.5;
                                    5.7, 2.6, 3.5, 1.0;
                                    5.5, 2.4, 3.8, 1.1;
                                    5.5, 2.4, 3.7, 1.0;
                                    5.8, 2.7, 3.9, 1.2;
                                    6.0, 2.7, 5.1, 1.6;
                                    5.4, 3.0, 4.5, 1.5;
                                    6.0, 3.4, 4.5, 1.6;
                                    6.7, 3.1, 4.7, 1.5;
                                    6.3, 2.3, 4.4, 1.3;
                                    5.6, 3.0, 4.1, 1.3;
                                    5.5, 2.5, 4.0, 1.3;
                                    5.5, 2.6, 4.4, 1.2;
                                    6.1, 3.0, 4.6, 1.4;
                                    5.8, 2.6, 4.0, 1.2;
                                    5.0, 2.3, 3.3, 1.0;
                                    5.6, 2.7, 4.2, 1.3;
                                    5.7, 3.0, 4.2, 1.2;
                                    5.7, 2.9, 4.2, 1.3;
                                    6.2, 2.9, 4.3, 1.3;
                                    5.1, 2.5, 3.0, 1.1;
                                    5.7, 2.8, 4.1, 1.3];

    let target: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    Dataset{ data: data,
             target: target }
}

static ENCLAVE_FILE: &'static str = "enclave.signed.so";
static ENCLAVE_TOKEN: &'static str = "enclave.token";

extern {
    fn calc_sha256 (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, input_str: *const u8, some_len: usize, hash: &mut [u8;32]) -> sgx_status_t;

    fn aes_gcm_128_encrypt (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, key: &[u8;16], plaintext: *const u8, text_len: usize, iv: &[u8;12], ciphertext: *mut u8, mac: &mut [u8;16]) -> sgx_status_t;

    fn aes_gcm_128_decrypt (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, key: &[u8;16], ciphertext: *const u8, text_len: usize, iv: &[u8;12], mac: &[u8;16], plaintext: *mut u8) -> sgx_status_t;

    fn aes_cmac(eid: sgx_enclave_id_t, retval: *mut sgx_status_t, text: *const u8, text_len: usize, key: &[u8;16], cmac: &mut [u8;16]) -> sgx_status_t;

    fn ras_key(eid: sgx_enclave_id_t, retval: *mut sgx_status_t, text: *const u8, text_len: usize) -> sgx_status_t;

    fn sample_main (eid: sgx_enclave_id_t, retval: *mut sgx_status_t) -> sgx_status_t;

    fn add_normal_noise (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, std_dev: f64, gradient: *mut u8, len: usize, key: &[u8;16], iv: &[u8;12], gradient_mac: &mut [u8;16]) -> sgx_status_t;

    fn compute_grad (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, params: *const u8,  params_len: usize, inputs: *const u8, inputs_len: usize, targets: *const u8, targets_len: usize, gradient: *mut u8, key: &[u8;16], iv: &[u8;12], model_mac: &[u8;16], inputs_mac: &[u8;16], targets_mac: &[u8;16], gradient_mac: &mut [u8;16]) -> sgx_status_t;

    fn update_model (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, model: *mut u8, gradient: *const u8, model_len: usize, alpha: f64, key: &[u8;16], iv: &[u8;12], model_mac: &mut [u8;16], gradient_mac: &[u8;16]) -> sgx_status_t;

    fn predict (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, model: *const f64, model_len: usize, samples: *const f64, samples_len: usize, prediction: *mut f64, prediction_len: usize);
}

fn init_enclave() -> SgxResult<SgxEnclave> {
    
    let mut launch_token: sgx_launch_token_t = [0; 1024];
    let mut launch_token_updated: i32 = 0;
    // Step 1: try to retrieve the launch token saved by last transaction 
    //         if there is no token, then create a new one.
    // 
    // try to get the token saved in $HOME */
    let mut home_dir = path::PathBuf::new();
    let use_token = match dirs::home_dir() {
        Some(path) => {
            println!("[+] Home dir is {}", path.display());
            home_dir = path;
            true
        },
        None => {
            println!("[-] Cannot get home dir");
            false
        }
    };

    let token_file: path::PathBuf = home_dir.join(ENCLAVE_TOKEN);;
    if use_token == true {
        match fs::File::open(&token_file) {
            Err(_) => {
                println!("[-] Open token file {} error! Will create one.", token_file.as_path().to_str().unwrap());
            },
            Ok(mut f) => {
                println!("[+] Open token file success! ");
                match f.read(&mut launch_token) {
                    Ok(1024) => {
                        println!("[+] Token file valid!");
                    },
                    _ => println!("[+] Token file invalid, will create new token file"),
                }
            }
        }
    }

    // Step 2: call sgx_create_enclave to initialize an enclave instance
    // Debug Support: set 2nd parameter to 1 
    let debug = 1;
    let mut misc_attr = sgx_misc_attribute_t {secs_attr: sgx_attributes_t { flags:0, xfrm:0}, misc_select:0};
    let enclave = try!(SgxEnclave::create(ENCLAVE_FILE, 
                                          debug, 
                                          &mut launch_token,
                                          &mut launch_token_updated,
                                          &mut misc_attr));
    
    // Step 3: save the launch token if it is updated 
    if use_token == true && launch_token_updated != 0 {
        // reopen the file with write capablity 
        match fs::File::create(&token_file) {
            Ok(mut f) => {
                match f.write_all(&launch_token) {
                    Ok(()) => println!("[+] Saved updated launch token!"),
                    Err(_) => println!("[-] Failed to save updated launch token!"),
                }
            },
            Err(_) => {
                println!("[-] Failed to save updated enclave token, but doesn't matter");
            },
        }
    }

    Ok(enclave)
}

fn main() { 
    let enclave = match init_enclave() {
        Ok(r) => {
            println!("[+] Init Enclave Successful {}!", r.geteid());
            r
        },
        Err(x) => {
            println!("[-] Init Enclave Failed {}!", x.as_str());
            return;
        },
    };

    let mut retval = sgx_status_t::SGX_SUCCESS; 

    println!("Test DP-SGD on Iris Dataset...");

    println!("Setting Meta Data...");
    let batch_size = 50;
    let alpha = 0.1;
    let iters = 100;
    // We do not have regularization term here so this is 1 for LR.
    let L = 1.0;
    // We now hard-code the eps and delta here, will change this to the arguments later.
    let eps = 1.0;
    let delta = 0.00001;

    println!("Loading Iris Data...");
    let dataset = load();
    let raw_samples = dataset.data();
    let ones = Matrix::<f64>::ones(raw_samples.rows(), 1);
    let samples = ones.hcat(raw_samples);
    let samples_ptr = samples.as_ptr();
    let targets = dataset.target();
    let sample_num = samples.rows();
    let feature_num = samples.cols();
    let (batch1, batch2) = samples.split_at(batch_size, Axes::Row);
    let mut iter = targets.chunks(batch_size);
    let targets1_ptr = unsafe{
        mem::transmute::<*const f64, *mut u8>(iter.next().unwrap().as_ptr())
    };
    let targets2_ptr = unsafe{
        mem::transmute::<*const f64, *mut u8>(iter.next().unwrap().as_ptr())
    };
    let batch1_ptr = unsafe {
        mem::transmute::<*const f64, *const u8>(batch1.as_ptr())
    };
    let batch2_ptr = unsafe {
        mem::transmute::<*const f64, *const u8>(batch2.as_ptr())
    };
    let std_dev = 4.0*L*((iters as f64)*((1.0/delta) as f64).log2()).sqrt()/((sample_num as f64)*eps);

    println!("Preparing Model...");
    let mut model: [f64; 5] = [0.0; 5];
    let model_ptr = unsafe{
        mem::transmute::<&[f64; 5], *mut u8>(&model)
    };
    let mut gradient: [f64; 5] = [0.0; 5];
    let gradient_ptr = unsafe{
        mem::transmute::<&[f64; 5], *mut u8>(&gradient)
    };

    println!("Preparing Encryption Meta Data...");
    let aes_gcm_key: [u8;16] = [0;16];
    let aes_gcm_iv: [u8;12] = [0;12];
    let mut inputs1_mac: [u8;16] = [0;16];
    let mut inputs2_mac: [u8; 16] = [0;16];
    let mut targets1_mac: [u8;16] = [0;16];
    let mut targets2_mac: [u8;16] = [0;16];
    let mut model_mac: [u8;16] = [0;16];
    let mut gradient_mac: [u8;16] = [0;16];

    println!("Encrypting Data...");
    let mut batch1_cipher: [u8;2000] = [0;2000];
    let mut batch2_cipher: [u8;2000] = [0;2000];
    let mut targets1_cipher: [u8;400] = [0;400];
    let mut targets2_cipher: [u8;400] = [0;400];
    let mut model_cipher: [u8;40] = [0;40];
    let batch1_cipher_ptr = unsafe{
        mem::transmute::<&[u8;2000], *mut u8>(&batch1_cipher)
    };
    let batch2_cipher_ptr = unsafe{
        mem::transmute::<&[u8;2000], *mut u8>(&batch2_cipher)
    };
    let targets1_cipher_ptr = unsafe{
        mem::transmute::<&[u8;400], *mut u8>(&targets1_cipher)
    };
    let targets2_cipher_ptr = unsafe{
        mem::transmute::<&[u8;400], *mut u8>(&targets2_cipher)
    };
    let model_cipher_ptr = unsafe{
        mem::transmute::<&[u8;40], *mut u8>(&model_cipher)
    };
    let sgx_ret = unsafe{
        aes_gcm_128_encrypt(enclave.geteid(),
                            &mut retval,
                            &aes_gcm_key,
                            batch1_ptr,
                            2000,
                            &aes_gcm_iv,
                            batch1_cipher_ptr,
                            &mut inputs1_mac)
    };
    let sgx_ret = unsafe{
        aes_gcm_128_encrypt(enclave.geteid(),
                            &mut retval,
                            &aes_gcm_key,
                            batch2_ptr,
                            2000,
                            &aes_gcm_iv,
                            batch2_cipher_ptr,
                            &mut inputs2_mac)
    };
    let sgx_ret = unsafe{
        aes_gcm_128_encrypt(enclave.geteid(),
                            &mut retval,
                            &aes_gcm_key,
                            targets1_ptr,
                            400,
                            &aes_gcm_iv,
                            targets1_cipher_ptr,
                            &mut targets1_mac)
    };
    let sgx_ret = unsafe{
        aes_gcm_128_encrypt(enclave.geteid(),
                            &mut retval,
                            &aes_gcm_key,
                            targets2_ptr,
                            400,
                            &aes_gcm_iv,
                            targets2_cipher_ptr,
                            &mut targets2_mac)
    };
    let sgx_ret = unsafe{
        aes_gcm_128_encrypt(enclave.geteid(),
                            &mut retval,
                            &aes_gcm_key,
                            model_ptr,
                            40,
                            &aes_gcm_iv,
                            model_cipher_ptr,
                            &mut model_mac)
    };


    for _ in 0..iters/2 {
        for batch_iter in 0..2 {
            let mut batch_ptr = batch2_cipher_ptr;
            let mut target_ptr = targets2_cipher_ptr;
            let mut inputs_mac = inputs2_mac;
            let mut targets_mac = targets2_mac;
            if batch_iter == 0 {
                batch_ptr = batch1_cipher_ptr;
                target_ptr = targets1_cipher_ptr;
                inputs_mac = inputs1_mac;
                targets_mac = targets1_mac;
            }
            let sgx_ret = unsafe{
                compute_grad(enclave.geteid(),
                             &mut retval,
                             model_cipher_ptr,
                             feature_num,
                             batch_ptr,
                             feature_num*batch_size,
                             target_ptr,
                             batch_size,
                             gradient_ptr,
                             &aes_gcm_key,
                             &aes_gcm_iv,
                             &model_mac,
                             &inputs_mac,
                             &targets_mac,
                             &mut gradient_mac)
            };
            let sgx_ret = unsafe{
                add_normal_noise(enclave.geteid(),
                                 &mut retval,
                                 std_dev,
                                 gradient_ptr,
                                 feature_num,
                                 &aes_gcm_key,
                                 &aes_gcm_iv,
                                 &mut gradient_mac)
            };
            let sgx_ret = unsafe{
                update_model(enclave.geteid(),
                             &mut retval,
                             model_cipher_ptr,
                             gradient_ptr,
                             feature_num,
                             alpha,
                             &aes_gcm_key,
                             &aes_gcm_iv,
                             &mut model_mac,
                             &gradient_mac)
            };
        }
    }

    let decrypted_model: [f64;5] = [0.0;5];
    let decrypted_model_ptr = unsafe{
        mem::transmute::<&[f64;5], *mut u8>(&decrypted_model)
    };
    let decrypted_model_float_ptr = &decrypted_model as *const f64;
    let sgx_ret = unsafe{
        aes_gcm_128_decrypt(enclave.geteid(),
                            &mut retval,
                            &aes_gcm_key,
                            model_cipher_ptr,
                            40,
                            &aes_gcm_iv,
                            &mut model_mac,
                            decrypted_model_ptr)
    };
    println!("{:?}", decrypted_model);

    let mut result: [f64; 100] = [0.0; 100];
    let result_ptr = unsafe{
        mem::transmute::<&[f64; 100], *mut f64>(&result)
    };
    let sgx_ret = unsafe{
        predict(enclave.geteid(),
                &mut retval,
                decrypted_model_float_ptr,
                feature_num,
                samples_ptr,
                feature_num*100,
                result_ptr,
                100)
    };
    for i in result.into_iter() {
        println!("{}", i);
    }
    let classes = result.into_iter().map(|x|if *x > 0.5 {return 1.0;} else {return 0.0;}).collect::<Vec<_>>();
    let matching = classes.into_iter().zip(targets.into_iter()).filter(|(a, b)| a==*b ).count();
    println!("Correct Number is {}", matching);
 
    enclave.destroy();
}
