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

    fn add_normal_noise (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, std_dev: f64, in_ptr: *const u8, len: usize, out_ptr: *mut u8, key: &[u8;16], iv: &[u8;12], in_mac: &[u8;16], out_mac: &mut [u8;16]) -> sgx_status_t;

    fn compute_grad (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, in_params: *const f64,  params_len: usize, inputs: *const f64, inputs_len: usize, targets: *const f64, targets_len: usize, out_params: *mut f64) -> sgx_status_t;

    fn update_model (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, model: *const f64, gradient: *const f64, model_len: usize, alpha: f64, updated_model: *mut f64) -> sgx_status_t;

    fn predict (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, model: *const f64, model_len: usize, samples: *const f64, samples_len: usize, prediction: *mut f64, prediction_len: usize);

    fn decrypt_encrypt (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, key: &[u8;16], ciphertext: *const u8, text_len: usize, iv: &[u8;12], mac: &[u8;16]);
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

    /*println!("Test DP-SGD on Iris Dataset...");
    let batch_size = 50;

    println!("Load Iris Data...");
    let dataset = load();
    let raw_samples = dataset.data();
    let ones = Matrix::<f64>::ones(raw_samples.rows(), 1);
    let samples = ones.hcat(raw_samples);

    let samples_ptr = samples.as_ptr();
    let targets = dataset.target();
    let sample_num = samples.rows();
    let feature_num = samples.cols();

    println!("Train Gradient on Batches...");
    let iters = 100;
    let alpha = 0.1;
    // This should be changed to dynamic later. We can use vector.capacity to allocate a large array and use a small portion as slice, or use ones to do this.
    let mut in_params: [f64; 5] = [0.0; 5];
    let in_params_ptr = &in_params as *const f64;
    let updated_model_ptr = unsafe{
        mem::transmute::<&[f64; 5], *mut f64>(&in_params)
    };
    let (batch1, batch2) = samples.split_at(batch_size, Axes::Row);
    //let batch1_ptr = batch1.as_ptr();
    //let batch2_ptr = batch2.as_ptr();
    let mut iter = targets.chunks(batch_size);
    let targets1_ptr = iter.next().unwrap().as_ptr();
    let targets2_ptr = iter.next().unwrap().as_ptr();
    let mut out_params: [f64; 5] = [0.0; 5];
    let out_params_ptr = unsafe{
        mem::transmute::<&[f64; 5], *mut f64>(&out_params)
    };
    let mut model: [f64; 5] = [0.0; 5];
    let model_ptr = &model as *const f64;

    println!("[+] Starting aes-gcm-128 encrypt calculation");
    let batch1_ptr = unsafe {
        mem::transmute::<*const f64, *const u8>(batch1.as_ptr())
    };
    let batch2_ptr = unsafe {
        mem::transmute::<*const f64, *const u8>(batch2.as_ptr())
    };
    let aes_gcm_key: [u8;16] = [0;16];
    let aes_gcm_iv: [u8;12] = [0;12];
    let mut cipher1: [u8;2000] = [0;2000];
    let mut cipher2: [u8;2000] = [0;2000];
    let cipher1_ptr = unsafe{
        mem::transmute::<&[u8;2000], *mut u8>(&cipher1)
    };
    let cipher2_ptr = unsafe{
        mem::transmute::<&[u8;2000], *mut u8>(&cipher2)
    };
    let mut aes_gcm_mac: [u8;16] = [0;16];

    println!("[+] aes-gcm-128 args prepared!");
    let sgx_ret = unsafe{
        aes_gcm_128_encrypt(enclave.geteid(),
                            &mut retval,
                            &aes_gcm_key,
                            batch1_ptr,
                            2000,
                            &aes_gcm_iv,
                            cipher1_ptr,
                            &mut aes_gcm_mac)
    };
    for i in 0..20 {
        println!("{}", cipher1[i]);
    }
    for i in 0..16 {
        println!("{}", aes_gcm_mac[i]);
    }
    println!("[+] aes-gcm-128 returned from enclave!");
    let sgx_ret = unsafe{
        decrypt_encrypt(enclave.geteid(),
                &mut retval,
                &aes_gcm_key,
                cipher1_ptr,
                250,
                &aes_gcm_iv,
                &aes_gcm_mac)
    };
    println!("Return from decrypt_encrypt");
    */

    /*for _ in 0..iters/2 {
        for batch_iter in 0..2 {
            let mut batch_ptr = batch2_ptr;
            let mut target_ptr = targets2_ptr;
            if batch_iter == 0 {
                batch_ptr = batch1_ptr;
                target_ptr = targets1_ptr;
            }
            let sgx_ret = unsafe{
                compute_grad(enclave.geteid(),
                             &mut retval,
                             in_params_ptr,
                             feature_num,
                             batch_ptr,
                             feature_num*batch_size,
                             target_ptr,
                             batch_size,
                             out_params_ptr)
            };
            println!("{:?}", out_params);
            model.clone_from(&in_params);
            let sgx_ret = unsafe{
                update_model(enclave.geteid(),
                             &mut retval,
                             model_ptr,
                             out_params_ptr,
                             feature_num,
                             alpha,
                             updated_model_ptr)
            };
            println!("{:?}", in_params);
        }
    }

    let mut result: [f64; 100] = [0.0; 100];
    let result_ptr = unsafe{
        mem::transmute::<&[f64; 100], *mut f64>(&result)
    };
    let sgx_ret = unsafe{
        predict(enclave.geteid(),
                &mut retval,
                model_ptr,
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
    */

    println!("Test Add Normal Noise...");
    println!("Encrypting...");
    let in_vec: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    let in_ptr = unsafe {
        mem::transmute::<&[f64; 5], *const u8>(&in_vec)
    };
    let aes_gcm_key: [u8;16] = [0; 16];
    let aes_gcm_iv: [u8;12] = [0; 12];
    let mut aes_gcm_ciphertext: [u8;40] = [0;40];
    let ciphertext_ptr = unsafe{
        mem::transmute::<&[u8;40], *mut u8>(&aes_gcm_ciphertext)
    };
    let mut aes_gcm_mac: [u8;16] = [0;16];
    let sgx_ret = unsafe{
        aes_gcm_128_encrypt(enclave.geteid(),
                            &mut retval,
                            &aes_gcm_key,
                            in_ptr,
                            40,
                            &aes_gcm_iv,
                            ciphertext_ptr,
                            &mut aes_gcm_mac)
    };
    println!("Encrypted");
    println!("Adding Noise...");
    let mut out_vec: [f64; 5] = [0.0; 5];
    let out_ptr = unsafe{
        mem::transmute::<&[f64; 5], *mut u8>(&out_vec)
    };
    let mut out_mac: [u8;16] = [0;16];
    let sgx_ret = unsafe{
        add_normal_noise(enclave.geteid(),
                         &mut retval,
                         1.0, 
                         ciphertext_ptr, 
                         5, 
                         out_ptr,
                         &aes_gcm_key,
                         &aes_gcm_iv,
                         &aes_gcm_mac,
                         &mut out_mac)
    };
    println!("{:?}", out_vec);
    println!("Noise added");
    println!("Decrypting...");
    let mut decrypted_vec: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    let decrypted_ptr = unsafe {
        mem::transmute::<&[f64; 5], *mut u8>(&decrypted_vec)
    };
    let sgx_ret = unsafe{
        aes_gcm_128_decrypt(enclave.geteid(),
                            &mut retval,
                            &aes_gcm_key,
                            out_ptr,
                            40,
                            &aes_gcm_iv,
                            &out_mac,
                            decrypted_ptr)
    };
    println!("{:?}", decrypted_vec);
    println!("Decrypted");

    /*println!("Test Compute Gradient");
    let in_params: [f64; 30] = [0.0; 30];
    let inputs: [f64; 900] = [0.0; 900];
    let targets: [f64; 30] = [0.0; 30];
    let mut out_params: [f64; 30] = [0.0; 30];
    let in_params_ptr = &in_params as *const f64;
    let inputs_ptr = &inputs as *const f64;
    let targets_ptr = &inputs as *const f64;
    let out_params_ptr = unsafe{
        mem::transmute::<&[f64; 30], *mut f64>(&out_params)
    };
    let sgx_ret = unsafe{
        compute_grad(enclave.geteid(),
                     &mut retval,
                     in_params_ptr, 
                     30, 
                     inputs_ptr, 
                     900, 
                     targets_ptr, 
                     30, 
                     out_params_ptr)
    };
    println!("{:?}", out_params);*/

    /*println!("Test Update Model");
    let model: [f64; 30] = [1.0; 30];
    let gradient: [f64; 30] = [0.5; 30];
    let mut updated_model: [f64; 30] = [10.0; 30];
    let model_ptr = &model as *const f64;
    let gradient_ptr = &gradient as *const f64;
    let updated_model_ptr = unsafe{
        mem::transmute::<&[f64; 30], *mut f64>(&updated_model)
    };
    let sgx_ret = unsafe{
        update_model(enclave.geteid(),
                     &mut retval,
                     model_ptr,
                     gradient_ptr,
                     30,
                     updated_model_ptr)
    };
    println!("{:?}", updated_model);*/

    /*println!("Testing Crypto Primitives...");

    let str: [u8; 3] = [97, 98, 99];
    let str_ptr = &str as *const u8;
    let mut output_hash: [u8; 32] = [0; 32];
    println!("[+] Expected SHA256 hash: {}", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    let sgx_ret = unsafe {
        calc_sha256(enclave.geteid(),
                    &mut retval,
                    str_ptr,
                    3,
                    &mut output_hash)
    };
    println!("{:?}", output_hash);

    println!("[+] Starting aes-gcm-128 encrypt calculation");
    let aes_gcm_plaintext: [u8;16] = [0; 16];
    let plaintext_ptr = &aes_gcm_plaintext as *const u8;
    let aes_gcm_key: [u8;16] = [0; 16];
    let aes_gcm_iv: [u8;12] = [0; 12];
    let mut aes_gcm_ciphertext: [u8;16] = [0;16];
    let ciphertext_ptr = unsafe{
        mem::transmute::<&[u8;16], *mut u8>(&aes_gcm_ciphertext)
    };
    let mut aes_gcm_mac: [u8;16] = [0;16];
    println!("[+] aes-gcm-128 args prepared!");
    println!("[+] aes-gcm-128 expected ciphertext: {}", "0388dace60b6a392f328c2b971b2fe78");
    let sgx_ret = unsafe{
        aes_gcm_128_encrypt(enclave.geteid(),
                            &mut retval,
                            &aes_gcm_key,
                            plaintext_ptr,
                            16,
                            &aes_gcm_iv,
                            ciphertext_ptr,
                            &mut aes_gcm_mac);
    };
    println!("[+] aes-gcm-128 returned from enclave!");

    println!("{:?}", aes_gcm_ciphertext);*/

    /*let result = unsafe {
        sample_main(enclave.geteid(),
                    &mut retval)
    };

    match result {
        sgx_status_t::SGX_SUCCESS => {},
        _ => {
            println!("[-] ECALL Enclave Failed {}!", result.as_str());
            return;
        }
    }

    println!("[+] say_something success...");*/
    
    enclave.destroy();
}
