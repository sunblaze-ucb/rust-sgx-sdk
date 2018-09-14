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
extern crate csv;

use sgx_types::*;
use sgx_urts::SgxEnclave;

use std::io::{Read, Write};
use std::fs;
use std::fs::File;
use std::path;
use std::mem;
use std::io;
use std::io::prelude::*;
use std::io::Cursor;
use std::fmt::Debug;
use std::vec::Vec;
use std::ptr;

use rulinalg::matrix::{Axes, Matrix, MatrixSlice, MatrixSliceMut, BaseMatrix, BaseMatrixMut};
use rulinalg::vector::Vector;
use rulinalg::norm;

const BATCH_SIZE: usize = 100;
const SAMPLE_COL_NUMBER: usize = 53;
const TARGET_COL_NUMBER: usize = 1;
const MAC_BYTE_NUMBER: usize = 16;
const IV_BYTE_NUMBER: usize = 12;
const TEST_SAMPLE_NUMBER: usize = 8190;
const BATCH_NUMBER: usize = 326;
const SAMPLE_NUMBER: usize = BATCH_SIZE*BATCH_NUMBER;

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

fn read_file(file_path: String, buffer_array: &mut [u8]) {
    let mut f = File::open(file_path).expect("file not found");
    f.read(buffer_array).expect("something went wrong reading the file");
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

static mut batch_cipher: [[u8;BATCH_SIZE*SAMPLE_COL_NUMBER*8];BATCH_NUMBER] = [[0;BATCH_SIZE*SAMPLE_COL_NUMBER*8];BATCH_NUMBER];
static mut targets_cipher: [[u8;BATCH_SIZE*8];BATCH_NUMBER] = [[0;BATCH_SIZE*8];BATCH_NUMBER];
static mut batch_cipher_ptr: [*mut u8;BATCH_NUMBER] = [ptr::null_mut();BATCH_NUMBER];
static mut targets_cipher_ptr: [*mut u8;BATCH_NUMBER] = [ptr::null_mut();BATCH_NUMBER];
static mut model: [f64; SAMPLE_COL_NUMBER] = [0.0; SAMPLE_COL_NUMBER];
static mut gradient: [f64; SAMPLE_COL_NUMBER] = [0.0; SAMPLE_COL_NUMBER];
static mut model_cipher: [u8;SAMPLE_COL_NUMBER*8] = [0;SAMPLE_COL_NUMBER*8];

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

    println!("Setting Meta Data..."); 
    //let sample_num = 200;
    let alpha = 0.1;
    let iters = 10000;
    // We do not have regularization term here so this is 1 for LR.
    let L = 1.0;
    // We now hard-code the eps and delta here, will change this to the arguments later.
    let eps = 1.0;
    let delta = 1.0/((SAMPLE_NUMBER as f64)*(SAMPLE_NUMBER as f64));

    println!("Preparing Model...");
    let std_dev = 4.0*L*((iters as f64)*((1.0/delta) as f64).log2()).sqrt()/((SAMPLE_NUMBER as f64)*eps);
    // let mut model: [f64; SAMPLE_COL_NUMBER] = [0.0; SAMPLE_COL_NUMBER];
    let model_ptr = unsafe{
        mem::transmute::<&[f64; SAMPLE_COL_NUMBER], *mut u8>(&model)
    };
    // let mut gradient: [f64; SAMPLE_COL_NUMBER] = [0.0; SAMPLE_COL_NUMBER];
    let gradient_ptr = unsafe{
        mem::transmute::<&[f64; SAMPLE_COL_NUMBER], *mut u8>(&gradient)
    };

    println!("Preparing Encryption Meta Data...");
    let aes_gcm_key: [u8;16] = [0;16];
    let aes_gcm_iv: [u8;12] = [0;12];
    let mut inputs_mac: [[u8;16];BATCH_NUMBER] = [[0;16];BATCH_NUMBER];
    let mut targets_mac: [[u8;16];BATCH_NUMBER] = [[0;16];BATCH_NUMBER];
    let mut model_mac: [u8;16] = [0;16];
    let mut gradient_mac: [u8;16] = [0;16];

    println!("Loading Data...");
    /*let mut batch_cipher: [[u8;BATCH_SIZE*SAMPLE_COL_NUMBER*8];BATCH_NUMBER] = [[0;BATCH_SIZE*SAMPLE_COL_NUMBER*8];BATCH_NUMBER];
    let mut targets_cipher: [[u8;BATCH_SIZE*8];BATCH_NUMBER] = [[0;BATCH_SIZE*8];BATCH_NUMBER];
    let mut batch_cipher_ptr: [*mut u8;BATCH_NUMBER] = [ptr::null_mut();BATCH_NUMBER];
    let mut targets_cipher_ptr: [*mut u8;BATCH_NUMBER] = [ptr::null_mut();BATCH_NUMBER];*/
    for i in 0..BATCH_NUMBER {
        unsafe {
            batch_cipher_ptr[i] = mem::transmute::<&[u8;BATCH_SIZE*SAMPLE_COL_NUMBER*8], *mut u8>(&batch_cipher[i]);
            targets_cipher_ptr[i] = mem::transmute::<&[u8;BATCH_SIZE*8], *mut u8>(&targets_cipher[i]);
        }
    }
    for i in 0..BATCH_NUMBER {
        let mut sample_file: String = "../datasets/facebook_train_encrypted_sample_".to_owned();
        let mut sample_mac_file: String = "../datasets/facebook_train_mac_sample_".to_owned();
        let mut target_file: String = "../datasets/facebook_train_encrypted_target_".to_owned();
        let mut target_mac_file: String = "../datasets/facebook_train_mac_target_".to_owned();
        let batch_index = i.to_string();
        sample_file.push_str(&batch_index);
        sample_mac_file.push_str(&batch_index);
        target_file.push_str(&batch_index);
        target_mac_file.push_str(&batch_index);
        println!("{}", sample_file);
        println!("{}", sample_mac_file);
        println!("{}", target_file);
        println!("{}", target_mac_file);
        unsafe {
            read_file(sample_file, &mut batch_cipher[i]);
            read_file(sample_mac_file, &mut inputs_mac[i]);
            read_file(target_file, &mut targets_cipher[i]);
            read_file(target_mac_file, &mut targets_mac[i]);
        }
        // println!("mac array {:x?}", &mac_array[..]);
    }

    println!("Encrypting Model...");
    // let mut model_cipher: [u8;SAMPLE_COL_NUMBER*8] = [0;SAMPLE_COL_NUMBER*8];
    let model_cipher_ptr = unsafe{
        mem::transmute::<&[u8;SAMPLE_COL_NUMBER*8], *mut u8>(&model_cipher)
    };
    let sgx_ret = unsafe{
        aes_gcm_128_encrypt(enclave.geteid(),
                            &mut retval,
                            &aes_gcm_key,
                            model_ptr,
                            SAMPLE_COL_NUMBER*8,
                            &aes_gcm_iv,
                            model_cipher_ptr,
                            &mut model_mac)
    };

    println!("Training...");
    for i in 0..iters {
        let batch_index = i%BATCH_NUMBER;
        let sgx_ret = unsafe{
            compute_grad(enclave.geteid(),
                         &mut retval,
                         model_cipher_ptr,
                         SAMPLE_COL_NUMBER,
                         batch_cipher_ptr[batch_index],
                         SAMPLE_COL_NUMBER*BATCH_SIZE,
                         targets_cipher_ptr[batch_index],
                         BATCH_SIZE,
                         gradient_ptr,
                         &aes_gcm_key,
                         &aes_gcm_iv,
                         &model_mac,
                         &inputs_mac[batch_index],
                         &targets_mac[batch_index],
                         &mut gradient_mac)
        };
        let sgx_ret = unsafe{
            add_normal_noise(enclave.geteid(),
                             &mut retval,
                             std_dev,
                             gradient_ptr,
                             SAMPLE_COL_NUMBER,
                             &aes_gcm_key,
                             &aes_gcm_iv,
                             &mut gradient_mac)
        };
        let sgx_ret = unsafe{
            update_model(enclave.geteid(),
                         &mut retval,
                         model_cipher_ptr,
                         gradient_ptr,
                         SAMPLE_COL_NUMBER,
                         alpha,
                         &aes_gcm_key,
                         &aes_gcm_iv,
                         &mut model_mac,
                         &gradient_mac)
        };
    }

    let decrypted_model: [f64;SAMPLE_COL_NUMBER] = [0.0;SAMPLE_COL_NUMBER];
    let decrypted_model_ptr = unsafe{
        mem::transmute::<&[f64;SAMPLE_COL_NUMBER], *mut u8>(&decrypted_model)
    };
    let decrypted_model_float_ptr = &decrypted_model as *const f64;
    let sgx_ret = unsafe{
        aes_gcm_128_decrypt(enclave.geteid(),
                            &mut retval,
                            &aes_gcm_key,
                            model_cipher_ptr,
                            SAMPLE_COL_NUMBER*8,
                            &aes_gcm_iv,
                            &mut model_mac,
                            decrypted_model_ptr)
    };
    for i in 0..SAMPLE_COL_NUMBER {
        println!("{}", decrypted_model[i]);
    }
    // println!("{:?}", decrypted_model);

    let mut test_samples: [f64;TEST_SAMPLE_NUMBER*SAMPLE_COL_NUMBER] = [0.0;TEST_SAMPLE_NUMBER*SAMPLE_COL_NUMBER];
    let mut test_targets: [f64;TEST_SAMPLE_NUMBER] = [0.0;TEST_SAMPLE_NUMBER];
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path("../datasets/facebook_test.csv").unwrap();
    let mut test_samples_cnt = 0;
    let mut test_targets_cnt = 0;
    for result in rdr.records() {
        let record = result.unwrap();
        test_targets[test_targets_cnt] = record.get(0).unwrap().parse::<f64>().unwrap();
        test_targets_cnt = test_targets_cnt + 1;
        for i in 1..SAMPLE_COL_NUMBER+1 {
            test_samples[test_samples_cnt] = record.get(i).unwrap().parse::<f64>().unwrap();
            test_samples_cnt = test_samples_cnt + 1;
        }
    }
    println!("{}", test_targets_cnt);
    println!("{}", test_samples_cnt);
    let test_samples_ptr = unsafe {
        mem::transmute::<&[f64;TEST_SAMPLE_NUMBER*SAMPLE_COL_NUMBER], *const f64>(&test_samples)
    };
    let mut result: [f64; TEST_SAMPLE_NUMBER] = [0.0; TEST_SAMPLE_NUMBER];
    let result_ptr = unsafe{
        mem::transmute::<&[f64; TEST_SAMPLE_NUMBER], *mut f64>(&result)
    };
    let sgx_ret = unsafe{
        predict(enclave.geteid(),
                &mut retval,
                decrypted_model_float_ptr,
                SAMPLE_COL_NUMBER,
                test_samples_ptr,
                SAMPLE_COL_NUMBER*TEST_SAMPLE_NUMBER,
                result_ptr,
                TEST_SAMPLE_NUMBER)
    };
    for i in result.iter() {
        println!("{}", i);
    }
    let classes = result.into_iter().map(|x|if *x > 0.5 {return 1.0;} else {return 0.0;}).collect::<Vec<_>>();
    let matching = classes.into_iter().zip(test_targets.into_iter()).filter(|(a, b)| a==*b ).count();
    println!("Correct Number is {}", matching);
 
    enclave.destroy();
}
