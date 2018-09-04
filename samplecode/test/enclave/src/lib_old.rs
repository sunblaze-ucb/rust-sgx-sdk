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

#![crate_name = "machinelearningsampleenclave"]
#![crate_type = "staticlib"]

#![cfg_attr(not(target_env = "sgx"), no_std)]
#![cfg_attr(target_env = "sgx", feature(rustc_private))]

extern crate sgx_types;
extern crate sgx_tcrypto;
extern crate sgx_trts;
#[cfg(not(target_env = "sgx"))]
#[macro_use]
extern crate sgx_tstd as std;

use sgx_types::*;
use sgx_tcrypto::*;
use sgx_trts::memeq::ConsttimeMemEq;
use std::vec::Vec;
use std::slice;
use std::ptr;
use std::mem;

extern crate rusty_machine;
extern crate sgx_rand as rand;

use rusty_machine::linalg::{Matrix, BaseMatrix, Vector};
use rusty_machine::learning::SupModel;
use rusty_machine::learning::logistic_reg::LogisticRegressor;
use rusty_machine::learning::dp_logistic_reg::DPLogisticRegressor;
use rusty_machine::datasets::iris;

use rand::thread_rng;
use rand::distributions::IndependentSample;
use rand::distributions::normal::Normal;

/// A Ecall function takes a string and output its SHA256 digest.
///
/// # Parameters
///
/// **input_str**
///
/// A raw pointer to the string to be calculated.
///
/// **some_len**
///
/// An unsigned int indicates the length of input string
///
/// **hash**
///
/// A const reference to [u8;32] array, which is the destination buffer which contains the SHA256 digest, caller allocated.
///
/// # Return value
///
/// **SGX_SUCCESS** on success. The SHA256 digest is stored in the destination buffer.
///
/// # Requirements
///
/// Caller allocates the input buffer and output buffer.
///
/// # Errors
///
/// **SGX_ERROR_INVALID_PARAMETER**
///
/// Indicates the parameter is invalid
#[no_mangle]
pub extern "C" fn calc_sha256(input_str: *const u8,
                              some_len: usize,
                              hash: &mut [u8;32]) -> sgx_status_t {

    println!("calc_sha256 invoked!");

    // First, build a slice for input_str
    let input_slice = unsafe { slice::from_raw_parts(input_str, some_len) };

    // slice::from_raw_parts does not guarantee the length, we need a check
    if input_slice.len() != some_len {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    println!("Input string len = {}, input len = {}", input_slice.len(), some_len);

    // Second, convert the vector to a slice and calculate its SHA256
    let result = rsgx_sha256_slice(&input_slice);

    // Third, copy back the result
    match result {
        Ok(output_hash) => *hash = output_hash,
        Err(x) => return x
    }

    sgx_status_t::SGX_SUCCESS
}

/// An AES-GCM-128 encrypt function sample.
///
/// # Parameters
///
/// **key**
///
/// Key used in AES encryption, typed as &[u8;16].
///
/// **plaintext**
///
/// Plain text to be encrypted.
///
/// **text_len**
///
/// Length of plain text, unsigned int.
///
/// **iv**
///
/// Initialization vector of AES encryption, typed as &[u8;12].
///
/// **ciphertext**
///
/// A pointer to destination ciphertext buffer.
///
/// **mac**
///
/// A pointer to destination mac buffer, typed as &mut [u8;16].
///
/// # Return value
///
/// **SGX_SUCCESS** on success
///
/// # Errors
///
/// **SGX_ERROR_INVALID_PARAMETER** Indicates the parameter is invalid.
///
/// **SGX_ERROR_UNEXPECTED** Indicates that encryption failed.
///
/// # Requirements
///
/// The caller should allocate the ciphertext buffer. This buffer should be
/// at least same length as plaintext buffer. The caller should allocate the
/// mac buffer, at least 16 bytes.
#[no_mangle]
pub extern "C" fn aes_gcm_128_encrypt(key: &[u8;16],
                                      plaintext: *const u8,
                                      text_len: usize,
                                      iv: &[u8;12],
                                      ciphertext: *mut u8,
                                      mac: &mut [u8;16]) -> sgx_status_t {

    println!("aes_gcm_128_encrypt invoked!");

    // First, we need slices for input
    let plaintext_slice = unsafe { slice::from_raw_parts(plaintext, text_len) };

    // Here we need to initiate the ciphertext buffer, though nothing in it.
    // Thus show the length of ciphertext buffer is equal to plaintext buffer.
    // If not, the length of ciphertext_vec will be 0, which leads to argument
    // illegal.
    let mut ciphertext_vec: Vec<u8> = vec![0; text_len];

    // Second, for data with known length, we use array with fixed length.
    // Here we cannot use slice::from_raw_parts because it provides &[u8]
    // instead of &[u8,16].
    let aad_array: [u8; 0] = [0; 0];
    let mut mac_array: [u8; SGX_AESGCM_MAC_SIZE] = [0; SGX_AESGCM_MAC_SIZE];

    // Always check the length after slice::from_raw_parts
    if plaintext_slice.len() != text_len {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    let ciphertext_slice = &mut ciphertext_vec[..];
    println!("aes_gcm_128_encrypt parameter prepared! {}, {}",
              plaintext_slice.len(),
              ciphertext_slice.len());

    // After everything has been set, call API
    let result = rsgx_rijndael128GCM_encrypt(key,
                                             &plaintext_slice,
                                             iv,
                                             &aad_array,
                                             ciphertext_slice,
                                             &mut mac_array);
    println!("rsgx calling returned!");

    // Match the result and copy result back to normal world.
    match result {
        Err(x) => {
            return x;
        }
        Ok(()) => {
            unsafe{
                ptr::copy_nonoverlapping(ciphertext_slice.as_ptr(),
                                         ciphertext,
                                         text_len);
            }
            *mac = mac_array;
        }
    }

    sgx_status_t::SGX_SUCCESS
}

/// An AES-GCM-128 decrypt function sample.
///
/// # Parameters
///
/// **key**
///
/// Key used in AES encryption, typed as &[u8;16].
///
/// **ciphertext**
///
/// Cipher text to be encrypted.
///
/// **text_len**
///
/// Length of cipher text.
///
/// **iv**
///
/// Initialization vector of AES encryption, typed as &[u8;12].
///
/// **mac**
///
/// A pointer to source mac buffer, typed as &[u8;16].
///
/// **plaintext**
///
/// A pointer to destination plaintext buffer.
///
/// # Return value
///
/// **SGX_SUCCESS** on success
///
/// # Errors
///
/// **SGX_ERROR_INVALID_PARAMETER** Indicates the parameter is invalid.
///
/// **SGX_ERROR_UNEXPECTED** means that decryption failed.
///
/// # Requirements
//
/// The caller should allocate the plaintext buffer. This buffer should be
/// at least same length as ciphertext buffer.
#[no_mangle]
pub extern "C" fn aes_gcm_128_decrypt(key: &[u8;16],
                                      ciphertext: *const u8,
                                      text_len: usize,
                                      iv: &[u8;12],
                                      mac: &[u8;16],
                                      plaintext: *mut u8) -> sgx_status_t {

    println!("aes_gcm_128_decrypt invoked!");

    // First, for data with unknown length, we use vector as builder.
    let ciphertext_slice = unsafe { slice::from_raw_parts(ciphertext, text_len) };
    let mut plaintext_vec: Vec<u8> = vec![0; text_len];

    // Second, for data with known length, we use array with fixed length.
    let aad_array: [u8; 0] = [0; 0];

    if ciphertext_slice.len() != text_len {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    let plaintext_slice = &mut plaintext_vec[..];
    println!("aes_gcm_128_decrypt parameter prepared! {}, {}",
              ciphertext_slice.len(),
              plaintext_slice.len());
    
    // After everything has been set, call API
    let result = rsgx_rijndael128GCM_decrypt(key,
                                             &ciphertext_slice,
                                             iv,
                                             &aad_array,
                                             mac,
                                             plaintext_slice);

    println!("rsgx calling returned!");

    // Match the result and copy result back to normal world.
    match result {
        Err(x) => {
            return x;
        }
        Ok(()) => {
            unsafe {
                ptr::copy_nonoverlapping(plaintext_slice.as_ptr(),
                                         plaintext,
                                         text_len);
            }
        }
    }

    sgx_status_t::SGX_SUCCESS
}

/// A sample aes-cmac function.
///
/// # Parameters
///
/// **text**
///
/// The text message to be calculated.
///
/// **text_len**
///
/// An unsigned int indicate the length of input text message.
///
/// **key**
///
/// The key used in AES-CMAC, 16 bytes sized.
///
/// **cmac**
///
/// The output buffer, at least 16 bytes available.
///
/// # Return value
///
/// **SGX_SUCCESS** on success.
///
/// # Errors
///
/// **SGX_ERROR_INVALID_PARAMETER** indicates invalid input parameters
///
/// # Requirement
///
/// The caller should allocate the output cmac buffer.
#[no_mangle]
pub extern "C" fn aes_cmac(text: *const u8,
                           text_len: usize,
                           key: &[u8;16],
                           cmac: &mut [u8;16]) -> sgx_status_t {

    let text_slice = unsafe { slice::from_raw_parts(text, text_len) };

    if text_slice.len() != text_len {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    let result = rsgx_rijndael128_cmac_slice(key, &text_slice);

    match result {
        Err(x) => return x,
        Ok(m) => *cmac = m
    }

    sgx_status_t::SGX_SUCCESS
}


#[no_mangle]
pub extern "C" fn rsa_key(text: * const u8, text_len: usize) -> sgx_status_t {

    let text_slice = unsafe { slice::from_raw_parts(text, text_len) };

    if text_slice.len() != text_len {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }

    let mod_size: i32 = 256;
    let exp_size: i32 = 4;
    let mut n: Vec<u8> = vec![0_u8; mod_size as usize];
    let mut d: Vec<u8> = vec![0_u8; mod_size as usize];
    let mut e: Vec<u8> = vec![1, 0, 1, 0];
    let mut p: Vec<u8> = vec![0_u8; mod_size as usize / 2];
    let mut q: Vec<u8> = vec![0_u8; mod_size as usize / 2];
    let mut dmp1: Vec<u8> = vec![0_u8; mod_size as usize / 2];
    let mut dmq1: Vec<u8> = vec![0_u8; mod_size as usize / 2];
    let mut iqmp: Vec<u8> = vec![0_u8; mod_size as usize / 2];

    let result = rsgx_create_rsa_key_pair(mod_size, 
                                          exp_size, 
                                          n.as_mut_slice(),
                                          d.as_mut_slice(),
                                          e.as_mut_slice(),
                                          p.as_mut_slice(),
                                          q.as_mut_slice(),
                                          dmp1.as_mut_slice(),
                                          dmq1.as_mut_slice(),
                                          iqmp.as_mut_slice());

    match result {
        Err(x) => {
            return x;
        },
        Ok(()) => {},
    }

    let privkey = SgxRsaPrivKey::new();
    let pubkey = SgxRsaPubKey::new();

    let result = pubkey.create(mod_size,
                               exp_size,
                               n.as_slice(),
                               e.as_slice());
    match result {
        Err(x) => return x,
        Ok(()) => {},
    };

    let result = privkey.create(mod_size,
                                exp_size,
                                e.as_slice(),
                                p.as_slice(),
                                q.as_slice(),
                                dmp1.as_slice(),
                                dmq1.as_slice(),
                                iqmp.as_slice());
    match result {
        Err(x) => return x,
        Ok(()) => {},
    };

    let mut ciphertext: Vec<u8> = vec![0_u8; 256];
    let mut chipertext_len: usize = ciphertext.len();
    let ret = pubkey.encrypt_sha256(ciphertext.as_mut_slice(),
                                    &mut chipertext_len,
                                    text_slice);
    match ret {
        Err(x) => {
            return x;
        },
        Ok(()) => {
            println!("rsa chipertext_len: {:?}", chipertext_len);
        },
    };

    let mut plaintext: Vec<u8> = vec![0_u8; 256];
    let mut plaintext_len: usize = plaintext.len();
    let ret = privkey.decrypt_sha256(plaintext.as_mut_slice(),
                                     &mut plaintext_len,
                                     ciphertext.as_slice());
    match ret {
        Err(x) => {
            return x;
        },
        Ok(()) => {
            println!("rsa plaintext_len: {:?}", plaintext_len);
        },
    };

    if plaintext[..plaintext_len].consttime_memeq(text_slice) == false {
        return sgx_status_t::SGX_ERROR_UNEXPECTED;
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

    println!("Encrypt Iris dataset:");
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
