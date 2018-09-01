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
use sgx_types::*;
use sgx_urts::SgxEnclave;

use std::io::{Read, Write};
use std::fs;
use std::path;
use std::mem;

static ENCLAVE_FILE: &'static str = "enclave.signed.so";
static ENCLAVE_TOKEN: &'static str = "enclave.token";

extern {
    fn calc_sha256 (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, input_str: *const u8, some_len: usize, hash: &mut [u8;32]) -> sgx_status_t;

    fn aes_gcm_128_encrypt (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, key: &[u8;16], plaintext: *const u8, text_len: usize, iv: &[u8;12], ciphertext: *mut u8, mac: &mut [u8;16]) -> sgx_status_t;

    fn aes_gcm_128_decrypt (eid: sgx_enclave_id_t, retval: *mut sgx_status_t, key: &[u8;16], ciphertext: *const u8, text_len: usize, iv: &[u8;12], mac: &[u8;16], plaintext: *mut u8) -> sgx_status_t;

    fn aes_cmac(eid: sgx_enclave_id_t, retval: *mut sgx_status_t, text: *const u8, text_len: usize, key: &[u8;16], cmac: &mut [u8;16]) -> sgx_status_t;

    fn ras_key(eid: sgx_enclave_id_t, retval: *mut sgx_status_t, text: *const u8, text_len: usize) -> sgx_status_t;

    fn sample_main (eid: sgx_enclave_id_t, retval: *mut sgx_status_t) -> sgx_status_t;
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
    
    println!("Testing Crypto Primitives...");

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

    println!("{:?}", aes_gcm_ciphertext);

    let result = unsafe {
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

    println!("[+] say_something success...");
    
    enclave.destroy();
}
