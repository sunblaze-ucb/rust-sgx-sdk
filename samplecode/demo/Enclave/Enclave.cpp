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

#include "Enclave.h"

#include <iostream>
#include <cmath>

#define BATCH_SIZE 100
#define SAMPLE_COL_NUMBER 53
#define TARGET_COL_NUMBER 1
#define MAC_BYTE_NUMBER 16
#define IV_BYTE_NUMBER 12
#define TEST_SAMPLE_NUMBER 8190
#define BATCH_NUMBER 326
#define SAMPLE_NUMBER BATCH_SIZE*BATCH_NUMBER

using namespace util;
using namespace std;

Enclave* Enclave::instance = NULL;

Enclave::Enclave() {}

Enclave* Enclave::getInstance() {
    if (instance == NULL) {
        instance = new Enclave();
    }

    return instance;
}


Enclave::~Enclave() {
    int ret = -1;

    if (INT_MAX != context) {
        int ret_save = -1;
        ret = enclave_ra_close(enclave_id, &status, context);
        if (SGX_SUCCESS != ret || status) {
            ret = -1;
            Log("Error, call enclave_ra_close fail", log::error);
        } else {
            // enclave_ra_close was successful, let's restore the value that
            // led us to this point in the code.
            ret = ret_save;
        }

        Log("Call enclave_ra_close success");
    }

    sgx_destroy_enclave(enclave_id);
}



sgx_status_t Enclave::createEnclave() {
    sgx_status_t ret;
    int launch_token_update = 0;
    int enclave_lost_retry_time = 1;
    sgx_launch_token_t launch_token = {0};

    memset(&launch_token, 0, sizeof(sgx_launch_token_t));

    do {
        ret = sgx_create_enclave(this->enclave_path,
                                 SGX_DEBUG_FLAG,
                                 &launch_token,
                                 &launch_token_update,
                                 &this->enclave_id, NULL);

        if (SGX_SUCCESS != ret) {
            Log("Error, call sgx_create_enclave fail", log::error);
            print_error_message(ret);
            break;
        } else {
            Log("Call sgx_create_enclave success");

            ret = enclave_init_ra(this->enclave_id,
                                  &this->status,
                                  false,
                                  &this->context);
        }

    } while (SGX_ERROR_ENCLAVE_LOST == ret && enclave_lost_retry_time--);

    if (ret == SGX_SUCCESS)
        Log("Enclave created, ID: %llx", this->enclave_id);


    return ret;
}


sgx_enclave_id_t Enclave::getID() {
    return this->enclave_id;
}

sgx_status_t Enclave::getStatus() {
    return this->status;
}

sgx_ra_context_t Enclave::getContext() {
    return this->context;
}

inline void readBytes(char* filename, uint8_t* byte_array, int length) {
    ifstream ifs(filename, ios::binary);
    if(ifs) {
        ifs.seekg(0, ios::beg);
        ifs.read((char*)byte_array, length);
        ifs.close();
    }
    else printf("file not found: %s", filename);
}

inline void readCSV(char* filename, double* data_array, double* label_array, int row, int col) {
    ifstream ifs(filename, ios::binary);
    if(ifs) {
        char delim;
        int label_cnt = 0;
        int data_cnt = 0;
        for(int i = 0; i < row; ++i) {
            ifs >> label_array[label_cnt];
            ++label_cnt;
            ifs >> delim;
            for(int j = 0; j < col; ++j) {
                ifs >> data_array[data_cnt];
                ++data_cnt;
                if(j!=col-1) ifs >> delim;
            }
        }
    }
    else printf("test file not found: %s", filename);
}

sgx_status_t Enclave::trainModel() {
    sgx_status_t ret;

    Log("Training DP Logistic Regression Model...");

    Log("Setting Meta Data...");
    int iters = 10000;
    double alpha = 0.1;
    // We do not have regularization term here so this is 1 for LR
    double L = 1.0;
    // We now hard-code the eps and delta here, will change this to the arguments later.
    double eps = 1.0;
    double delta = 1.0/(SAMPLE_NUMBER*SAMPLE_NUMBER);
    double std_dev = 4.0*L*sqrt(iters*log2(1.0/delta))/(SAMPLE_NUMBER*eps);

    Log("Preparing Model and Gradient...");
    double* model = new double[SAMPLE_COL_NUMBER];
    memset(model, 0, SAMPLE_COL_NUMBER*8);
    uint8_t* gradient = new uint8_t[SAMPLE_COL_NUMBER*8];

    Log("Preparing Encryption...");
    uint8_t* aes_gcm_key = new uint8_t[16];
    memset(aes_gcm_key, 0, 16);
    uint8_t* aes_gcm_iv = new uint8_t[12];
    memset(aes_gcm_iv, 0, 12);
    uint8_t* inputs_mac[BATCH_NUMBER];
    for(int i = 0; i < BATCH_NUMBER; ++i) {
        inputs_mac[i] = new uint8_t[16];
    }
    uint8_t* targets_mac[BATCH_NUMBER];
    for(int i = 0; i < BATCH_NUMBER; ++i) {
        targets_mac[i] = new uint8_t[16];
    }
    uint8_t* model_mac = new uint8_t[16];
    uint8_t* gradient_mac = new uint8_t[16];

    Log("Loading Data...");
    uint8_t* batch_cipher[BATCH_NUMBER];
    for(int i = 0; i < BATCH_NUMBER; ++i) {
        batch_cipher[i] = new uint8_t[BATCH_SIZE*SAMPLE_COL_NUMBER*8];
    }
    uint8_t* targets_cipher[BATCH_NUMBER];
    for(int i = 0; i < BATCH_NUMBER; ++i) {
        targets_cipher[i] = new uint8_t[BATCH_SIZE*8];
    }

    Log("Read Training Data Files...");
    char sample_name[256];
    char sample_mac_name[256];
    char target_name[256];
    char target_mac_name[256];
    for(int i = 0; i < BATCH_NUMBER; ++i) {
        sprintf(sample_name, "../Datasets/facebook_train_encrypted_sample_%d", i);
        sprintf(sample_mac_name, "../Datasets/facebook_train_mac_sample_%d", i);
        sprintf(target_name, "../Datasets/facebook_train_encrypted_target_%d", i);
        sprintf(target_mac_name, "../Datasets/facebook_train_mac_target_%d", i);
        readBytes(sample_name, batch_cipher[i], BATCH_SIZE*SAMPLE_COL_NUMBER*8);
        readBytes(sample_mac_name, inputs_mac[i], 16);
        readBytes(target_name, targets_cipher[i], BATCH_SIZE*8);
        readBytes(target_mac_name, targets_mac[i], 16);
    }

    Log("Encrypting Model...");
    uint8_t* model_cipher  = new uint8_t[SAMPLE_COL_NUMBER*8];
    ret = aes_gcm_128_encrypt(this->enclave_id,
                              &this->status,
                              aes_gcm_key,
                              (uint8_t*)model,
                              SAMPLE_COL_NUMBER*8,
                              aes_gcm_iv,
                              model_cipher,
                              model_mac);

    Log("Training");
    for(int i = 0; i < iters; ++i) {
        int batch_index = i%BATCH_NUMBER;
        ret = compute_grad(this->enclave_id,
                           &this->status,
                           model_cipher,
                           SAMPLE_COL_NUMBER,
                           batch_cipher[batch_index],
                           SAMPLE_COL_NUMBER*BATCH_SIZE,
                           targets_cipher[batch_index],
                           BATCH_SIZE,
                           gradient,
                           aes_gcm_key,
                           aes_gcm_iv,
                           model_mac,
                           inputs_mac[batch_index],
                           targets_mac[batch_index],
                           gradient_mac);
        ret = add_normal_noise(this->enclave_id,
                               &this->status,
                               std_dev,
                               gradient,
                               SAMPLE_COL_NUMBER,
                               aes_gcm_key,
                               aes_gcm_iv,
                               gradient_mac);
        ret = update_model(this->enclave_id,
                           &this->status,
                           model_cipher,
                           gradient,
                           SAMPLE_COL_NUMBER,
                           alpha,
                           aes_gcm_key,
                           aes_gcm_iv,
                           model_mac,
                           gradient_mac);
    }
    double* decrypted_model = new double[SAMPLE_COL_NUMBER];
    ret = aes_gcm_128_decrypt(this->enclave_id,
                              &this->status,
                              aes_gcm_key,
                              model_cipher,
                              SAMPLE_COL_NUMBER*8,
                              aes_gcm_iv,
                              model_mac,
                              (uint8_t*)decrypted_model);

    Log("Printing Trained Model...");
    for (int i = 0; i < SAMPLE_COL_NUMBER; ++i) {
        printf("%lf ", decrypted_model[i]);
    }
    printf("\n");

    /*Log("Freeing Training Memory...");
    delete[] model;
    delete[] gradient;
    for(int i = 0; i < BATCH_NUMBER; ++i) {
        delete[] targets_mac[i];
    }
    delete[] model_mac;
    delete[] gradient_mac;
    for(int i = 0; i < BATCH_NUMBER; ++i) {
        delete[] batch_cipher[i];
    }
    for(int i = 0; i < BATCH_NUMBER; ++i) {
        delete[] targets_cipher[i];
    }*/

    /*Log("Loading Test Dataset...");
    double* test_data = new double[TEST_SAMPLE_NUMBER*SAMPLE_COL_NUMBER];
    double* test_labels = new double[TEST_SAMPLE_NUMBER];
    double* test_result = new double[TEST_SAMPLE_NUMBER];
    char test_name[256];
    sprintf(test_name, "../Datasets/facebook_test.csv");
    readCSV(test_name, test_data, test_labels, TEST_SAMPLE_NUMBER, SAMPLE_COL_NUMBER);
    for(int i = 0; i < 100; ++i) {
        printf("#%lf ", test_data[i]);
    }
    printf("\n");
    for(int i = 0; i < 100; ++i) {
        printf("*%lf ", test_labels[i]);
    }
    printf("\n");

    Log("Predicting...");
    ret = sgx_predict(this->enclave_id,
                      &this->status,
                      decrypted_model,
                      SAMPLE_COL_NUMBER,
                      test_data,
                      TEST_SAMPLE_NUMBER*SAMPLE_COL_NUMBER,
                      test_result,
                      TEST_SAMPLE_NUMBER);
    int correct_num = 0;
    for(int i = 0; i < TEST_SAMPLE_NUMBER; ++i) {
        double sample_res = 0.0;
        if(test_result[i] > 0.5) sample_res = 1.0;
        // if(!(i%100)) printf("%lf %lf %lf\n", test_result[i], sample_res, test_labels);
        if(test_labels[i] == sample_res) ++correct_num;
    }
    printf("Correct Ratio is %lf\n", (double)correct_num/TEST_SAMPLE_NUMBER);*/

    /*Log("Freeing Testing Memory");
    delete[] decrypted_model;
    delete[] test_data;
    delete[] test_labels;
    delete[] test_result;*/

    return SGX_SUCCESS;
}
