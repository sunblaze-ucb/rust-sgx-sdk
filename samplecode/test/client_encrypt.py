import os
import numpy as np
import struct
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import (
    Cipher, algorithms, modes
)
def encrypt(key, plaintext, associated_data):
    #iv = os.urandom(12) #TODO: need to delete hardcoded iv below and use a random IV for each encryption
    iv_array = [0] * 12
    iv = "".join(map(chr, iv_array))
    #print("iv in hex: ", iv.format(255, '#04x'))
    # Construct AES-GCM Cipher object for encryption
    encryptor = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend()).encryptor()
    # Addition message used to calculate MAC but will not be encrypted
    encryptor.authenticate_additional_data(associated_data)
    # Encrypt the secret, GCM is basically AES-CTR + MAC
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return (iv, ciphertext, encryptor.tag)
def decrypt(key, associated_data, iv, ciphertext, tag):
    # Construct AES-GCM Cipher object for decription
    decryptor = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend()).decryptor()
    # Inclued additional message for MAC calculation and verify the integraty
    decryptor.authenticate_additional_data(associated_data)
    return decryptor.update(ciphertext) + decryptor.finalize()
# read a CSV file, define the batch size and data type to read. 
# Then the sample/target batch will be encrypted and save to files.
# dataset_file: path to dataset file
# batch_size: number of rows for a batch
# data_type: sample data or target data
def read_split_encrypt_data(dataset_file, batch_size, data_type):
    print "Processing ", data_type, " data..."
    #TODO: need to have a better key
    key_arr = [0x0] * 16
    key = "".join(map(chr, key_arr))
    #print "key is: ", key
    associated_text = b"" #TODO: currently there is no associated text for MAC calculation
    with open(dataset_file) as f:
        ncols = len(f.readline().split(','))
    print 'The CSV has ', ncols, 'columns.'
    if 'sample' == data_type: 
        # read a dataset and convert the numbers to be float
        # skip 1st column
        dataset_np = np.loadtxt(dataset_file, dtype='d', delimiter=',', usecols=range(1,ncols))
        ncols = ncols - 1
    elif 'target' == data_type:
        # only read 1st column for target
        dataset_np = np.loadtxt(dataset_file, dtype='d', delimiter=',', usecols=(0,))
        ncols = 1
    print dataset_np
    print len(dataset_np)
    total_batch_number = len(dataset_np)/batch_size
    print "There are ", total_batch_number, "batches in the dataset. If the tail rows are less than the batch_size, they will be discarded."
    batch_count = 0
    for batch_count in range(0,total_batch_number):
        print "Reading and encrypting batch ", batch_count
        batch_start = batch_count * batch_size
        batch_end = (batch_count + 1) * batch_size
        # dump the array to byte array (the data type is string)
        dataset_np_byte_array = dataset_np[batch_start:batch_end].tobytes()
        #print "len(dataset_np_byte_array): ", len(dataset_np_byte_array)
        print "dataset_np_byte_array hex: ", ":".join("{:02x}".format(ord(c)) for c in dataset_np_byte_array)
        # encrypt the byte array
        iv, dataset_np_byte_array_encrypted, mac = encrypt(key, dataset_np_byte_array, associated_text)
        print "dataset_np_byte_array_encrypted hex: ", ":".join("{:02x}".format(ord(c)) for c in dataset_np_byte_array_encrypted)
        print "mac hex: ", ":".join("{:02x}".format(ord(c)) for c in mac)
        DEBUG_MODE = False
        if True == DEBUG_MODE:
            # decrypt the ciphertext and check the result
            decrypted_dataset = decrypt(key, associated_text, iv, dataset_np_byte_array_encrypted, mac)
            print "decrypted_dataset len: ", len(decrypted_dataset)
            print "decrypted_dataset hex: ", ":".join("{:02x}".format(ord(c)) for c in decrypted_dataset)
            dataset_np_float = np.frombuffer(dataset_np_byte_array, dtype=float)
            print "decrypted_dataset float: ", dataset_np_float
        # save the byte array into a file
        print "Saving processed batch ", batch_count, " into files"
        dataset_encrypted_file = dataset_file[:-4] + '_encrypted_' + data_type + '_' + str(batch_count)
        dataset_mac_file = dataset_file[:-4] + '_mac_' + data_type + '_' + str(batch_count)
        # dataset_iv_file = dataset_file[:-4] + '_iv_' + data_type + '_' + str(batch_count)
        
        f =  open(dataset_encrypted_file, 'wb')
        f.write(dataset_np_byte_array_encrypted)
        f.close()
        f = open(dataset_mac_file, 'wb')
        f.write(mac)
        f.close()
        # f = open(dataset_iv_file, 'wb')
        # f.write(iv)
        # f.close()
        
        batch_count += 1
    return
# validating using all zero key,iv, and 16 byte all zero data
#read_split_encrypt_data("dataset.txt", 1, 'sample')
#print "aes-gcm-128 expected ciphertext:0388dace60b6a392f328c2b971b2fe78"
# process csv data
batch_size = 100
read_split_encrypt_data("datasets/liver-disorders-train.csv", batch_size, 'sample')
read_split_encrypt_data("datasets/liver-disorders-train.csv", batch_size, 'target')
# read_split_encrypt_data("../datasets/credit-score-train.csv", batch_size, 'sample')
# read_split_encrypt_data("../datasets/credit-score-train.csv", batch_size, 'target')
