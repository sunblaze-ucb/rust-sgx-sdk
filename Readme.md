# Privacy-Preserving Machine Learning as a Service Demo

More and more cloud service providers (like Ali Cloud, Amazon EC2, Microsoft Azure) start to provide MLaaS which allows user to upload their datasets, press a button and get the machine learning result they want. 
However, when the users want to train some private data like medical records or financial datasets, they can not trust the service providers because they can steal the data for multiple reasons such as government surveillance or commercial benefits. Trusted execution environment can provide such a protection against malicious service providers.
Another concern is that the model itself can leak some privacy of the dataset which can be used to reconstruct some of the data samples. Differential privacy provides a formal approach to defend against this attack.
As a result, we want to provide differential private machine learning on Intel SGX as a MLaaS on Ali Cloud. We implement [differential private stochastic gradient descent](http://cs-people.bu.edu/omthkkr/papers/TPDPCO.pdf) on Intel SGX on Ali cloud together with the complete workflow like remote attestation and file encryption on client side without SGX. Slides of the demo can be found [here](https://github.com/spartazhihu/spartazhihu.github.io/blob/master/files/Ali%20Cloud%20Computing%20Demo%20Slides.pptx).

# Requirement

Ubuntu 16.04

[Intel SGX SDK 2.2 for Linux](https://01.org/intel-software-guard-extensions/downloads) installed

Docker (Recommended)

# How to run the demo

The demo codes reside in samplecode/demo. It is built on top of baidu rust-sgx-sdk. This repo is also a fork of the baidu rust-sgx-sdk.

`$ docker pull baiduxlab/sgx-rust`

`$ docker run -v /your/path/to/rust-sgx:/root/sgx -ti --device /dev/isgx baiduxlab/sgx-rust`

`$ export SGX_MODE=SW` 

`$ cd samplecode/demo`

`$ cd Application`

`make`

`./app &`

`$ cd ../ServiceProvider`

`make`

`$ ./app`

# License

The demo is provided under the BSD license inherited from Baidu Rust-SGX SDK. Please refer to the [License file](LICENSE) for details.

# Authors

[Lun Wang](wanglunucb.com), Li Shen, Yanhui Zhao

# Acknowledgement

Thanks to [Baidu Rust-SGX SDK](https://github.com/baidu/rust-sgx-sdk) project.

# Contacts

[Lun Wang](wanglunucb.com), wanglun@berkeley.edu

