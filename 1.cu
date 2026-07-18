#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include<iostream>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<chrono>
//global means this code is called from cpu to gpu
__global__ void vecAdd(float* a,float* b,float* c,int n){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    //this is the idx global for the thread
    if(idx<n){
        c[idx]=a[idx]+b[idx];
    }
}
int main(){
    float *a,*b,*c;
    int n=100000;
    int byte=n*sizeof(float);
    float* ram_a= new float[n];
    float* ram_b=new float[n];
    float* ram_c=new float[n];
    for(int i=0;i<n;i++){
        ram_a[i]=1.0f;
        ram_b[i]=3.0f;
    }
    //cudaMalloc allocates mem on the gpu
    cudaMalloc(&a,byte);
    cudaMalloc(&b,byte);
    cudaMalloc(&c,byte);
    int num_threads=256;
    int num_blocks=(n+num_threads-1)/num_threads;
    cudaMemcpy(a,ram_a,byte,cudaMemcpyHostToDevice);
    cudaMemcpy(b,ram_b,byte,cudaMemcpyHostToDevice);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vecAdd<<<num_blocks,num_threads>>>(a,b,c,n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpums=0;
    cudaEventElapsedTime(&gpums,start,stop);
    printf("GPU TIME: :%.3f ms \n",gpums);
    cudaMemcpy(ram_c,c,byte,cudaMemcpyDeviceToHost);
    // printf("%f\n can i access c",c[0]);
    float* cpu_c=new float[n];
    auto cpu_start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<n;i++){
        cpu_c[i]=ram_a[i]+ram_b[i];
    }
    auto cpu_end=std::chrono::high_resolution_clock::now();
    std::chrono::duration<float,std::milli> cpu_dur=cpu_end-cpu_start;
    std::cout<<"CPU TIME:"<<cpu_dur.count()<<"ms"<<std::endl;
    std::cout<<"speedup->"<<cpu_dur.count()/gpums<<"\n";
    cudaFree(&a);
    cudaFree(&b);
    cudaFree(&c);
    delete[] ram_a;
    delete[] ram_b;
    delete[] ram_c;
    return 0;
}
/*
 *
 *
 GPU TIME: :0.045 ms
 CPU TIME:0.32537ms
 speedup->7.24203
 *
 */
