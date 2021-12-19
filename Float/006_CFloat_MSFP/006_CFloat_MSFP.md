# Floating point accumulation order impact on numerical stability

在深度学习硬件崛起之前，大多数的科学计算或者Machine Learning的算法是基于IEEE Float32 和 Float64的。 随着AI处理器的演进，我们也
在学术研究或者是商业产品中看到了一些新推出的浮点类型。今天，我们就来盘点一些有意思的浮点类型。

### IEEE Float32 and Float16


### BF16

BF16 是由Google brain提出的一个16-bit的浮点类型。 BF16是由 1-bit sign，8-bit exponent， 和 7-bit fraction组成。 
![image](./assets/001_BF16.PNG)

BF16的提出是为了解决FP16在deep learning应用中遇到的一些问题。
* BF16 和 FP32 的range是一致的，远大于FP16. 缺点则是，BF16只有7个bit的mantissa，精度上是低于FP16的。
* BF16 基本上可以看作成一个“截断”版的 FP32, 两者之间的转换是非常直接，所需要的电路也会非常简单。 BF16 <--> FP32之间的转换在training的过程
中是会频繁发生的，BF16的使用能有效的降低电路的面积。

BF16 首先是在Google的TPU中得到支持，其后在业界得到了广泛的支持。当前主流的硬件厂商都对BF16做了深度的优化实现。
* **Google** TPUs and Tensorflow.
* **Nvidia** CUDA TensorCore  
* **Intel** Intel Habana Gaudi, Xeon processors (AVX-512 BF16 extensions), and Intel FPGAs
* **Arm** ArmV8.6-A
* **AMD** AMD ROCm

### TF32

[TF32](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) 是由Nvidia提出，首发于A100 GPU中。
![image](./assets/002_TF32.PNG)


## BF8 

### MSFP

### Tesla CFloat

One key property enabling this configurability is the fact that different
parameters, namely weights, gradients and activations, have different
precision and dynamic range requirements to achieve high training accuracy
and convergence. 

