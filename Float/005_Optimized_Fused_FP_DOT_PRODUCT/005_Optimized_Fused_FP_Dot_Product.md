# Floating point accumulation order impact on numerical stability

Floating Point在AI芯片中是一个非常重要的数据类型，从早期的FP32, FP16， 到最新提出的BF16，NVIDIA家的TF32，微软提出的MSFP系列 [MSFP16 ~ MSFP11](https://www.microsoft.com/en-us/research/blog/a-microsoft-custom-data-type-for-efficient-inference/).

准备写一些文章来探讨一下这些float point的计算背后的事情，欢迎大家的关注和讨论。

## NNP-T 

[Fused Floating Point Many-Term Dot Product Design from Intel NNP-T](http://arith2020.arithsymposium.org/resources/paper_69.pdf) 


