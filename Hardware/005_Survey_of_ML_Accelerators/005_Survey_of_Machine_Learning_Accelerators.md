#

##
全文来自 《Survey of Machine Learning Accelerators》 [link](https://arxiv.org/pdf/2009.00993.pdf), 介绍了能看到的大多数关于深度学习
的AI芯片，涵盖了CPU, GPU, VPU, xPU, FPGA， Neuromorphic Chip 等几乎所有门类的芯片。 这篇文章的精髓如下图所示。

![Survey](assets/1_overview.PNG)

* 这张图的横轴为 Peak Power，峰值功耗， 这张图的竖轴为 Peak Performance，以Gops/sec （giga-operations per seconds）来作为单位，这个指标显示了芯片的运算能力。 
这里边的power和performance都是峰值，对于不同的网络来说，未必能得到理想的utilization。如何能更好的支持不同的workload，提高utilization，这也是一个很值得讨论的问题。
* 图的右上方，是那些运算能力很强同时所需功耗也很大的那些芯片，比如DGX系列的Data Center Systems了，不光是单卡或单芯片的系统了。
* 图的左下方，是那些在AIoT，穿戴式设备上的芯片，运算能力要求相对没那么高，但是功耗要求比较高。
* 图中还有几条直线，如 10 TeraOps/W ， 1 TeraOps/W， 100 Gops/W ，这些显示的是另一个重要指标，每Watt功耗带来的计算量，来表征
power efficiency，一般来说越大越好。
* 图中用不同的形状来标示了芯片支持的numeric precision 计算精度，从 analog，int1， int2 到 fp64 都有涵盖。 主流的inference芯片基本都支持
int8；而training芯片一般都支持fp32/fp16。

### Very Low Power

这类的Chip大多数应用在功耗敏感的场景中，如AIot，语音唤醒等地方。由于功耗的限制，一般能提供的Gops不会太高。如何提高计算效率，是此类芯片面临的一个核心
挑战。
* Intel Movidius VPU 这是Intel收购Movidius之后的一个针对IOT市场的产品线，主要针对的市场包括安防市场和零售业IOT等。
* TPU Edge  19年Google出的一个低功耗版的TPU， 还配合TF lite的推广。 同年，一个maker的开发板[coral](https://coral.ai/products/)也同步release了。
* DianNao 系列的芯片，有DianNao，DaDianNao， ShiDianNao，PuDianNao （取名鬼才）， 这一系列的芯片都是来自陈天石团队的研究成果。 这些目前还
都是研究芯片，并没有正式的量产和发布。 好在，寒武纪公司成立了，把中科院的研究成果持续的商业化。
* [AIStorm](https://aistorm.ai/) 是一个硅谷的startup公司的产品，走的是analog和mixed signal的技术路线，所以在功耗比上有明显的优势。

### Embedded 


### Autonomous 
* Hailo.AI
* Horizon Robotics
* Huawei HiSilicon Ascend 310
* NVIDIA Jetson-TX1
* The Tesla Full Self-Driving (FSD)
* MobileEye EyeQ5 

### Data Center Chips and Cards 
* The Intel second-generation Xeon Scalable processors
* Grayskull accelerator Tenstorrenti
* FPGA
* GPU-based Accelerators
### Data Center System