<div align="center">

  <h2><b> Discrete Diffusion in Large Language and Multimodal Models: A Survey </b></h2>
  <h4> An overview of research in discrete diffusion large language and multimodal models</h4>

</div>

<div align="center">


<a href="https://arxiv.org/pdf/2506.13759" target="_blank"><img src="https://img.shields.io/badge/arXiv-2506.13759-009688.svg" alt="arXiv"></a>

![](https://img.shields.io/github/stars/LiQiiiii/Discrete-Diffusion-Large-Language-Models-Survey?color=yellow)
![](https://img.shields.io/github/forks/LiQiiiii/Discrete-Diffusion-Large-Language-Models-Survey?color=lightblue)
![](https://img.shields.io/github/last-commit/LiQiiiii/DLLM-Survey?color=green)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-blue)](https://github.com/LiQiiiii/DLLM-Survey/pulls)


</div>





This repository is for our paper:

> **[Discrete Diffusion in Large Language and Multimodal Models: A Survey](https://arxiv.org/pdf/2506.13759)** \
> [Runpeng Yu*](https://scholar.google.ca/citations?user=sMgjdygAAAAJ&hl=en)<sup>1</sup>, [Qi Li*](https://scholar.google.ca/citations?user=gsV-g2wAAAAJ&hl=en)<sup>1</sup>, [Xinchao Wang](https://scholar.google.ca/citations?user=w69Buq0AAAAJ&hl=en)<sup>†1</sup> \
> <sup>1</sup>National University of Singapore, Singapore \
> <sup>*</sup>Equal contribution, random order. \
> <sup>†</sup>Corresponding author: xinchao@nus.edu.sg

<p align="center">
  <img src="assets/ar_vs_diff.png" alt="Diagram 2" style="max-width:100%;" /><br />
  <strong>Figure 1. Difference between autoregressive models and discrete diffusion models.</strong>
</p>

<p align="center">
  <img src="assets/timeline.png" alt="Diagram 2" style="max-width:100%;" /><br />
  <strong>Figure 2. A timeline of existing dLLMs and dMLLMs in recent years.</strong>
</p>

## Citation
```
@article{yu2025discrete,
  title={Discrete Diffusion in Large Language and Multimodal Models: A Survey},
  author={Yu, Runpeng and Li, Qi and Wang, Xinchao},
  journal={arXiv preprint arXiv:2506.13759},
  year={2025}
}
```

---
> 🌟 If you find this resource helpful, please consider to star this repository and cite our [work](#citation)!
> 
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> **To add new papers or models**
>
> If you want to add your paper or update details like conference info or code URLs, please raise an PR. You can generate the necessary markdown for each paper by filling out `generate_item.py` and running `python generate_item.py`. We greatly appreciate your contributions. Alternatively, you can emal me (liqi@u.nus.edu) the links to your paper and code, and I will add your paper to the list as soon as possible.

---
<p align="center">
<img src="assets/taxonomy.png" width = "95%" alt="" align=center />
</p>

### Quick Links

- [Mathematics](#mathematics)
  - [Discrete Diffusion Model and Transition Matrix](#discrete-diffusion-model-and-transition-matrix)
  - [Simplified Masked Diffusion Model](#simplified-masked-diffusion-model)
  - [Continuous Time Discrete Denoising Models](#continuous-time-discrete-denoising-models)
  - [Concrete Score](#concrete-score)
  - [Reparameterized Discrete Diffusion Model](#reparameterized-discrete-diffusion-model)
- [Modeling Language Diffusion](#modeling-language-diffusion)
- [Representative Model](#representative-model)
  - [Discrete Diffusion Models around 1B](#discrete-diffusion-models-around-1b)
  - [Large Diffusion Language Models](#large-diffusion-language-models)
  - [Large Diffusion Multimodal Models](#large-diffusion-multimodal-models)
  - [Large Unified Models](#large-unified-models)
- [Training Techniques](#training-techniques)
  - [Initialization Technique](#initialization-technique)
  - [Complementary Masking Technique](#complementary-masking-technique)
  - [Masking Scheduling Technique](#masking-scheduling-technique)
  - [Reweighting Technique](#reweighting-technique)
  - [Distillation](#distillation)
  - [Training-Testing Input Discrepancy](#training-testing-input-discrepancy)
- [Inference Techniques](#inference-techniques)
  - [Unmasking Techniques](#unmasking-techniques)
  - [Remasking Techniques](#remasking-techniques)
  - [Prefilling and Caching Technique](#prefilling-and-caching-technique)
  - [Guidance Technique](#guidance-technique)
  - [Sampling Technique](#sampling-technique)
  - [Context Length Extension](#context-length-extension)
  - [Sparse Computation](#sparse-computation)
  - [Response Length Control](#response-length-control)
- [Quantization](#quantization)
- [Privacy and Safety](#privacy-and-safety)
- [Applications](#applications)
  - [Text Generation and Style Control](#text-generation-and-style-control)
  - [Text Editing and Summarization](#text-editing-and-summarization)
  - [Sentiment Analysis and Data Augmentation](#sentiment-analysis-and-data-augmentation)
  - [Knowledge and Reasoning](#knowledge-and-reasoning)
  - [Vision and Multimodal](#vision-and-multimodal)
  - [Biological and Drug Discovery](#biological-and-drug-discovery)
- [Toolkits](#toolkits)



### Mathematics

#### Discrete Diffusion Model and Transition Matrix
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Deep unsupervised learning using nonequilibrium thermodynamics](https://proceedings.neurips.cc/paper/2021/file/958c530554f78bcd8e97125b70e6973d-Paper.pdf) <br> Sohl-Dickstein, Jascha, Eric Weiss, Niru Maheswaranathan, Surya Ganguli |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/1503.03585v8.png"> |[Paper](https://proceedings.neurips.cc/paper/2021/file/958c530554f78bcd8e97125b70e6973d-Paper.pdf)|[//]: #06/16
|[Structured denoising diffusion models in discrete state-spaces](https://proceedings.neurips.cc/paper/2021/file/958c530554f78bcd8e97125b70e6973d-Paper.pdf) <br> Austin, Jacob, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne Van Den Berg |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2107.03006v3.png"> |[Paper](https://proceedings.neurips.cc/paper/2021/file/958c530554f78bcd8e97125b70e6973d-Paper.pdf)|[//]: #06/16
|[Argmax flows and multinomial diffusion: Learning categorical distributions](https://proceedings.neurips.cc/paper/2021/file/67d96d458abdef21792e6d8e590244e7-Paper.pdf) <br> Hoogeboom, Emiel, Didrik Nielsen, Priyank Jaini, Patrick Forré, Max Welling |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2102.05379v3.png"> |[Paper](https://proceedings.neurips.cc/paper/2021/file/67d96d458abdef21792e6d8e590244e7-Paper.pdf)|[//]: #06/16
|[Generalized interpolating discrete diffusion](https://arxiv.org/abs/2503.04482) <br> Dimitri von Rütte, Janis Fluri, Yuhui Ding, Antonio Orvieto, Bernhard Schölkopf, Thomas Hofmann |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2503.04482v2.png"> |[Paper](https://arxiv.org/abs/2503.04482)|[//]: #09/30


#### Simplified Masked Diffusion Model
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Simple and Effective Masked Diffusion Language Models](https://arxiv.org/pdf/2406.07524) <br> Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, Volodymyr Kuleshov |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2406.07524v2.png"> |[Paper](https://arxiv.org/pdf/2406.07524)|[//]: #06/16
|[Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.04329) <br> Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, Michalis K. Titsias |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2406.04329v4.png"> |[Paper](https://arxiv.org/abs/2406.04329)|[//]: #06/17


#### Continuous Time Discrete Denoising Models
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[A Continuous Time Framework for Discrete Denoising Models](https://arxiv.org/abs/2205.14987) <br> Andrew Campbell, Joe Benton, Valentin De Bortoli, Tom Rainforth, George Deligiannidis, Arnaud Doucet |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2205.14987v2.png"> |[Paper](https://arxiv.org/abs/2205.14987)|[//]: #09/30


#### Concrete Score
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Concrete Score Matching: Generalized Score Matching for Discrete Data](https://papers.nips.cc/paper_files/paper/2022/file/df04a35d907e894d59d4eab1f92bc87b-Supplemental-Conference.pdf) <br> Chenlin Meng, Kristy Choi, Jiaming Song, Stefano Ermon |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2211.00802v2.png"> |[Paper](https://papers.nips.cc/paper_files/paper/2022/file/df04a35d907e894d59d4eab1f92bc87b-Supplemental-Conference.pdf)|[//]: #06/17
|[Score-based Continuous-time Discrete Diffusion Models](https://arxiv.org/abs/2211.16750) <br> Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, Hanjun Dai |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2211.16750v2.png"> |[Paper](https://arxiv.org/abs/2211.16750)|[//]: #06/17
|[Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data](https://arxiv.org/abs/2406.03736) <br> Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, Chongxuan Li |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2406.03736v3.png"> |[Paper](https://arxiv.org/abs/2406.03736)|[//]: #06/17
|[Target concrete score matching: A holistic framework for discrete diffusion](https://arxiv.org/abs/2504.16431) <br> Ruixiang Zhang, Shuangfei Zhai, Yizhe Zhang, James Thornton, Zijing Ou, Joshua Susskind, Navdeep Jaitly |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2504.16431v1.png"> |[Paper](https://arxiv.org/abs/2504.16431)|[//]: #07/07
|[Simple and Effective Masked Diffusion Language Models](https://arxiv.org/pdf/2406.07524) <br> Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, Volodymyr Kuleshov |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2406.07524v2.png"> |[Paper](https://arxiv.org/pdf/2406.07524)|[//]: #06/16
|[Efficient perplexity bound and ratio matching in discrete diffusion language models](https://openreview.net/forum?id=Mri9WIfxSm) <br> Etrit Haxholli, Yeti Z. Gurbuz, Oğul Can, Eli Waxman |<img width="1002" alt="image" src="https://iclr.cc/media/PosterPDFs/ICLR%202025/29916.png?t=1744207216.6464744"> |[Paper](https://openreview.net/forum?id=Mri9WIfxSm)|[//]: #07/07


#### Reparameterized Discrete Diffusion Model
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[A Reparameterized Discrete Diffusion Model for Text Generation](https://arxiv.org/pdf/2302.05737) <br> Lin Zheng, Jianbo Yuan, Lei Yu, Lingpeng Kong |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2302.05737v3.png"> |[Paper](https://arxiv.org/pdf/2302.05737)|[//]: #06/16



### Modeling Language Diffusion
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573) <br> Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, Volodymyr Kuleshov |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2503.09573v3.png"> |[Paper](https://arxiv.org/abs/2503.09573)|[//]: #09/30
|[Any-Order Flexible Length Masked Diffusion](https://arxiv.org/abs/2509.01025) <br> Jaeyeon Kim, Lee Cheuk-Kit, Carles Domingo-Enrich, Yilun Du, Sham Kakade, Timothy Ngotiaoco, Sitan Chen, Michael Albergo |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2509.01025v2.png"> |[Paper](https://arxiv.org/abs/2509.01025)|[//]: #09/30
|[Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](https://arxiv.org/abs/2505.18495) <br> Chen-Hao Chao, Wei-Fang Sun, Hanwen Liang, Chun-Yi Lee, Rahul G. Krishnan |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.18495v1.png"> |[Paper](https://arxiv.org/abs/2505.18495)|[//]: #09/30
|[Flexible-length Text Infilling for Discrete Diffusion Models](https://arxiv.org/abs/2506.13579v1) <br> Andrew Zhang, Anushka Sivakumar, Chiawei Tang, Chris Thomas |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2506.13579v1.png"> |[Paper](https://arxiv.org/abs/2506.13579v1)|[//]: #09/30




### Representative Model

#### Discrete Diffusion Models around 1B
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) <br> Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2107.03006v3.png"> |[Paper](https://arxiv.org/abs/2107.03006)|[//]: #06/17
|[DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models](https://arxiv.org/abs/2211.15029) <br> Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, Xipeng Qiu |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2211.15029v2.png"> |[Paper](https://arxiv.org/abs/2211.15029)|[//]: #06/17
|[A Reparameterized Discrete Diffusion Model for Text Generation](https://arxiv.org/abs/2302.05737) <br> Lin Zheng, Jianbo Yuan, Lei Yu, Lingpeng Kong |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2302.05737v3.png"> |[Paper](https://arxiv.org/abs/2302.05737)|[//]: #06/17
|[A Cheaper and Better Diffusion Language Model with Soft-Masked Noise](https://arxiv.org/abs/2304.04746) <br> Jiaao Chen, Aston Zhang, Mu Li, Alex Smola, Diyi Yang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2304.04746v1.png"> |[Paper](https://arxiv.org/abs/2304.04746)|[//]: #06/17
|[Diffusion-NAT: Self-Prompting Discrete Diffusion for Non-Autoregressive Text Generation](https://arxiv.org/abs/2305.04044) <br> Kun Zhou, Yifan Li, Wayne Xin Zhao, Ji-Rong Wen |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2305.04044v1.png"> |[Paper](https://arxiv.org/abs/2305.04044)|[//]: #06/17
|[TESS: Text-to-Text Self-Conditioned Simplex Diffusion](https://arxiv.org/abs/2305.08379) <br> Rabeeh Karimi Mahabadi, Hamish Ivison, Jaesung Tae, James Henderson, Iz Beltagy, Matthew E. Peters, Arman Cohan |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2305.08379v2.png"> |[Paper](https://arxiv.org/abs/2305.08379)|[//]: #06/17
|[Likelihood-Based Diffusion Language Models](https://arxiv.org/abs/2305.18619) <br> Ishaan Gulrajani, Tatsunori B. Hashimoto |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2305.18619v1.png"> |[Paper](https://arxiv.org/abs/2305.18619)|[//]: #06/17
|[Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834) <br> Aaron Lou, Chenlin Meng, Stefano Ermon |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2310.16834v3.png"> |[Paper](https://arxiv.org/abs/2310.16834)|[//]: #06/17
|[Simple and Effective Masked Diffusion Language Models](https://arxiv.org/pdf/2406.07524) <br> Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, Volodymyr Kuleshov |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2406.07524v2.png"> |[Paper](https://arxiv.org/pdf/2406.07524)|[//]: #06/16
|[Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.04329) <br> Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, Michalis K. Titsias |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2406.04329v4.png"> |[Paper](https://arxiv.org/abs/2406.04329)|[//]: #06/17
|[Unified Multimodal Discrete Diffusion](https://arxiv.org/abs/2503.20853) <br> Alexander Swerdlow, Mihir Prabhudesai, Siddharth Gandhi, Deepak Pathak, Katerina Fragkiadaki |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2503.20853v1.png"> |[Paper](https://arxiv.org/abs/2503.20853)|[//]: #06/17


#### Large Diffusion Language Models
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) <br> Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2502.09992v2.png"> |[Paper](https://arxiv.org/abs/2502.09992)|[//]: #06/17
|[Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning](https://arxiv.org/abs/2308.12219) <br> Jiasheng Ye, Zaixiang Zheng, Yu Bao, Lihua Qian, Quanquan Gu |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2308.12219v3.png"> |[Paper](https://arxiv.org/abs/2308.12219)|[//]: #06/17
|[Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://arxiv.org/abs/2410.17891) <br> Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, Hao Peng, Lingpeng Kong |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2410.17891v3.png"> |[Paper](https://arxiv.org/abs/2410.17891)|[//]: #06/17
|[Dream 7b](https://hkunlp.github.io/blog/2025/dream/) <br> Jiacheng Ye*, Zhihui Xie*, Lin Zheng*, Jiahui Gao*, Zirui Wu, Xin Jiang, Zhenguo Li, Lingpeng Kong |<img width="1002" alt="image" src="https://hkunlp.github.io/assets/img/2025-04-02-dream-img/model-1400.webp"> |[Paper](https://hkunlp.github.io/blog/2025/dream/)|[//]: #06/17
|[LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models](https://arxiv.org/abs/2505.19223) <br> Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, Chongxuan Li |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.19223v1.png"> |[Paper](https://arxiv.org/abs/2505.19223)|[//]: #06/17
|[TESS 2: A Large-Scale Generalist Diffusion Language Model](https://arxiv.org/abs/2502.13917) <br> Jaesung Tae, Hamish Ivison, Sachin Kumar, Arman Cohan |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2502.13917v2.png"> |[Paper](https://arxiv.org/abs/2502.13917)|[//]: #06/17
|[DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](https://hkunlp.github.io/blog/2025/dreamon/) <br> Zirui Wu*, Lin Zheng*, Zhihui Xie, Jiacheng Ye, Jiahui Gao, Yansong Feng, Zhenguo Li, Victoria W., Guorui Zhou , Lingpeng Kong |<img width="1002" alt="image" src="https://github.com/DreamLM/DreamOn/raw/main/figs/code_infilling_dreamon_from_short.gif"> |[Paper](https://hkunlp.github.io/blog/2025/dreamon/)|[//]: #09/30
|[Dream-Coder 7B: An Open Diffusion Language Model for Code](https://arxiv.org/abs/2509.01142) <br> Zhihui Xie, Jiacheng Ye, Lin Zheng, Jiahui Gao, Jingwei Dong, Zirui Wu, Xueliang Zhao, Shansan Gong, Xin Jiang, Zhenguo Li, Lingpeng Kong |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2509.01142v1.png"> |[Paper](https://arxiv.org/abs/2509.01142)|[//]: #09/30
|[DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation](https://arxiv.org/abs/2506.20639) <br> Shansan Gong, Ruixiang Zhang, Huangjie Zheng, Jiatao Gu, Navdeep Jaitly, Lingpeng Kong, Yizhe Zhang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2506.20639v2.png"> |[Paper](https://arxiv.org/abs/2506.20639)|[//]: #09/30
|[Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference](https://arxiv.org/abs/2508.02193) <br> Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, Zheng Li, Yuehang Yang, Hongli Yu, Xingwei Qu, Yuwei Fu, Jing Su, Ge Zhang, Wenhao Huang, Mingxuan Wang, Lin Yan, Xiaoying Jia, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Yonghui Wu, Hao Zhou |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2508.02193v1.png"> |[Paper](https://arxiv.org/abs/2508.02193)|[//]: #09/30

#### Large Diffusion Multimodal Models
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding](https://arxiv.org/abs/2505.16990) <br> Runpeng Yu, Xinyin Ma, Xinchao Wang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.16990v2.png"> |[Paper](https://arxiv.org/abs/2505.16990)|[//]: #06/17
|[LaViDa: A Large Diffusion Language Model for Multimodal Understanding](https://arxiv.org/abs/2505.16839) <br> Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, Aditya Grover |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.16839v2.png"> |[Paper](https://arxiv.org/abs/2505.16839)|[//]: #06/17
|[LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning](https://arxiv.org/abs/2505.16933) <br> Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, Chongxuan Li |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.16933v2.png"> |[Paper](https://arxiv.org/abs/2505.16933)|[//]: #06/17



#### Large Unified Models
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809) <br> Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, Mengdi Wang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.15809v1.png"> |[Paper](https://arxiv.org/abs/2505.15809)|[//]: #06/17
|[FUDOKI: Discrete Flow-based Unified Understanding and Generation via Kinetic-Optimal Velocities](https://arxiv.org/abs/2505.20147) <br> Jin Wang, Yao Lai, Aoxue Li, Shifeng Zhang, Jiacheng Sun, Ning Kang, Chengyue Wu, Zhenguo Li, Ping Luo |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.20147v1.png"> |[Paper](https://arxiv.org/abs/2505.20147)|[//]: #06/17
|[Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model](https://arxiv.org/abs/2505.23606) <br> Qingyu Shi, Jinbin Bai, Zhuoran Zhao, Wenhao Chai, Kaidong Yu, Jianzong Wu, Shuangyong Song, Yunhai Tong, Xiangtai Li, Xuelong Li, Shuicheng Yan |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.23606v1.png"> |[Paper](https://arxiv.org/abs/2505.23606)|[//]: #06/17

### Training Techniques

#### Initialization Technique
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models](https://arxiv.org/abs/2211.15029) <br> Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, Xipeng Qiu |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2211.15029v2.png"> |[Paper](https://arxiv.org/abs/2211.15029)|[//]: #06/17
|[Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://arxiv.org/abs/2410.17891) <br> Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, Hao Peng, Lingpeng Kong |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2410.17891v3.png"> |[Paper](https://arxiv.org/abs/2410.17891)|[//]: #06/17
|[Dream 7b](https://hkunlp.github.io/blog/2025/dream/) <br> Jiacheng Ye*, Zhihui Xie*, Lin Zheng*, Jiahui Gao*, Zirui Wu, Xin Jiang, Zhenguo Li, Lingpeng Kong |<img width="1002" alt="image" src="https://hkunlp.github.io/assets/img/2025-04-02-dream-img/model-1400.webp"> |[Paper](https://hkunlp.github.io/blog/2025/dream/)|[//]: #06/17
|[Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding](https://arxiv.org/abs/2505.16990) <br> Runpeng Yu, Xinyin Ma, Xinchao Wang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.16990v2.png"> |[Paper](https://arxiv.org/abs/2505.16990)|[//]: #06/17

#### Complementary Masking Technique
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[LaViDa: A Large Diffusion Language Model for Multimodal Understanding](https://arxiv.org/abs/2505.16839) <br> Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, Aditya Grover |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.16839v2.png"> |[Paper](https://arxiv.org/abs/2505.16839)|[//]: #06/17


#### Masking Scheduling Technique
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Structured denoising diffusion models in discrete state-spaces](https://proceedings.neurips.cc/paper/2021/file/958c530554f78bcd8e97125b70e6973d-Paper.pdf) <br> Austin, Jacob, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne Van Den Berg |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2107.03006v3.png"> |[Paper](https://proceedings.neurips.cc/paper/2021/file/958c530554f78bcd8e97125b70e6973d-Paper.pdf)|[//]: #06/16
|[Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834) <br> Aaron Lou, Chenlin Meng, Stefano Ermon |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2310.16834v3.png"> |[Paper](https://arxiv.org/abs/2310.16834)|[//]: #06/17
|[MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200) <br> Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2202.04200v1.png"> |[Paper](https://arxiv.org/abs/2202.04200)|[//]: #06/17
|[DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models](https://arxiv.org/abs/2211.15029) <br> Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, Xipeng Qiu |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2211.15029v2.png"> |[Paper](https://arxiv.org/abs/2211.15029)|[//]: #06/17

#### Reweighting Technique
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Beyond autoregression: Discrete diffusion for complex reasoning and planning](https://arxiv.org/abs/2410.14157) <br> Jiacheng Ye, Jiahui Gao, Shansan Gong, Lin Zheng, Xin Jiang, Zhenguo Li, Lingpeng Kong |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2410.14157v3.png"> |[Paper](https://arxiv.org/abs/2410.14157)|[//]: #07/07

#### Distillation
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Distillation of Discrete Diffusion through Dimensional Correlations](https://arxiv.org/abs/2410.08709) <br> Satoshi Hayakawa, Yuhta Takida, Masaaki Imaizumi, Hiromi Wakaki, Yuki Mitsufuji |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2410.08709v4.png"> |[Paper](https://arxiv.org/abs/2410.08709)|[//]: #07/07

#### Reinforcement Learning
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809) <br> Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, Mengdi Wang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.15809v1.png"> |[Paper](https://arxiv.org/abs/2505.15809)|[//]: #06/17
|[LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models](https://arxiv.org/abs/2505.19223) <br> Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, Chongxuan Li |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.19223v1.png"> |[Paper](https://arxiv.org/abs/2505.19223)|[//]: #06/17
|[Discrete Diffusion Trajectory Alignment via Stepwise Decomposition](https://arxiv.org/abs/2507.04832) <br> Jiaqi Han, Austin Wang, Minkai Xu, Wenda Chu, Meihua Dang, Yisong Yue, Stefano Ermon |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2507.04832v1.png"> |[Paper](https://arxiv.org/abs/2507.04832)|[//]: #10/01
|[wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](https://arxiv.org/abs/2507.08838) <br> Xiaohang Tang, Rares Dolga, Sangwoong Yoon, Ilija Bogunovic |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2507.08838v1.png"> |[Paper](https://arxiv.org/abs/2507.08838)|[//]: #10/01
|[Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](https://arxiv.org/abs/2505.10446) <br> Zemin Huang, Zhiyang Chen, Zijun Wang, Tiancheng Li, Guo-Jun Qi |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.10446v2.png"> |[Paper](https://arxiv.org/abs/2505.10446)|[//]: #10/01


#### Training-Testing Input Discrepancy
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Addressing the Training-Inference Discrepancy in Discrete Diffusion for Text Generation](https://aclanthology.org/2025.coling-main.477/) <br> Masaki Asada, Makoto Miwa |<img width="1002" alt="image" src="https://tse3.mm.bing.net/th/id/OIP.gTC2EHytVcrCj7KPhAwmngHaHa?r=0&rs=1&pid=ImgDetMain&o=7&rm=3"> |[Paper](https://aclanthology.org/2025.coling-main.477/)|[//]: #07/07




### Inference Techniques

#### Unmasking Techniques
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Dream 7b](https://hkunlp.github.io/blog/2025/dream/) <br> Jiacheng Ye*, Zhihui Xie*, Lin Zheng*, Jiahui Gao*, Zirui Wu, Xin Jiang, Zhenguo Li, Lingpeng Kong |<img width="1002" alt="image" src="https://hkunlp.github.io/assets/img/2025-04-02-dream-img/model-1400.webp"> |[Paper](https://hkunlp.github.io/blog/2025/dream/)|[//]: #06/17
|[Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions](https://arxiv.org/abs/2502.06768) <br> Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham Kakade, Sitan Chen |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2502.06768v3.png"> |[Paper](https://arxiv.org/abs/2502.06768)|[//]: #10/01
|[Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding](https://arxiv.org/abs/2505.16990) <br> Runpeng Yu, Xinyin Ma, Xinchao Wang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.16990v2.png"> |[Paper](https://arxiv.org/abs/2505.16990)|[//]: #06/17
|[LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning](https://arxiv.org/abs/2505.16933) <br> Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, Chongxuan Li |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.16933v2.png"> |[Paper](https://arxiv.org/abs/2505.16933)|[//]: #06/17
|[Discrete Flow Matching](https://arxiv.org/abs/2407.15595) <br> Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve, Yossi Adi, Yaron Lipman |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2407.15595v2.png"> |[Paper](https://arxiv.org/abs/2407.15595)|[//]: #06/17
|[Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking](https://arxiv.org/abs/2505.24857) <br> Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas Nolte, Brian Karrer |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.24857v1.png"> |[Paper](https://arxiv.org/abs/2505.24857)|[//]: #10/01
|[Pc-sampler: Position-aware calibration of decoding bias in masked diffusion models](https://arxiv.org/abs/2508.13021) <br> Pengcheng Huang, Shuhao Liu, Zhenghao Liu, Yukun Yan, Shuo Wang, Zulong Chen, Tong Xiao |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2508.13021v2.png"> |[Paper](https://arxiv.org/abs/2508.13021)|[//]: #10/01
|[Plan for Speed--Dilated Scheduling for Masked Diffusion Language Models](https://arxiv.org/abs/2506.19037) <br> Omer Luxembourg, Haim Permuter, Eliya Nachmani |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2506.19037v3.png"> |[Paper](https://arxiv.org/abs/2506.19037)|[//]: #10/01


#### Remasking Techniques 
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Remasking Discrete Diffusion Models with Inference-Time Scaling](https://arxiv.org/abs/2503.00307) <br> Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, Volodymyr Kuleshov |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2503.00307v2.png"> |[Paper](https://arxiv.org/abs/2503.00307)|[//]: #06/17
|[Discrete Flow Matching](https://arxiv.org/abs/2407.15595) <br> Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve, Yossi Adi, Yaron Lipman |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2407.15595v2.png"> |[Paper](https://arxiv.org/abs/2407.15595)|[//]: #06/17
|[Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs](https://arxiv.org/abs/2507.18578) <br> Feng Hong, Geng Yu, Yushi Ye, Haicheng Huang, Huangjie Zheng, Ya Zhang, Yanfeng Wang, Jiangchao Yao |<img width="1002" alt="image" src="https://arxiv.org/html/2507.18578v2/x2.png"> |[Paper](https://arxiv.org/abs/2507.18578)|[//]: #10/01


#### Prefilling and Caching Technique
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding](https://arxiv.org/abs/2505.16990) <br> Runpeng Yu, Xinyin Ma, Xinchao Wang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.16990v2.png"> |[Paper](https://arxiv.org/abs/2505.16990)|[//]: #06/17
|[LaViDa: A Large Diffusion Language Model for Multimodal Understanding](https://arxiv.org/abs/2505.16839) <br> Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, Aditya Grover |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.16839v2.png"> |[Paper](https://arxiv.org/abs/2505.16839)|[//]: #06/17
|[dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/abs/2505.15781) <br> Xinyin Ma, Runpeng Yu, Gongfan Fang, Xinchao Wang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.15781v1.png"> |[Paper](https://arxiv.org/abs/2505.15781)|[//]: #06/17
|[dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](https://arxiv.org/abs/2506.06295) <br> Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, Linfeng Zhang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2506.06295v1.png"> |[Paper](https://arxiv.org/abs/2506.06295)|[//]: #06/17
|[Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding](https://arxiv.org/abs/2505.22618) <br> Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, Enze Xie |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.22618v3.png"> |[Paper](https://arxiv.org/abs/2505.22618)|[//]: #07/07




#### Guidance Technique
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514) <br> Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, Chongxuan Li |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2410.18514v3.png"> |[Paper](https://arxiv.org/abs/2410.18514)|[//]: #06/17
|[CtrlDiff: Boosting Large Diffusion Language Models with Dynamic Block Prediction and Controllable Generation](https://arxiv.org/abs/2505.14455) <br> Chihan Huang, Hao Tang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.14455v1.png"> |[Paper](https://arxiv.org/abs/2505.14455)|[//]: #06/17
|[TESS 2: A Large-Scale Generalist Diffusion Language Model](https://arxiv.org/abs/2502.13917) <br> Jaesung Tae, Hamish Ivison, Sachin Kumar, Arman Cohan |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2502.13917v2.png"> |[Paper](https://arxiv.org/abs/2502.13917)|[//]: #06/17
|[Energy-based diffusion language models for text generation](https://arxiv.org/abs/2410.21357) <br> Minkai Xu, Tomas Geffner, Karsten Kreis, Weili Nie, Yilun Xu, Jure Leskovec, Stefano Ermon, Arash Vahdat |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2410.21357v4.png"> |[Paper](https://arxiv.org/abs/2410.21357)|[//]: #07/07


#### Sampling Technique
|[Diffusion Language Models Know the Answer Before Decoding](https://arxiv.org/abs/2508.19982) <br> Pengxiang Li, Yefan Zhou, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Yi Liang, Soroush Vosoughi, Shiwei Liu |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2508.19982v1.png"> |[Paper](https://arxiv.org/abs/2508.19982)|[//]: #10/01
|[Thinking Inside the Mask: In-Place Prompting in Diffusion LLMs](https://arxiv.org/abs/2508.10736) <br> Xiangqi Jin, Yuxuan Wang, Yifeng Gao, Zichen Wen, Biqing Qi, Dongrui Liu, Linfeng Zhang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2508.10736v1.png"> |[Paper](https://arxiv.org/abs/2508.10736)|[//]: #10/01
|[Inference-Time Scaling of Diffusion Language Models with Particle Gibbs Sampling](https://arxiv.org/abs/2507.08390) <br> Meihua Dang, Jiaqi Han, Minkai Xu, Kai Xu, Akash Srivastava, Stefano Ermon |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2507.08390v1.png"> |[Paper](https://arxiv.org/abs/2507.08390)|[//]: #10/01
|[Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models](https://arxiv.org/abs/2508.09138) <br> Wen Wang, Bozhen Fang, Chenchen Jing, Yongliang Shen, Yangyi Shen, Qiuyu Wang, Hao Ouyang, Hao Chen, Chunhua Shen |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2508.09138v2.png"> |[Paper](https://arxiv.org/abs/2508.09138)|[//]: #10/01


#### Context Length Extension
|[LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs](https://arxiv.org/abs/2506.14429) <br> Xiaoran Liu, Zhigeng Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, Xipeng Qiu |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2506.14429v2.png"> |[Paper](https://arxiv.org/abs/2506.14429)|[//]: #10/01


#### Sparse Computation
|[Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction](https://arxiv.org/abs/2508.02558) <br> Yuerong Song, Xiaoran Liu, Ruixiao Li, Zhigeng Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, Xipeng Qiu |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2508.02558v1.png"> |[Paper](https://arxiv.org/abs/2508.02558)|[//]: #10/01
|[DPad: Efficient Diffusion Language Models with Suffix Dropout](https://arxiv.org/abs/2508.14148) <br> Xinhua Chen, Sitao Huang, Cong Guo, Chiyue Wei, Yintao He, Jianyi Zhang, Hai "Helen" Li, Yiran Chen |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2508.14148v2.png"> |[Paper](https://arxiv.org/abs/2508.14148)|[//]: #10/01
|[Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing](https://arxiv.org/abs/2508.09192) <br> Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, Zhijie Deng |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2508.09192v1.png"> |[Paper](https://arxiv.org/abs/2508.09192)|[//]: #10/01


#### Response Length Control
|[Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models](https://arxiv.org/abs/2508.00819) <br> Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang, Dahua Lin |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2508.00819v2.png"> |[Paper](https://arxiv.org/abs/2508.00819)|[//]: #10/01


### Quantization
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs](https://arxiv.org/abs/2508.14896) <br> Haokun Lin, Haobo Xu, Yichen Wu, Ziyu Guo, Renrui Zhang, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2508.14896v1.png"> |[Paper](https://arxiv.org/abs/2508.14896)|[//]: #10/01
|[DLLMQuant: Quantizing Diffusion-based Large Language Models](https://arxiv.org/abs/2508.14090) <br> Chen Xu, Dawei Yang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2508.14090v2.png"> |[Paper](https://arxiv.org/abs/2508.14090)|[//]: #10/01

### Privacy and Safety
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Every Step Counts: Decoding Trajectories as Authorship Fingerprints of dLLMs](https://arxiv.org/abs/2510.05148) <br> Qi Li, Runpeng Yu, Haiquan Lu, Xinchao Wang |<img width="1002" alt="image" src="https://chatdoc-arxiv.oss-us-west-1.aliyuncs.com/images/arxiv/2510.05148/two_page_thumbnail.jpeg?AWSAccessKeyId=LTAI5t6b2G8eTtEBczAMwjhc&Signature=RXnOoPy%2BQtD8rx6sYq3%2BXpuRnWI%3D&Expires=9223372038616549376"> |[Paper](https://arxiv.org/abs/2510.05148)|[//]: #11/01
|[The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs](https://arxiv.org/abs/2507.11097) <br> Zichen Wen, Jiashu Qu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, Linfeng Zhang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2507.11097v1.png"> |[Paper](https://arxiv.org/abs/2507.11097)|[//]: #10/01
|[Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation](https://arxiv.org/abs/2507.19227) <br> Yuanhe Zhang, Fangzhou Xie, Zhenhong Zhou, Zherui Li, Hao Chen, Kun Wang, Yufei Guo |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2507.19227v1.png"> |[Paper](https://arxiv.org/abs/2507.19227)|[//]: #10/01
|[Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position](https://arxiv.org/abs/2508.12398) <br> Zhixin Xie, Xurui Song, Jun Luo |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2508.12398v1.png"> |[Paper](https://arxiv.org/abs/2508.12398)|[//]: #10/01

### Applications

#### Text Generation and Style Control
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Fine-grained Text Style Transfer with Diffusion-Based Language Models](https://aclanthology.org/2023.repl4nlp-1.6/) <br> Yiwei Lyu, Tiange Luo, Jiacheng Shi, Todd Hollon, Honglak Lee |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2305.19512v2.png"> |[Paper](https://aclanthology.org/2023.repl4nlp-1.6/)|[//]: #06/17
|[Diffusion vs. Autoregressive Language Models: A Text Embedding Perspective](https://arxiv.org/abs/2505.15045v1) <br> Siyue Zhang, Yilun Zhao, Liyuan Geng, Arman Cohan, Anh Tuan Luu, Chen Zhao |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.15045v1.png"> |[Paper](https://arxiv.org/abs/2505.15045v1)|[//]: #06/17
|[Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](https://arxiv.org/abs/2412.11333) <br> Xiaochen Zhu, Georgi Karadzhov, Chenxi Whitehouse, Andreas Vlachos |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2412.11333v2.png"> |[Paper](https://arxiv.org/abs/2412.11333)|[//]: #06/17
|[DiffusPoll: Conditional Text Diffusion Model for Poll Generation](https://aclanthology.org/2024.findings-acl.54/) <br> Le Cheng, Shuangyin Li |<img width="1002" alt="image" src="https://aclanthology.org/thumb/2024.findings-acl.54.jpg"> |[Paper](https://aclanthology.org/2024.findings-acl.54/)|[//]: #06/17
|[PoetryDiffusion: Towards Joint Semantic and Metrical Manipulation in Poetry Generation](https://arxiv.org/abs/2306.08456) <br> Zhiyuan Hu, Chumin Liu, Yue Feng, Anh Tuan Luu, Bryan Hooi |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2306.08456v3.png"> |[Paper](https://arxiv.org/abs/2306.08456)|[//]: #06/17

#### Text Editing and Summarization
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](https://arxiv.org/abs/2502.19765) <br> Che Hyun Lee, Heeseung Kim, Jiheum Yeom, Sungroh Yoon |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2502.19765v2.png"> |[Paper](https://arxiv.org/abs/2502.19765)|[//]: #06/17
|[Discrete Diffusion Language Model for Efficient Text Summarization](https://arxiv.org/abs/2407.10998) <br> Do Huu Dat, Do Duc Anh, Anh Tuan Luu, Wray Buntine |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2407.10998v2.png"> |[Paper](https://arxiv.org/abs/2407.10998)|[//]: #06/17
|[DiffETM: Diffusion Process Enhanced Embedded Topic Model](https://arxiv.org/abs/2501.00862v1) <br> Wei Shao, Mingyang Liu, Linqi Song |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2501.00862v1.png"> |[Paper](https://arxiv.org/abs/2501.00862v1)|[//]: #06/17
|[TermDiffuSum: A Term-guided Diffusion Model for Extractive Summarization of Legal Documents](https://aclanthology.org/2025.coling-main.216/) <br> Xiangyun Dong, Wei Li, Yuquan Le, Zhangyue Jiang, Junxi Zhong, Zhong Wang |<img width="1002" alt="image" src="https://th.bing.com/th/id/OIP.4c6xMWsrQi7jGbTTe57YVwHaHa?r=0&rs=1&pid=ImgDetMain"> |[Paper](https://aclanthology.org/2025.coling-main.216/)|[//]: #06/17


#### Sentiment Analysis and Data Augmentation
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[CDAˆ2: Counterfactual Diffusion Augmentation for Cross-Domain Adaptation in Low-Resource Sentiment Analysis](https://aclanthology.org/2025.coling-main.6/) <br> Dancheng Xin, Kaiqi Zhao, Jingyun Sun, Yang Li |<img width="1002" alt="image" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR5m1jKS9eCmuL_gEFdKYmqewORHE8B71LSfg&s"> |[Paper](https://aclanthology.org/2025.coling-main.6/)|[//]: #06/17
|[An Effective Deployment of Diffusion LM for Data Augmentation in Low-Resource Sentiment Classification](https://aclanthology.org/2024.emnlp-main.109/) <br> Zhuowei Chen, Lianxi Wang, Yuben Wu, Xinfeng Liao, Yujia Tian, Junyang Zhong |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2409.03203v2.png"> |[Paper](https://aclanthology.org/2024.emnlp-main.109/)|[//]: #06/17
|[Pinpointing Diffusion Grid Noise to Enhance Aspect Sentiment Quad Prediction](https://aclanthology.org/2024.findings-acl.222/) <br> Linan Zhu, Xiangfan Chen, Xiaolei Guo, Chenwei Zhang, Zhechao Zhu, Zehai Zhou, Xiangjie Kong |<img width="1002" alt="image" src="https://th.bing.com/th/id/OIP.sYT4Wjehq3VYV9pPfSTNpwHaHa?r=0&rs=1&pid=ImgDetMain"> |[Paper](https://aclanthology.org/2024.findings-acl.222/)|[//]: #06/17

#### Knowledge and Reasoning
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models](https://arxiv.org/abs/2402.07754) <br> Jiacheng Ye, Shansan Gong, Liheng Chen, Lin Zheng, Jiahui Gao, Han Shi, Chuan Wu, Xin Jiang, Zhenguo Li, Wei Bi, Lingpeng Kong |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2402.07754v3.png"> |[Paper](https://arxiv.org/abs/2402.07754)|[//]: #06/17
|[DiffuCOMET: Contextual Commonsense Knowledge Diffusion](https://arxiv.org/abs/2402.17011) <br> Silin Gao, Mete Ismayilzada, Mengjie Zhao, Hiromi Wakaki, Yuki Mitsufuji, Antoine Bosselut |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2402.17011v2.png"> |[Paper](https://arxiv.org/abs/2402.17011)|[//]: #06/17
|[DPCL-Diff: The Temporal Knowledge Graph Reasoning based on Graph Node Diffusion Model with Dual-Domain Periodic Contrastive Learning](https://ojs.aaai.org/index.php/AAAI/article/view/33623) <br> Yukun Cao, Lisheng Wang, Luobing Huang |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2411.01477v1.png"> |[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/33623)|[//]: #06/17
|[d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2504.12216) <br> Siyan Zhao, Devaansh Gupta, Qinqing Zheng, Aditya Grover |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2504.12216v2.png"> |[Paper](https://arxiv.org/abs/2504.12216)|[//]: #06/17
|[Beyond autoregression: Discrete diffusion for complex reasoning and planning](https://arxiv.org/abs/2410.14157) <br> Jiacheng Ye, Jiahui Gao, Shansan Gong, Lin Zheng, Xin Jiang, Zhenguo Li, Lingpeng Kong |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2410.14157v3.png"> |[Paper](https://arxiv.org/abs/2410.14157)|[//]: #07/07
|[Neurosymbolic diffusion models](https://arxiv.org/abs/2505.13138) <br> Emile van Krieken, Pasquale Minervini, Edoardo Ponti, Antonio Vergari |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.13138v1.png"> |[Paper](https://arxiv.org/abs/2505.13138)|[//]: #07/07

#### Vision and Multimodal 
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[DiffVLA: Vision-Language Guided Diffusion Planning for Autonomous Driving](https://arxiv.org/abs/2505.19381) <br> Anqing Jiang, Yu Gao, Zhigang Sun, Yiru Wang, Jijun Wang, Jinghao Chai, Qian Cao, Yuweng Heng, Hao Jiang, Yunda Dong, Zongzheng Zhang, Xianda Guo, Hao Sun, Hao Zhao |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.19381v4.png"> |[Paper](https://arxiv.org/abs/2505.19381)|[//]: #06/17
|[Underwater Diffusion Attention Network with Contrastive Language-Image Joint Learning for Underwater Image Enhancement](https://arxiv.org/abs/2505.19895) <br> Afrah Shaahid, Muzammil Behzad |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.19895v1.png"> |[Paper](https://arxiv.org/abs/2505.19895)|[//]: #06/17
|[Underwater Diffusion Attention Network with Contrastive Language-Image Joint Learning for Underwater Image Enhancement](https://arxiv.org/abs/2402.14407) <br> Haoran He, Chenjia Bai, Ling Pan, Weinan Zhang, Bin Zhao, Xuelong Li |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2402.14407v4.png"> |[Paper](https://arxiv.org/abs/2402.14407)|[//]: #06/17
|[M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](https://arxiv.org/abs/2407.14502) <br> Seunggeun Chi, Hyung-gun Chi, Hengbo Ma, Nakul Agarwal, Faizan Siddiqui, Karthik Ramani, Kwonjoon Lee |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2407.14502v1.png"> |[Paper](https://arxiv.org/abs/2407.14502)|[//]: #06/17
|[AR-Diffusion: Asynchronous Video Generation with Auto-Regressive Diffusion](https://arxiv.org/abs/2503.07418) <br> Mingzhen Sun, Weining Wang, Gen Li, Jiawei Liu, Jiahui Sun, Wanquan Feng, Shanshan Lao, SiYu Zhou, Qian He, Jing Liu |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2503.07418v1.png"> |[Paper](https://arxiv.org/abs/2503.07418)|[//]: #06/17

#### Biological and Drug Discovery
[🔝 Back to Top](#quick-links)
| Title & Authors | Introduction | Links |
|:--|  :----: | :---:|
|[MolEditRL: Structure-Preserving Molecular Editing via Discrete Diffusion and Reinforcement Learning](https://arxiv.org/abs/2505.20131) <br> Yuanxin Zhuang, Dazhong Shen, Ying Sun |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.20131v1.png"> |[Paper](https://arxiv.org/abs/2505.20131)|[//]: #06/17
|[CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](https://arxiv.org/abs/2505.22869v1) <br> YJunbo Yin, Chao Zha, Wenjia He, Chencheng Xu, Xin Gao |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2505.22869v1.png"> |[Paper](https://arxiv.org/abs/2505.22869v1)|[//]: #06/17
|[Text-Guided Multi-Property Molecular Optimization with a Diffusion Language Model](https://arxiv.org/abs/2410.13597) <br> Yida Xiong, Kun Li, Jiameng Chen, Hongzhi Zhang, Di Lin, Yan Che, Wenbin Hu |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2410.13597v2.png"> |[Paper](https://arxiv.org/abs/2410.13597)|[//]: #06/17
|[GenMol: A Drug Discovery Generalist with Discrete Diffusion](https://arxiv.org/abs/2501.06158) <br> Seul Lee, Karsten Kreis, Srimukh Prasad Veccham, Meng Liu, Danny Reidenbach, Yuxing Peng, Saee Paliwal, Weili Nie, Arash Vahdat |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2501.06158v2.png"> |[Paper](https://arxiv.org/abs/2501.06158)|[//]: #06/17
|[DPLM-2: A Multimodal Diffusion Protein Language Model](https://arxiv.org/abs/2410.13782) <br> Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, Quanquan Gu |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2410.13782v1.png"> |[Paper](https://arxiv.org/abs/2410.13782)|[//]: #06/17
|[PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion](https://arxiv.org/abs/2412.17780) <br> Sophia Tang, Yinuo Zhang, Pranam Chatterjee |<img width="1002" alt="image" src="https://paper-assets.alphaxiv.org/image/2412.17780v4.png"> |[Paper](https://arxiv.org/abs/2412.17780)|[//]: #06/17

### Toolkits
* Open-dLLM: https://github.com/pengzhangzhi/Open-dLLM
* dllm-trainer: https://github.com/ZHZisZZ/dllm-trainer/tree/main
