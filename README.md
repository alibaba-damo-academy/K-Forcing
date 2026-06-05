# K-Forcing: Joint Next-K-Token Decoding via Push-Forward Language Modeling

## Authors

Zhiwei Tang<sup>1,2,3,*</sup>, Yuanyu He<sup>3,*</sup>, Yizheng Han<sup>1</sup>, Wangbo Zhao<sup>4</sup>, Jiasheng Tang<sup>1,2</sup>, Fan Wang<sup>1</sup>, Bohan Zhuang<sup>1,3</sup>

<sup>1</sup> DAMO Academy, Alibaba Group &nbsp; <sup>2</sup> Hupan Lab &nbsp; <sup>3</sup> Zhejiang University &nbsp; <sup>4</sup> The Hong Kong University of Science and Technology

<sup>*</sup> Equal contribution

## Description

K-Forcing is a push-forward language modeling paradigm for **joint next-k-token decoding**. It distills an existing autoregressive (AR) model into a conditional push-forward mapping that transforms independent uniform noise variables into a joint sample of multiple future tokens in a single forward pass. This design preserves fixed-length outputs, reuses the AR backbone architecture, and enables significant inference speedup under high-load batch serving — the scenario most critical for industrial-scale deployment.

## TODO

- [ ] Arxiv paper release
- [ ] Code release

## Contact

- Zhiwei Tang: [mcstzw@gmail.com](mailto:mcstzw@gmail.com)
- Bohan Zhuang: [bohan.zhuang@gmail.com](mailto:bohan.zhuang@gmail.com)
