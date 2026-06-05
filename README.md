# K-Forcing: Joint Next-K-Token Decoding via Push-Forward Language Modeling

## Description

K-Forcing is a push-forward language modeling paradigm for **joint next-k-token decoding**. It distills an existing autoregressive (AR) model into a conditional push-forward mapping that transforms independent uniform noise variables into a joint sample of multiple future tokens in a single forward pass. This design preserves fixed-length outputs, reuses the AR backbone architecture, and enables significant inference speedup under high-load batch serving — the scenario most critical for industrial-scale deployment.

## TODO

- [ ] Arxiv paper release
- [ ] Code release
