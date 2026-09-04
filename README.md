# AIAYN: Attention Is All You Need (PyTorch Implementation from Scratch)

A complete, production-grade PyTorch implementation of the landmark paper **"Attention Is All You Need" (Vaswani et al., NIPS 2017)**.

## 🚀 Key Highlights

- **Pure PyTorch Implementation**: Built without high-level wrappers (`nn.Transformer` or `nn.MultiheadAttention`) to showcase deep architectural understanding.
- **Dual Model Support**:
  - **Seq2Seq Encoder-Decoder Transformer**: Full paper architecture for Sequence-to-Sequence tasks.
  - **Causal Decoder-Only Transformer (GPT)**: Modern autoregressive decoder language model.
- **AIAYN Optimization**: Custom **Noam Learning Rate Warmup Scheduler** ($lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})$).
- **Visualizations**: Multi-head self-attention heatmap visualization across attention heads.
- **Inference Engine**: Autoregressive sampling supporting Greedy decoding, Temperature scaling, and Top-K sampling.

## 📂 Repository Structure

- `AIAYN_1.ipynb`: Complete portfolio Jupyter Notebook with LaTeX equations, modular code, training loop, convergence plots, and text generation engine.
- `data/wizard_of_oz.txt`: Dataset used for character-level language modeling.

## 🛠️ Quick Start

```bash
pip install -r requirements.txt
jupyter notebook AIAYN_1.ipynb
```
