## WWW2026 Submission 
## Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement

This repository contains partial implementation code for the paper "Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement" (submitted to WWW 2026).
The repo includes core modules and experiment scripts necessary for helping the review process.
Complete codes along with full training pipelines, will be open-sourced upon acceptance of the submission.

## For all the baselines benchmarked under compound graph deficiencies in this paper, please refer to their corresponding papers and official code repositories for details.
ACM-GNN: Luan, Sitao, et al. "Revisiting heterophily for graph neural networks." Advances in neural information processing systems 35 (2022): 1362-1375.
DEEP GRAPH INFOMAX: Veličković, Petar, et al. "Deep graph infomax." arXiv preprint arXiv:1809.10341 (2018).
DropEdge: Rong, Yu, et al. "Dropedge: Towards deep graph convolutional networks on node classification." arXiv preprint arXiv:1907.10903 (2019).
DropMessage: Fang, Taoran, et al. "Dropmessage: Unifying random dropping for graph neural networks." Proceedings of the AAAI conference on artificial intelligence. Vol. 37. No. 4. 2023.
Feature Propagation: Rossi, Emanuele, et al. "On the unreasonable effectiveness of feature propagation in learning on graphs with missing node features." Learning on graphs conference. PMLR, 2022.
NeiborMean
GRCN: Yu, Donghan, et al. "Graph-revised convolutional network." Joint European conference on machine learning and knowledge discovery in databases. Cham: Springer International Publishing, 2020.
LLM4NG: Yu, Jianxiang, et al. "Leveraging large language models for node generation in few-shot learning on text-attributed graphs." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 12. 2025.
RSGNN: Dai, Enyan, et al. "Towards robust graph neural networks for noisy graphs with sparse labels." Proceedings of the fifteenth ACM international conference on web search and data mining. 2022.
TAPE: He, Xiaoxin, et al. "Harnessing explanations: Llm-to-lm interpreter for enhanced text-attributed graph representation learning." arXiv preprint arXiv:2305.19523 (2023).
TAE: He, Xiaoxin, et al. "Harnessing explanations: Llm-to-lm interpreter for enhanced text-attributed graph representation learning." arXiv preprint arXiv:2305.19523 (2023).
SGL: Wu, Jiancan, et al. "Self-supervised graph learning for recommendation." Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval. 2021.
SimGCL: Liu, Cheng, et al. "SimGCL: graph contrastive learning by finding homophily in heterophily." Knowledge and Information Systems 66.3 (2024): 2089-2114.

## Highlights

- **Synthetic Node Generator**: `SGGM.py` produces category-aware papers with an LLM + RAG feedback loop.
- **Keyword-Aware Encoding**: `weighted_encode.py` fuses document content and keyword embeddings with configurable weights.
- **Contrastive Embedding Refinement**: `R2CL.py` jointly optimizes original and synthetic node embeddings using a supervised contrastive objective with periodic RAG augmentation.
- **GNN Evaluation Suite**: `RoGRAD_GCN.py`, `RoGRAD_GAT.py`, and `RoGRAD_sage.py` benchmark multiple backbones under diverse perturbation settings.

---

## Quick Start

### 0. Environment

```
python==3.8.0
torch==1.12.0
numpy==1.24.3
scikit_learn==1.1.1
torch-cluster==1.6.0
torch-geometric==2.3.1
torch-scatter==2.1.0
torch-sparse==0.6.16
torch-spline-conv==1.2.1
```

### 1. Configure LLM Access

- **Closed-source APIs**: `SGGM.py` and `R2CL.py` expect an OpenAI-compatible key. Replace the hard-coded tokens with environment variables or your own configuration before running.
- **Local SentenceTransformer**: Update `weighted_encode/weighted_encode.py` to point `model_path` to your local SentenceTransformer checkpoint (e.g., `os.path.join(repo_root, "model_big")`).

### 2. End-to-End Pipeline

1. **Generate synthetic papers**  
   ```bash
   python SGGM.py
   ```  


2. **Encode content & keywords with weights**  
   ```bash
   cd weighted_encode
   python weighted_encode.py
   cd ..
   ```  

3. **Contrastive refinement of embeddings**  
   ```bash
   python R2CL.py
   ```  

4. **Train downstream GNNs**  
   ```bash
   python RoGRAD_GCN.py --dataset pubmed --model_type Node
   ```  
   Each run sweeps node/edge/feature drop ratios and logs results to `results/*.csv`.


---

## Code Structure

```
Ours/
├── Dataset
├── SGGM.py                # LLM-driven synthetic paper generator with similarity feedback
├── weighted_encode/
│   └── weighted_encode.py # Weighted SentenceTransformer encoding
├── R2CL.py                # Supervised contrastive refinement + RAG augmentation
├── load_pubmed.py         # Data preparation pipeline for PubMed
├── load_cora.py           # Equivalent pipeline for Cora
├── RoGRAD_GCN.py          # GCN backbone with augmentation sweeps
├── RoGRAD_GAT.py          # GAT backbone
├── RoGRAD_sage.py         # GraphSAGE backbone
├── utils.py               # Argument parser with shared hyperparameters
└── results/               # CSV logs produced by experiments
```

