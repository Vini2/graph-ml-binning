# graph-ml-binning

Comparing different graph machine learning techniques for metagenomic binning.

This project explores graph-based and embedding-based approaches for grouping metagenomic contigs into genome bins. It uses SPAdes assembly graph data, contig sequence features, coverage information, marker-gene constraints, clustering, and evaluation metrics.

## Methods

The notebooks cover several graph ML approaches, including:

- GCN
- GraphSAGE
- GAT
- APPNP
- DeepWalk + Node2Vec
- DGI
- VGAE
- GraphCL
- Node2Vec
- LINE
- Laplacian Eigenmaps
- GCNII

## Repository Structure

```text
notebooks/        Experiment notebooks and shared utilities
tests/data/       Example SPAdes graph, contigs, markers, paths, and labels
environment.yml   Conda environment definition
