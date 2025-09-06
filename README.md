# RecGraph: Athletic Recovery Prediction using Graph Neural Networks

Graph neural network framework for predicting athletic recovery using multi-modal physiological and subjective data.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download data:**
   Download [PMData](https://datasets.simula.no/downloads/pmdata.zip), unzip it and place in the same directory as the notebook.

3. **Run the model:**
   ```bash
   jupyter notebook graphrec.ipynb
   ```

## Results

### Without Pretraining
- **Dataset:** 1651 recovery graphs from 1747 samples across 16 participants
- **Architecture:** Input dim: 9, Hidden dim: 128, Parameters: 922,577

| Metric | MAE | RMSE | Correlation |
|--------|-----|------|-------------|
| **Readiness** | 0.0322 | 0.0423 | 0.9952 |
| **Quality** | 0.0173 | 0.0244 | 0.9717 |
| **Training** | 0.0484 | 0.1035 | 0.9009 |
| **Overreach** | 0.3434 | 0.3737 | 0.6714 |

**Overall Performance:** Average Correlation: 0.8848, Average MAE: 0.1103

### With Pretraining
- **Architecture:** Input dim: 9, Hidden dim: 128, Parameters: 674,126
- **Training:** 30 epochs contrastive pre-training + 150 epochs fine-tuning

| Metric | MAE | RMSE | Correlation |
|--------|-----|------|-------------|
| **Readiness** | 0.0578 | 0.0763 | 0.9682 |
| **Quality** | 0.0095 | 0.0136 | 0.9898 |
| **Training** | 0.0907 | 0.1291 | 0.8612 |
| **Overreach** | 0.3907 | 0.4249 | 0.5268 |

**Overall Performance:** Average Correlation: 0.8365, Average MAE: 0.1372

## Model Files

- `graphrec_model_wo_pretraining.pth` - Model trained without pretraining
- `recgraph_model_w_pretraining.pth` - Model trained with contrastive pretraining
- `processed_graphs_cache.pkl` - Cached processed graph data