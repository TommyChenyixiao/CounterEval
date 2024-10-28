# GAT-Soccer-Counter-Attack
Attention-Based GNNs and Graph Transformers with GNNExplainer for Graph Classification in Soccer Counterattack Analysis

## Setting up the Environment

To create and set up a new Conda environment named `casestudy`, follow these steps:

1. **Create the Conda environment:**

   ```
   conda create -n casestudy python=3.10 -y
   ```
   
2. **Activate the environment:**

   ```
   conda activate casestudy
   ```

3. **Install dependencies:**
  
   ```
   pip install -r requirements.txt
   ```

## Project Overview

This project explores the integration of attention-based Graph Neural Networks (GNNs) and graph transformers for classifying soccer counterattacks. These models aim to capture complex interactions and dependencies, using GNNExplainer for enhanced interpretability. The evaluation will focus on accuracy, log-loss, and interpretability.

## Methodology

### Data Description

The dataset contains 20,863 graphs from MLS, NWSL, and international women's soccer matches, with features like positional coordinates, velocity, and distance to goal. Each graph is labeled based on whether the counterattack was successful.

![Sample Graph Visualization](<sample_graph.png>)

![Sample Graph Visualization in Soccer Field](<soccer_graph.png>)


### Proposed Architectures

- **Attention-Based GNN:** 
   - Utilizes attention layers to prioritize significant nodes and edges, improving feature aggregation for capturing complex interactions.

- **Graph Transformers:** 
   - Adapts self-attention layers to handle both local and global dependencies. Incorporates positional encoding and a feedforward neural network for final classification.

### Interpretability with GNNExplainer

GNNExplainer will provide insights into which nodes, edges, and features contribute most to predictions, offering explanations for different aspects of successful counterattacks.

### Training and Evaluation

- The data will be split into training (70%), validation (15%), and test sets (15%).
- Cross-entropy loss will be used for training, with accuracy and log-loss as primary evaluation metrics. GNNExplainer will be applied to analyze feature contributions.

## Expected Outcomes

- Improved accuracy in graph classification over traditional GNN models.
- Enhanced interpretability through GNNExplainer, offering detailed insights into influential graph components.

## Timeline

- **Phase 1:** Literature Review & Data Preparation (Oct. 21 - Oct. 27)
- **Phase 2:** Model Development (Oct. 28 - Nov. 10)
- **Phase 3:** GNNExplainer Integration & Training (Nov. 11 - Nov. 24)
- **Phase 4:** Report Writing & Visualization (Nov. 25 - Dec. 1)

## References

1. Sahasrabudhe, A., & Bekkers, J. (2023). _A Graph Neural Network Deep-Dive into Successful Counterattacks_. In 17th Annual MIT Sloan Sports Analytics Conference.

   
