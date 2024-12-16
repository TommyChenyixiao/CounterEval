# CounterEval: GNN-Based Predictive Modeling and Performance Evaluation for Soccer Counterattacks Using Tracking Data

TODO: Put a countereval .gif here.

## Description

Counterattacking is an effective strategy for scoring in modern football, championed by renowned coaches like José Mourinho, Diego Simeone, and Antonio conte. A successful counterattack is often executed by key players, making it valuable to develop metrics that evaluate individual contributions to the play. The analysis of soccer player movements and team tactics requires sophisticated computational approaches that can capture both individual dynamics and collective behavior. This study introduces **CounterEval**, a deep learning framework for evaluating player performance in counterattacks using tracking data from 632 games across MLS (2022), NWSL (2022), and international women’s soccer (2020 to 2022). The dataset includes detailed information on player and ball locations, velocity, and acceleration, and was refined through our quality improvements, such as correcting mislabeled counterattacks and filling missing player IDs. The core of the framework comprises two neural network models: a movement model, a variational autoencoder model fitting the distribution of a player’s next location given the spatial positions of all players, and a graph-based counterattacking success model, which estimates the probability of a successful counterattack given the current game context. By combining the two models, we develop a real-time evaluation metric that quantifies player performance during counterattacks. The framework is robust and scalable, with potential for further enhancement through higher-quality data and model fine-tuning.

## Reproduce our work

### Virtual Environment

```
conda env create -f environment.yml
conda activate sds625_soccer
```

### Run All Results

You can get all results of this project by running one of the two following commands.

```
Rscript scripts/Main.R # Option 1
# bash scripts/Main.sh # Option 2
```

### Data

#### Raw Datasets

The source of the raw dataset is [ussf_ssac_23_soccer_gnn](https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn). They represent each frame of tracking data as a heterogeneous graph, where nodes correspond to individual players (offensive and defensive) and edges represent the spatial and temporal relationships between players.

* Edge Features:
  * Player Distance - Distance between two players connected to each other
  * Speed Difference - Speed difference between two players connected to each other
  * Positional Sine angle - Sine of the angle created between two players in the edge
  * Positional Cosine angle - Cosine of the angle created between two players in the edge
  * Velocity Sine angle - Sine of the angle created between the velocity vectors of two players in the edge
  * Velocity Cosine angle - Coine of the angle created between the velocity vectors of two players in the edge
  
* Node Features:
  * x coordinate - x coordinate on the 2D pitch for the player / ball
  * y coordinate - y coordinate on the 2D pitch for the player / ball
  * vx - Velocity vector's x coordinate
  * vy - Velocity vector's y coordinate
  * Velocity - magnitude of the velocity
  *  Velocity Angle - angle made by the velocity vector
  * Distance to Goal - distance of the player from the goal post
  * Angle with Goal - angle made by the player with the goal
  * Distance to Ball - distance from the ball (always 0 for the ball)
  * Angle with Ball - angle made with the ball (always 0 for the ball)
  * Attacking Team Flag - 1 if the team is attacking, 0 if not and for the ball
  * Potential Receiver - 1 if player is a potential receiver, 0 otherwise

The Graph configuration is depicted by the graph below:

![Graph Configuration](img/graph.png)

(source: [ussf_ssac_23_soccer_gnn](https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn))

However, the data quality of this dataset is very low. Most other soccer tracking data is not free in the market and we advocate for more open access to high-quality soccer tracking data to facilitate research and development in this field.

To download all raw datasets for our project, run the following command.

```
bash scripts/get_raw_dataset.sh
```

#### Data Cleaning

### Visualization

We also provide a streamlit app to visualize the counterattack process to facilitate our EDA. 

```
streamlit run scripts/visualize_helper.py
```

### Methodology

Please refer to our report for interpretation of CounterEval framework in details.

## Acknowledgement 

## References
1. Sahasrabudhe, A., & Bekkers, J. (2023). _A Graph Neural Network Deep-Dive into Successful Counterattacks_. In 17th Annual MIT Sloan Sports Analytics Conference.
2. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2017). Graph Attention Networks. arXiv, 1710.10903.


