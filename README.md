# IntegratedMotionPrediction
Integrated Data-driven Inference and Planning-based Human Motion Prediction for Safe Human-Robot Interaction

# Overall Architecture  
![algorithm_](https://github.com/HMCL-UNIST/IntegratedMotionPrediction/assets/86097031/8e75b2f7-6857-4063-b1b4-4c5a2e2c0058)

We develop the algorithm for a autonomous vehicle to safely and actively interact with an uncertain human-driven vehicle.

The algorithm combines:
  -  A hierarchical prediction strategy that integrates data-driven human internal state inference with planning-based human motion prediction
  -  An Active motion planning algorithm for the autonomous vehicle to ensure safety against uncertain human motions

## Conference Proceeding
Y. Nam and C. Kwon, “Integrated Data-driven Inference and Planning-based Human Motion Prediction for Safe Human-Robot Interaction”, ICRA 2024: International Conference on Robotics and Automation, Yokohama, Japan, May 2024

## How to train inference module
1. Generate human driving data for training  <br/>
  -> python ./gen_human_data.py <br/>
2. Train rationality inference module <br/>
  -> python ./train_inference/train_beta.py <br/>
3. Train driving style inference module  <br/>
  -> python ./train_inference/train_psi.py <br/>

## How to run main algorithm
1. Run main code <br/>
  -> python ./main.py <br/>
  ** After training the inference module, 'model_id' for each inference module should be changed accordingly.
   
