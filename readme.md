# CloudMotionWind

This program is aiming to retrieve accurate upper level wind based on two successive radar/satellite observation. 
The feature detection is based on SIFT (Scale-Invariant Feature Transform). This method can account for the warp (convergence/divergence) 
and rotation of cloud clusters.

## Step 1: target points generation

Generate feature points for both images：

![featpnt](https://github.com/wangminzheng/CloudMotionWind/blob/master/results/img_keyp.png)

## Step 2: target points matching

Match target points in both images, considering the consistency of wind vector in the whole field.

![featpnt](https://github.com/wangminzheng/CloudMotionWind/blob/master/results/matched_right.png)

## Step 3: calculate wind vector 

Draw vecotr based on the position shift of the matched points.

![featpnt](https://github.com/wangminzheng/CloudMotionWind/blob/master/results/speed_arrow.png)

