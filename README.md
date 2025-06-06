# SmoothMoCapRetarget
SmoothMoCapRetarget converts human motion capture data into smooth, kinematically feasible trajectories for humanoid robots. Built on Levenberg-Marquardt optimization, it ensures natural motion transitions while respecting joint limits and maintaining temporal smoothness across the entire sequence.

## Python Environment

```cmd
conda create -n smooth-mocap-retarget python=3.8
conda activate smooth-mocap-retarget
pip install -r requirements.txt
```

## Recourse Preparation

### AMASS Dataset Preparation
Download [AMASS Dataset](https://amass.is.tue.mpg.de/index.html) with `SMPL + H G` format and put it under `smooth-mocap-retarget/data/AMASS/AMASS_Complete/`:
```
|-- smooth-mocap-retarget
   |-- data
      |-- AMASS
         |-- AMASS_Complete 
               |-- ACCAD.tar.bz2
               |-- BMLhandball.tar.bz2
               |-- BMLmovi.tar.bz2
               |-- BMLrub.tar
               |-- CMU.tar.bz2
               |-- ...
               |-- Transitions.tar.bz2

```

And then `cd smooth-mocap-retarget/data/AMASS/AMASS_Complete` extract all the motion files by running:
```
for file in *.tar.bz2; do
    tar -xvjf "$file"
done
```

Then you should have:
```
|-- smooth-mocap-retarget
   |-- data
      |-- AMASS
         |-- AMASS_Complete 
               |-- ACCAD
               |-- BioMotionLab_NTroje
               |-- BMLhandball
               |-- BMLmovi
               |-- CMU
               |-- ...
               |-- Transitions

```

### Occlusion Label Preparation

Download [Occlusion Labels](https://drive.google.com/uc?id=1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc) and put the `amass_copycat_occlusion_v3.pkl` file under `smooth-mocap-retarget/data/occlusion/`, then you should have:
```
|-- smooth-mocap-retarget
   |-- data
      |-- occlusion
         |-- amass_copycat_occlusion_v3.pkl
```

### Text Label Preparation

Extract `texts.zip` into `smooth-mocap-retarget/data/` directory, and you should have:
```
|-- smooth-mocap-retarget
   |-- data
      |-- texts
         |-- 000000.txt
         |-- 000001.txt
         |-- ...
         |-- 014615.txt
         |-- M000000.txt
         |-- ...
         |-- M014615.txt
```

## SMPL Model Preparation

Download [SMPL](https://smpl.is.tue.mpg.de/download.php) with `pkl` format and put it under `smooth-mocap-retarget/data/smpl/`, and you should have:
```
|-- smooth-mocap-retarget
   |-- data
      |-- smpl
         |-- SMPL_python_v.1.1.0.zip
```

Then `cd smooth-mocap-retarget/data/smpl` and  `unzip SMPL_python_v.1.1.0.zip`, you should have 
```
|-- smooth-mocap-retarget
   |-- data
      |-- smpl
         |-- SMPL_python_v.1.1.0
            |-- models
               |-- basicmodel_f_lbs_10_207_0_v1.1.0.pkl
               |-- basicmodel_m_lbs_10_207_0_v1.1.0.pkl
               |-- basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
            |-- smpl_webuser
            |-- ...
```
Rename these three pkl files and move it under smpl like this:
```
|-- smooth-mocap-retarget
   |-- data
      |-- smpl
         |-- SMPL_FEMALE.pkl
         |-- SMPL_MALE.pkl
         |-- SMPL_NEUTRAL.pkl
```

### Robot Model Preparation

Put the model description of your robot under `smooth-mocap-retarget/resources/robots`.
Take `g1` robot for example, put `g1_29dof.xml` under `smooth-mocap-retarget/resources/robots/g1`

## Retargeting Procedure

### Shape Fitting

Create the directory to store the fitted shape, and run the shape fitting script to align the standard smpl model with your own robot model.
Take `g1` robot for example
- Create the `smooth-mocap-retarget/data/g1` directory
- Run `smooth-mocap-retarget/scripts/g1/fit_robot_shape.py`, the fitted shape will be saved as `smooth-mocap-retarget/data/g1/fit_robot_shape_g1.pkl`

### Retargeting based on Fitted Shape

Run the motion retargeting script.
Take `g1` robot for example
- Run `smooth-mocap-retarget/scripts/g1_lockwaist/process_humanml3d_g1.py`, the retargeted dataset will be saved under `smooth-mocap-retarget/data/g1/`