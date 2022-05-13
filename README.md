# GAIL-Fail
## Introduction
When GAIL Fails
* [Genarative Adversarial Imitation Learning](https://drive.google.com/drive/folders/1dzMWxlfDdd7ISSL54xLWl5W9QcS_5PAe?usp=sharing)
* Shared documents can be found [here](https://drive.google.com/drive/folders/1oqh0YBPZee6LZ-eDDqUF29NxexmIUDmR?usp=sharing).
* ~~The link to the [GAIL-Lab](https://drive.google.com/drive/folders/1lw-oqXVYCBflGoGWsmuu-2ACDAbJa2IS?usp=sharing)~~
  * ~~[Colab NoteBook](https://colab.research.google.com/drive/1kJnkAh6l_mdw0LiR8i378fIdcLdlyXa8?usp=sharing)~~
* [Intro](https://drive.google.com/drive/folders/1dzMWxlfDdd7ISSL54xLWl5W9QcS_5PAe?usp=sharing) to Imiation Learning
  * [An Algrithmic Perspective on Imiation Learning](https://drive.google.com/file/d/1XqoaPp4p8I23-VclvcBv-3BLylM16aun/view?usp=sharing)
  * [Introduction to Imiation Learning](https://drive.google.com/file/d/1FJOrce8YYeWBaJocnz-ycWQbfWc_0q_r/view?usp=sharing)
  * [Deep Reinforcement Learning](https://drive.google.com/file/d/1qzlw5vkePg7yjvgjRY0hTjQP02bhvGuC/view?usp=sharing) 


## Week2
* Week2 Meeting, `4:00PM~4:30PM, May06`, with `Dr. Mingfei Sun`.
  * `#TODO` for next week:
    * revise the lab0, draw a performance figure with all the 6 games.
      * draw a figure like this
      ![Figure3](static/Snipaste_2022-05-06_17-02-00.png)
      * two figures:`training curve` and `evaluation curve`.
      * get the figure with x-axis to be the `time step` and y-axis to be the `cumulative rewards`.
    * realize the gail1, with trpo replaced with td3
    * Pay attention to the discriminator, as different discrimimators affect lab performance hugely.
      * some papers add regulization into the discriminator.
    * Perhaps, in the future, we can directly download the sb3 and edit the package source code.
      * only in need of replacing the discriminator in the TRPO.discriminater.


## Week 1
* `Lab 0, Vanilla GAIL` &rarr; `Lab 1, DPG & DQN` &rarr; `Lab 2, Determine reward function` &rarr; `Lab 3, Non-stationary policy.`
* ~~`Lab 0` **An** [**error**](https://github.com/KangOxford/GAIL-Fail/blob/main/error) **needs to be fixed while runnning the** [**GAIL-Lab(in clolab)**](https://colab.research.google.com/drive/1kJnkAh6l_mdw0LiR8i378fIdcLdlyXa8?usp=sharing), refer to the [issue#1](https://github.com/KangOxford/GAIL-Fail/issues/1)~~
  * **Solved**
    * `Lab 0` is the original GAIL lab.
    * Refer to the new [Colab Notebook](https://drive.google.com/file/d/1osgXmgahlLzmaG8gsggkMmkUWtgG9F-S/view?usp=sharing) here
    * Here is the new [GAIL_Lab Dictionary](https://drive.google.com/drive/folders/1oDC83U29djewKynQRj4CnuuzyncbImOc?usp=sharing) 
    * `Lab 0` Successfully Running Now 
    [![Lab Successfully Running Now](https://github.com/KangOxford/GAIL-Fail/blob/main/static/Snipaste_2022-05-01_04-53-47.png?raw=true)](https://colab.research.google.com/drive/1LZDevFUyNxqgKzDm_LhrTqAUHPYYRmri?usp=sharing)
    * `Lab 0` Result 
    [![Lab Result](https://github.com/KangOxford/GAIL-Fail/blob/main/static/Snipaste_2022-05-02_04-51-23.png?raw=true)](https://colab.research.google.com/drive/1LZDevFUyNxqgKzDm_LhrTqAUHPYYRmri?usp=sharing)
    * Duration : `1.5 hour`, start from `2022-05-02 02:10:37` and end by `2022-05-02 03:34:39`. 
* `Lab 1` Next Step `#TODO`:
  * Replace the `TRPO` in `/gail/main` with `DPG & DQN` (line 90 ~ 93) 
  ```python
    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    policy = GaussianMLPPolicy(dim_state, dim_action, FLAGS.TRPO.policy_hidden_sizes, normalizer=normalizers.state)
    vfn = MLPVFunction(dim_state, FLAGS.TRPO.vf_hidden_sizes, normalizers.state)
    algo = TRPO(vfn=vfn, policy=policy, dim_state=dim_state, dim_action=dim_action, **FLAGS.TRPO.algo.as_dict())
  ```
* Network architecture can be found in the [GitHub Wiki](https://github.com/KangOxford/GAIL-Fail/wiki)
* [Week1 Slides](https://www.overleaf.com/5346254815htstspxcpchc)
[![Week1 Slides](https://github.com/KangOxford/GAIL-Fail/blob/main/static/Snipaste_2022-04-30_14-56-13.png?raw=true)](https://drive.google.com/file/d/1gg4eMApZ8NNAHndkfC_k4SHMzqTcQz3r/view?usp=sharing)

