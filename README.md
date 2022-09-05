
<!-- 
### Visturalization

| Perturbation Type | NMT     | Seed Inputs                                                                                             | Mutated Inputs                                                                                               |
|-------------|---------|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Token       | H-NLP   | Let me see.                                                                                        | 哎 me see                                                                                                    |
|             |         | I am here.                                                                                         | I▁am going.                                                                                                 |
|             |         | He is back.                                                                                        | He is 视                                                                                                     |
|             | AllenAI | configure, verify, and troubleshoot vtp                                                            | configure, bebelus, and troubleshoot vtp                                                                    |
|             |         | finger-pointing politicians and chest-beating nationalists                                         | classmates-pointing politicians and chest-beating nationalists                                              |
|             |         | in the two nations will make rational discussion nearly impossible                                 |  in the two nations will make rational discussion nearly impossible.                                        |
|             | T5      | The Commission's report sets out a detailed analysis of all the replies.                           | The Commission's report sets out a detailed analysis of all the replies Gefahr                              |
|             |         | The October European Council meeting will return to the issue of migration.                        | The October European Council meeting will return to the issue of migration not                              |
|             |         | on z/os the maximum length of the attribute is 256 bytes. on all other platforms it is 1024 bytes. | on z presenceos the maximum length of the attribute is 256 bytes. on all other platforms it is 1024 bytes.  |
| Character   | H-NLP   | You have heart disease.                                                                            | You have heart Odisease.                                                                                    |
|             |         | This is my question.                                                                               | ThWis is my question.                                                                                       |
|             |         | I'm a little confused.                                                                             | I'm a litt$le confused.                                                                                     |
|             | AllenAI | President Juncker: ‘Europe needs a genuine security union'                                         | President Juncker: ‘Europe needs a genuine security uni(on' "                                               |
|             |         | step up the cooperation on return and readmission.                                                 | step up the cooperation on return and readmis0sion.                                                         |
|             |         | In the refugee crisis, Turkey and the EU walk together and work together.                          | In Ythe refugee crisis, Turkey and Ythe EU walk togeYther and work togeYther.                               |
|             | T5      | load, performance, or scalability                                                                  | load, perf"ormance, "or scalability                                                                         |
|             |         | will hell swallow you up?                                                                          | will hell swallow $you up?                                                                                  |
|             |         | shengli oilfield used euronavy es301 to make coatings to offshore tidal zone.                      | shengli oil[field used euronavy es301 to make coatings to offshore tidal zone.                              |
|             |         |                                                                                                    |                                                                                                             |
 -->



# NMTSloth

*NMTSloth* is designed to generate test samples to test efficiency degradation of neural machine translation (NMT) systems.  Specifically, *NMTSloth* perturbs the seed sentences with three types of perturbations, and the perturbed inputs will consume more computational resources.



## Design Overview
<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/NMTSloth/blob/main/fig/overview.png" width="800" height="300" alt="Design Overview"/><br/>
</div>    

*NMTSloth* is based on two observation: (i) NMT systems' generation process are Markov processes, the number of decoder calls are undetermined (ii) existing NMT systems usually set a large threshold to avoid incomplete translation.
With the above two observation, *NMTSloth* applies the gradient guided search to generate human unnoticeable perturbations.
The design overview of *NMTSloth* is shown in the above figure. 
At high level, *NMTSloth* includes three main steps: (i) find critical tokens, (ii) input muation, and (iii) efficiency degradation detection. For the detail desgin of each steps, we refer the readers to our paper.


## File Structure
* **src** -main source codes.
  * **./src/base_attack.py** - the wrapper calss for each testing methods.
  * **./src/TranslateAPI.py** - the basical translation api files.
  * **./src/TransRepair.py** -the method of TransRepair
  * **./src/baseline_attack.py** -the implementation of Seq2Sick and NoisyError.
  * **./src/my_attack.py** -the implementation of NMTSloth.
* **spider.py** -the script is used to study the NMT systemss' configurations in HuggingFace.
* **generate_adv.py** -the script is used for generating test samples.
* **measure_latency.py** -this script measures the latency/energy consumption of the generated adversarial examples.
* **measure_loops.py**   -this script measures the iteration numbers of the generated adversarial examples.
* **measure_senstive.py** -this script measures the hyperparameter senstivelity.
* **gpuXX.sh** -bash script to run experiments (**XX** are integer numbers).


## How to run

We provide the bash script that generate adversarial examples and measure the efficiency in **gpu0.sh**. **gpu1.sh**, **gpu2.sh**,**gpu3.sh**, **gpu4.sh**. **gpu5.sh**, **gpu6.sh**, are implementing the similar functionality but for different gpus. 

So just run `bash gpu0.sh`.
 
 

## Performance Degradation Results Under Different Perfubations
<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/NMTSloth/blob/main/fig/validity.png" width="500" height="230" alt="Examples"/><br/>
</div> 

The above figure shows the degradation success ratio for different experimental subjects.

## I-Loops Under Different Beam Search Sizes
<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/NMTSloth/blob/main/fig/senstive.png" width="500" height="230" alt="Examples"/><br/>
</div> 

The above figure shows the efficiency degradation (I-Loops) for different beam search sizes.


## Generated Test Samples.

Seed Input:  **Death comes often to the soldiers and marines who are fighting in anbar province, which is roughly the size of louisiana and is the most intractable region in iraq.**

Test Input: **Death comes often to the soldiers and marines who are fighting in anbar province, which is roughly the (size of of louisiana and is the most intractable region in iraq.** 



## Random Testing Results

The following table shows the I-Loops results of random testing. Recall that NMTSloth can decrease efficiency up to 3917.91%, which is significantly more effective than random testing.



| Perturbation |  |   H-NLP     |       |  |    AllenAI    |        |   |     T5     |         | 
|--------------|-------|--------|-------|---------|--------|--------|-------|--------|---------|
|              | C     | T      | S     | C       | T      | S      | C     | T      | S       | 
| 1            | 4.18  | 6.24   | 3.21  | 3.43    | 4.09  | 2.52   | 10.09 | 13.32  | 10.32   |  
| 2            | 6.03  | 10.98  | 6.54  | 5.42    | 6.42  | 3.42   | 11.47 | 15.43  | 12.12   |   
| 3            | 12.94 | 13.56  | 9.10  | 7.31   | 8.31  | 5.31  | 16.43 | 18.42  | 17.42   | 


## Removing One Objective in Eq. 2

From the I-Loops results in the following Table, we can observe that only combing two objective can achieve better results.

| Type | Perturbation |                | H-NLP            |          |                | AllenAI          |         |                | T5               |          |
|------|--------------|----------------|------------------|----------|----------------|------------------|---------|----------------|------------------|----------|
|      |              | Minimizing EOS | Break Dependency | Combine  | Minimizing EOS | Break Dependency | Combine | Minimizing EOS | Break Dependency | Combine  |
| C    | 1            | 218.57         | 348.16           | 564.45   | 14.29          | 18.99            | 35.16   | 33.85          | 78.68            | 168.92   |
|      | 2            | 315.70         | 451.89           | 995.45   | 20.32          | 34.85            | 74.90   | 56.43          | 86.48            | 198.36   |
|      | 3            | 515.43         | 940.23           | 1357.77  | 44.18          | 44.43            | 103.36  | 89.86          | 130.12           | 205.37   |
| T    | 1            | 1000.38        | 1156.04          | 2697.77  | 10.31          | 10.12            | 24.83   | 69.56          | 170.64           | 307.27   |
|      | 2            | 1284.29        | 2031.31          | 3735.38  | 13.08          | 18.08            | 42.04   | 82.98          | 178.29           | 328.94   |
|      | 3            | 1387.42        | 2203.32          | 3917.91  | 15.53          | 36.66            | 56.75   | 121.34         | 179.96           | 328.94   |
| S    | 1            | 49.02          | 66.59            | 142.31   | 28.44          | 44.38            | 66.21   | 18.21          | 25.78            | 77.67    |
|      | 2            | 113.28         | 151.79           | 311.06   | 24.79          | 52.21            | 108.67  | 23.14          | 33.25            | 80.56    |
|      | 3            | 184.25         | 318.75           | 612.08   | 43.90          | 64.59            | 128.60  | 33.81          | 45.19            | 82.52    |




<!-- | Type | Perturbation | H-NLP          |                  |          | AllenAI        |                  |         | T5             |                  |          |   |   |   |   |   |   |   |   |   |
|------|--------------|----------------|------------------|----------|----------------|------------------|---------|----------------|------------------|----------|---|---|---|---|---|---|---|---|---|
|      |              | Minimizing EOS | Break Dependency | Combine  | Minimizing EOS | Break Dependency | Combine | Minimizing EOS | Break Dependency | Combine  |   |   |   |   |   |   |   |   |   |
| C    | 1            | 218.57         | 348.16           | 564.45   | 14.29          | 18.99            | 35.16   | 33.85          | 78.68            | 168.92   |   |   |   |   |   |   |   |   |   |
|      | 2            | 315.70         | 451.89           | 995.45   | 20.32          | 34.85            | 74.90   | 56.43          | 86.48            | 198.36   |   |   |   |   |   |   |   |   |   |
|      | 3            | 515.43         | 940.23           | 1357.77  | 44.18          | 44.43            | 103.36  | 89.86          | 130.12           | 205.37   |   |   |   |   |   |   |   |   |   |
| T    | 1            | 1000.38        | 1156.04          | 2697.77  | 10.31          | 10.12            | 24.83   | 69.56          | 170.64           | 307.27   |   |   |   |   |   |   |   |   |   |
|      | 2            | 1284.29        | 2031.31          | 3735.38  | 13.08          | 18.08            | 42.04   | 82.98          | 178.29           | 328.94   |   |   |   |   |   |   |   |   |   |
|      | 3            | 1387.42        | 2203.32          | 3917.91  | 15.53          | 36.66            | 56.75   | 121.34         | 179.96           | 328.94   |   |   |   |   |   |   |   |   |   |
| S    | 1            | 49.02          | 66.59            | 142.31   | 28.44          | 44.38            | 66.21   | 18.21          | 25.78            | 77.67    |   |   |   |   |   |   |   |   |   |
|      | 2            | 113.28         | 151.79           | 311.06   | 24.79          | 52.21            | 108.67  | 23.14          | 33.25            | 80.56    |   |   |   |   |   |   |   |   |   |
|      | 3            | 184.25         | 318.75           | 612.08   | 43.90          | 64.59            | 128.60  | 33.81          | 45.19            | 82.52    |   |   |   |   |   |   |   |   |   |
 -->

## Transferability Results
The following table shows the maximum I-Loops results of transferability. That is, we use the test inputs generated from the source model to test the target model. The results show there exists transferability in different models in terms of efficiency and suggest that black-box testing is possible.

| Source Model  | Target Model  | Perturbation Size=1     | Perturbation Size=2        | Perturbation Size=3       | 
|---------|---------|----------|----------|----------| 
| H-NLP   | AllenAI | 900.00   | 273.08   | 2400.00  | 
| H-NLP   | T5      | 1700.00  | 1700.00  | 1700.00  |  
| AllenAI | H-NLP   | 250.00   | 325.00   | 400.00   |  
| AllenAI | T5      | 1400.00  | 1400.00  | 1400.00  | 
| T5      | H-NLP   | 3733.33  | 3733.33  | 3733.33  |  
| T5      | AllenAI | 1177.78  | 1337.50  | 1542.86  | 






## Distribution of the Output Length Increament

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/NMTSloth/blob/main/fig/distribution1.png" width="1300" height="300" alt="cdf"/><br/>
</div>    

The above figure shows the Probability Density Function (PDF) and Cumulative Distribution Function (CDF) of the output length increment.
For better visualization, we use **log2(I-Loops + 100%)** as the x-axis. From the figure, we can observe that the output length increment distribution differs for NMTSloth and the comparison baselines. Specifically, NMTSloth can generate test inputs that produce super longer translations (i.e., around $2^{14}$% maximum increment for H-NLP, $2^{12}$% maximum increment for AllenAI, and $2^{10}$% maximum increment for T5) while the baselines can not.




## Relationship Between Output Length and Latency/Energy
<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/NMTSloth/blob/main/fig/study.png" width="1300" height="400" alt="cdf"/><br/>
</div>    

The above figure shows the relationship between output length and latency/energy of randomly selected seed inputs. From the results, we observe that there is a linear relationship between output length and latency/energy.


## Distribution of the Latency/Energy Increament

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/NMTSloth/blob/main/fig/new_distribution_1.png" width="1000" height="170" alt="cdf"/><br/>
</div>    

                                 Distribution of H-NLP Latency/Energy Increament

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/NMTSloth/blob/main/fig/new_distribution_2.png" width="1000" height="170" alt="cdf"/><br/>
</div>    

                                 Distribution of AllenAI Latency/Energy Increament 

<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/NMTSloth/blob/main/fig/new_distribution_3.png" width="1000" height="170" alt="cdf"/><br/>
</div>    

                                 Distribution of T5 Latency/Energy Increament

The above figure shows the Probability Density Function (PDF) and Cumulative Distribution Function (CDF) of latency/energy increament.
From the figure, we can observe that NMTSloth can generate test inputs that requires much more computational resources than the baselines.


