# Multi-metrics

This codebase is the official PyTorch implementation of our paper:

***Multi-metrics adaptively identifies backdoors in Federated learning***

>**Abstract.** The decentralized and privacy-preserving nature of federated learning (FL) makes it vulnerable to backdoor attacks aiming to manipulate the behavior of the resulting model on specific adversary-chosen inputs. However, most existing defenses based on statistical differences take effect only against specific attacks, especially when the malicious gradients are similar to benign ones or the data are highly non-independent and identically distributed (non-IID). In this paper, we revisit the distance-based defense methods and discover that i) Euclidean distance becomes meaningless in high dimensions and ii) malicious gradients with diverse characteristics cannot be identified by a single metric. To this end, we present a simple yet effective defense strategy with multi-metrics and dynamic weighting to identify backdoors adaptively. Furthermore, our novel defense has no reliance on predefined assumptions over attack settings or data distributions and little impact on the benign performance. To evaluate the effectiveness of our approach, we conduct comprehensive experiments on different datasets under various attack settings, where our method achieves the best defense performance. For instance, we achieve the lowest backdoor accuracy of $3.06%$ under the difficult Edge-case PGD, showing significant superiority over previous defenses. The results also demonstrate that our method can be well-adapted to a wide range of non-IID degrees without sacrificing the benign performance. 

## Citation
If you find our work useful in your research, please consider citing:
```
@article{huang2023multi,
  title={Multi-metrics adaptively identifies backdoors in Federated learning},
  author={Huang, Siquan and Li, Yijiang and Chen, Chong and Shi, Leyu and Gao, Ying},
  journal={arXiv preprint arXiv:2303.06601},
  year={2023}
}
```

## Results

The source results of various defenses against Edge-case PGD attack are in `log`.

The table shows the robustness of our approach compared to the SOTA defenses for various challenging attacks.

|  defenses  | Model  Replacement |      DBA       |      PGD       |  Edge-case PGD  |
| :--------: | :----------------: | :------------: | :------------: | :-------------: |
|            |       MA/BA        |     MA/BA      |     MA/BA      |      MA/BA      |
|   FedAvg   |  **86.95**/64.80   |  79.23/90.44   |  87.04/14.44   |   87.14/55.10   |
|    RFA     |    86.69/25.56     |  79.60/57.69   | **87.1**/52.56 |   86.47/65.31   |
| Foolsgold  |     85.71/6.67     |   77.56/3.43   |  84.92/14.44   |   84.76/51.53   |
|    Krum    |     82.17/6.11     |   78.18/6.01   |  82.32/66.67   |   81.23/59.18   |
| Multi-Krum |     86.55/1.67     |  79.33/91.39   |  86.52/17.78   | **87.40**/60.20 |
|  Weak-DP   |     74.41/46.1     |    10/**0**    |  74.43/22.22   |   73.84/53.06   |
|   Flame    |   80.58/**0.56**   |  76.78/37.24   | 81.24/**0.56** |   81.41/5.12    |
|    Ours    |   86.34/**0.56**   | **79.61**/9.98 | 86.44/**0.56** | 86.86/**3.06**  |

 The figure shows MA(%) and BA(%) of various defenses under Edge-case PGD attack.

![cifar_compare](fig/cifar.PNG)

## Getting Started

```bash
cd multi-metrics/codebase
conda create -n multi-metrics python=3.6.9
conda activate multi-metrics
pip install -r requirements.txt
```

## Data preparation

The backdoor dataset SouthWest Airlines for CIFAR10 is in the project. To get backdoor dataset Ardis for MNIST/EMNIST please see **[edgecase_backdoors](https://github.com/SanaAwan5/edgecase_backdoors)**, which is the official code of the Edge-case attack. 

## Usage

#### example command

The follow command is to conduct Multi-metrics (ous) defense to defend against Edge-case PGD attack on CIFAR10. Please see **[edgecase_backdoors](https://github.com/SanaAwan5/edgecase_backdoors)** to get detailed description on the arguments of FL system.

```bash
python simulated_averaging.py \
--lr 0.02 \
--gamma 0.998 \
--num_nets 200 \
--fl_round 1500 \
--part_nets_per_round 10 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset cifar10 \
--model vgg9 \
--fl_mode fixed-freq \
--defense_method multi-metrics \
--attack_method pgd \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--stddev 0.0025 \
--eps 2 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type southwest \
--norm_bound 2 \
--device=cuda\
```


## Acknowledgement

We thank **[edgecase_backdoors](https://github.com/SanaAwan5/edgecase_backdoors)** for their amazing open-sourced project! We just add our Multi-metrics defense to this project.

