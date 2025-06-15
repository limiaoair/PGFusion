# **PGFusion: Prototype-contrastive Interaction and Group Selective-driven Fusion for Co-salient Object Detection**


## Network Architecture
![fig1.png](figs/fig2.jpg)

## Results and Saliency maps
We perform quantitative comparisons and qualitative comparisons with 12 RGB-D SOD
methods on six RGB-D datasets.
![fig2.jpg](figs/fig3.jpg)
![fig3.jpg](figs/fig4.jpg)

### Prerequisites
- Python 3.6
- Pytorch 1.10.2
- Torchvision 0.11.3
- Numpy 1.19.2

  install SSM
   ```
  conda create -n vmunet python=3.8
  conda activate vmunet
  pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
  pip install packaging
  pip install timm==0.4.12
  pip install pytest chardet yacs termcolor
  pip install submitit tensorboardX
  pip install triton==2.0.0
  pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
  pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
  pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
   ```

### Datasets
 Put the [CoCo-SEG](https://drive.google.com/file/d/1GbA_WKvJm04Z1tR8pTSzBdYVQ75avg4f/view), [CoCA](http://zhaozhang.net/coca.html), [CoSOD3k](http://dpfan.net/CoSOD3K/) and [Cosal2015](https://drive.google.com/u/0/uc?id=1mmYpGx17t8WocdPcw2WKeuFpz6VHoZ6K&export=download) datasets to `DCFM/data` as the following structure:
  ```
  PGFusion
     ├── other codes
     ├── ...
     │ 
     └── data
           
           ├── CoCo-SEG (CoCo-SEG's image files)
           ├── CoCA (CoCA's image files)
           ├── CoSOD3k (CoSOD3k's image files)
           └── Cosal2015 (Cosal2015's image files)
  ```


### Contact
Feel free to send e-mails to me (lmiao@tongji.edu.cn).

## Relevant Literature

```text
coming soon
```
