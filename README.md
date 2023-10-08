# BORex

## How to run

1. Clone this repository.
2. Setup Python environment according to [Setup](#setup) section.
3. Download [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) from Pascal VOC dataset Development Kit.
    - [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/) > `The VOC2012 Challenge` > `Development Kit`
4. Unzip .tar file, and place some directories to `RISEBO/src/` as below.
    - `Annotations` as `voc_annotation`
    - `JPEGImages` as `voc_image`
    - `SegmentationClass` as `voc_segmentation`
5. Call `python exec_borex.py` or Run `exec_borex.ipynb`.

## Setup

1. Upgrade pip.

    ```
    python -m pip3 install --upgrade pip
    ```

2. Install pytorch.

   Search .whl link according to environment at [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html).

   ```
   python -m pip install [.whl link]
   ```

   For example, `https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp310-cp310-win_amd64.whl` for `Win10 64bit`, `CUDA 11.6`, `Python3.10`.
   
3. Install other packages.

    ```
    python -m pip install -r requirements.txt
    ```

    - requirements.txt

        ```
        torchvision
        scikit-learn
        matplotlib
        pyyaml
        tqdm
        scikit-image
        opencv-python
        ```

## Confirmed environment

### Python 3.9

- Python 3.9.13
- CUDA 11.6
    - NVIDIA Driver 528.02
- pip

    ```pip freeze
    certifi==2022.12.7
    charset-normalizer==3.0.1
    colorama==0.4.6
    contourpy==1.0.7
    cycler==0.11.0
    fonttools==4.38.0
    idna==3.4
    imageio==2.25.0
    joblib==1.2.0
    kiwisolver==1.4.4
    matplotlib==3.6.3
    networkx==3.0
    numpy==1.24.1
    opencv-python==4.7.0.68
    packaging==23.0
    Pillow==9.4.0
    pyparsing==3.0.9
    python-dateutil==2.8.2
    PyWavelets==1.4.1
    PyYAML==6.0
    requests==2.28.2
    scikit-image==0.19.3
    scikit-learn==1.2.1
    scipy==1.10.0
    six==1.16.0
    threadpoolctl==3.1.0
    tifffile==2023.1.23.1
    torch @ https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp39-cp39-win_amd64.whl
    torchvision==0.14.1
    tqdm==4.64.1
    typing_extensions==4.4.0
    urllib3==1.26.14
    ```

### Python 3.10

- Python 3.10.8
- CUDA 11.6
    - NVIDIA Driver 528.02
- pip

    ```
    certifi==2022.12.7
    charset-normalizer==3.0.1
    colorama==0.4.6
    contourpy==1.0.7
    cycler==0.11.0
    fonttools==4.38.0
    idna==3.4
    imageio==2.25.0
    joblib==1.2.0
    kiwisolver==1.4.4
    matplotlib==3.6.3
    networkx==3.0
    numpy==1.24.1
    opencv-python==4.7.0.68
    packaging==23.0
    Pillow==9.4.0
    pyparsing==3.0.9
    python-dateutil==2.8.2
    PyWavelets==1.4.1
    PyYAML==6.0
    requests==2.28.2
    scikit-image==0.19.3
    scikit-learn==1.2.1
    scipy==1.10.0
    six==1.16.0
    threadpoolctl==3.1.0
    tifffile==2023.1.23.1
    torch @ https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp310-cp310-win_amd64.whl
    torchvision==0.14.1
    tqdm==4.64.1
    typing_extensions==4.4.0
    urllib3==1.26.14
    ```