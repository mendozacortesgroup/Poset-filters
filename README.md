# Poset-filters

This reposity contains the code of the paper "Order theory in the context of machine learning: an application"

## Contents


* **Overview**
  - Poset-filters are a family of convolutional filters, sometimes they produce better accuracy than average pooling, max pooling or mix pooling.

    
 
* **Example Usage**: 
For the poset N:
```python
from poset_utils import filter_n 
#on the init section of the NN
self.poset_pool = filter_n

#on the forward section of the NN
out= self.poset_pool(out)
```
   
For another poset (for example the number 12):
```python
from poset_utils import PosetFilter,  dict_posets
#on the init section of the NN
self.poset_pool = PosetFilter(coeffs=dict_posets[12])

#on the forward section of the NN
out= self.poset_pool(out)
```
The index of the poset is the row in the following table, starting from 0 (the slowest) to 15 (the fastest):

  ![standart](img/table.png)

1 corresponds to the disjoint union of points/cube.

13 corresponds to the four chain/simplex.


* **Test**
To run the PyTorch test run
```bash
 python -m unittest  test/test_pytorch.py
```




* **Getting Started**
  - Clone this repo:
 
    git clone https://github.com/mendozacortesgroup/Poset-filters.git

    cd Poset_filters

  - prerequisites

    python >=3.7

    pytorch >= 2.2.0

  - location of:
    - code: [poset_filters PyTorch](poset_utils.py)
    - issue tracker : [report issues](https://github.com/mendozacortesgroup/Poset-filters/issues)



* **Notes**
  - version : v1.0
  - If the input has odd dimentions, the code automatically adds padding one on the right and/or bottom.
  - The combination ReLU followed by a poset filter seems to work well.
  - The seeds used in experiments were:
```python
import torch
import numpy as np
import random

for seed in  [2, 315, 568, 6664, 32168, 35156, 351646, 789465, 798648, 4861351, 8465864, 9876568, 6567979, 83115846]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(seed)
    #if cuda:
    torch.cuda.manual_seed(seed)
    trainLoader = DataLoader(
        dset.Somedataset(root='data', train=True, download = True,
        transform=trainTransform),
        batch_size=batchSz, shuffle=True, worker_init_fn=seed_worker, generator=g,  **kwargs) #Note the seed_worker and generator
    testLoader = DataLoader(
        dset.Somedataset(root='data', train=False, download = True,
        transform=testTransform),
        batch_size=batchSz, shuffle=False, worker_init_fn=seed_worker, generator=g, **kwargs)
```


* **Colophon**
  - Credits -- code, algorithm, implementation/deployment, testing and and overall direction: Eric Dolores Cuenca, Aldo Guzman-Saenz and Susana Lopez Moreno. Principal Investigator: Jose L. Mendoza-Cortes and Sangil Kim.  
  - Copyright and License -- see [LICENSE](somefile) file.
  - How to contribute: submit questions or issues.
  - This project has received funding from the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (2022R1A5A1033624, 2021R1A2B5B03087097).
  - This work was supported in part through computational resources and services provided by the Institute for Cyber-Enabled Research at Michigan State University
  - References:  https://arxiv.org/abs/
  
* **Citation**
If you use this code for your research, please cite our paper:

```
@article{poset_filters,
  title={Order theory in the context of machine learning: an application},
  author={},
  journal={},
  year={2024}
}
```
