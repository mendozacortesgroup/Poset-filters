# Poset-filters

This reposity contains the code of the paper "Order theory in the context of machine learning: an application"

## Contents


* **Project Titles**
  - Poset-filters are a family of convolutional filters.

* **Overview**
  - Poset-filters are a family of convolutional filters, sometimes they produce better accuracy than average pooling, max pooling or mix pooling.

    
 
* **Example Usage**: 
For the poset {w<x>y<z} :
```python
from poset_utils import filter_n 
#on the init section of the NN
self.poset_pool = filter_n

#on the forward section of the NN
out= self.poset_pool(out)
```
   
For another poset (for example the number 12):
```python
from poset_utils import CustomMultiPolyActivation,  dict_posets
#on the init section of the NN
self.poset_pool = CustomMultiPolyActivation(coeffs=dict_posets[12])

#on the forward section of the NN
out= self.poset_pool(out)
```

Test:
```python
import torch
from poset_utils import filter_n 

input_tensor = torch.ones(1, 3,8, 8)
input_tensor[0,0,0::2,0::2]=-1 #the simplex 4 should work here
input_tensor[0,1,0::2,:]=-1 #the simplex 0 or 4 should work here
input_tensor[0,2,1::2,:]=-1
input_tensor[0,2,:,0::2]=-1 #the simplex 1 should work here

#input_tensor[0,2,0::2,:]=-1 #the simplex should work here
print('We make an input that has 3 layers and each layer has different patterns.')
print(input_tensor)
print("output:")
print(filter_n.forward(input_tensor))
#should return a matrix with only the value 3, and two matrices of only 1's.
```


The index of the poset is the row in the following table, starting from 0 the slowest to 14 the fastest:

  ![standart](img/table.png)


* **Getting Started**
  - installation
    Clone this repo:
 
    git clone https://github.com/mendozacortesgroup/Poset-filters.git

    cd Poset_filters

  - prerequisites
    python >=3.7
    pytorch >= 2.2.0

  - location of:
    - code: [poset_filters](somelink)
    - issue tracker
    - notes:



* **Developer info**
  - Limitations and known issues
    If the input has odd dimentions, the code automatically adds padding one on the right and/or bottom.
    The combination ReLU followed by a poset filter seems to work well.

* **Colophon**
  - Credits -- code and algorithm: Eric Dolores Cuenca and Susana Lopez Moreno.
  - Copyright and License -- see [LICENSE](somefile) file.
  - How to contribute: .
  - This project has received funding from the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (2022R1A5A1033624, 2021R1A2B5B03087097).
  - References:  https://arxiv.org/abs/
  - How to cite this project:

