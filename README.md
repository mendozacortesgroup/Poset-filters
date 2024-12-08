# Poset-filters

This reposity contains the code of the paper "Order theory in the context of machine learning: an application

## Contents


* **Project Titles**
  - Poset-filters are a family of convolutional filters.

* **Overview**
  - Poset-filters are a family of convolutional filters, sometimes they produce better accuracy than average pooling, max pooling or mix pooling.

    
 
* **Example Usage**: 
For the poset {w<x>y<z} :
```#python
from poset_utils import filter_n 
#on the init section of the NN
self.poset_pool = filter_n

#on the forward section of the NN
out= self.poset_pool(out)
```
   
For another poset (for example the number 12):
```#python
from poset_utils import CustomMultiPolyActivation,  dict_posets
#on the init section of the NN
self.poset_pool = CustomMultiPolyActivation(coeffs=dict_posets[12])

#on the forward section of the NN
out= self.poset_pool(out)
```

The index of the poset is the row in the following table, note that posets with higher index are faster in this particular case:

  ![standart](img/table.png)


* **Getting Started**
  - installation
    Clone this repo:
 
    git clone https://github.com/mendozacortesgroup/Poset-filters.git
    cd Poset_filters
  - prerequisites
    python >=3.7
    pytorch

  - location of:
    - code: [poset_filters](somelink)
    - issue tracker
    - notes:



* **Developer info**
  - Limitations and known issues
    The combination ReLU followed by a poset filter seems to work well.

* **Colophon**
  - Credits -- code, algorithm, implementation/deployment, testing and direction: Eric Dolores Cuenca and Susana Lopez Moreno. Principal Investigator: Jose L. Mendoza-Cortes.  
  - Copyright and License -- see [LICENSE](somefile) file.
  - How to contribute: .
  - This project has received funding from the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. 2020R1C1C1A01008261).
  - This work was supported in part through computational resources and services provided by the Institute for Cyber-Enabled Research at Michigan State University
  - References:  https://arxiv.org/abs/
  - How to cite this project:

