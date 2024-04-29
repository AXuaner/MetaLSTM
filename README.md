

This code contains deep learning code used to streamflow forecast. 

This released code depends on our hydroDL repository, please follow our original github repository where we will release new updates occasionally
https://github.com/AXuaner/MetaLSTM/hydroDL


# Examples
The environment we are using is shown as the file `requirements.txt`.

# Installation
There are two different methods for hydroDL installation:

### Create a new environment, then activate it
  ```Shell
conda create -n mhpihydrodl python=3.6
conda activate mhpihydrodl
```

### 1) Using PyPI (stable package)
Install our hydroDL stable package from pip (Python version>=3.0)
```
pip install hydroDL
```

### 2) Source latest version
Install our latest hydroDL package from github

```
pip install git+https://github.com/AXuaner/MetaLSTM/hydroDL
```

_Note:_
If you want to run our examples directly, please download the [example folder](https://github.com/AXuaner/MetaLSTM/hydroDL)

There exists a small compatibility issue with our code when using the latest pyTorch version. Feel free to contact us if you find any issues or code bugs that you cannot resolve.


Please download both forcing, observation data `CAMELS time series meteorology, observed flow, meta data (.zip)` and basin attributes `CAMELS Attributes (.zip)`. 
Put two unzipped folders under the same directory, like `your/path/to/Camels/basin_timeseries_v1p2_metForcing_obsFlow`, and `your/path/to/Camels/camels_attributes_v2.0`. Set the directory path `your/path/to/Camels`
as the variable `rootDatabase` inside the code later.

Computational benchmark: training of CAMELS data with 671 basins, 10 years, 300 epochs with GPU.


# License
Non-Commercial Software License Agreement

By downloading the hydroDL software (the “Software”) you agree to
the following terms of use:
Copyright (c) 2020, The Pennsylvania State University (“PSU”). All rights reserved.

1. PSU hereby grants to you a perpetual, nonexclusive and worldwide right, privilege and
license to use, reproduce, modify, display, and create derivative works of Software for all
non-commercial purposes only. You may not use Software for commercial purposes without
prior written consent from PSU. Queries regarding commercial licensing should be directed
to The Office of Technology Management at 814.865.6277 or otminfo@psu.edu.
2. Neither the name of the copyright holder nor the names of its contributors may be used
to endorse or promote products derived from this software without specific prior written
permission.
3. This software is provided for non-commercial use only.
4. Redistribution and use in source and binary forms, with or without modification, are
permitted provided that redistributions must reproduce the above copyright notice, license,
list of conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS &quot;AS IS&quot;
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
