# Wainscot: Tailoring Model Parallelism to Fit Device Memory Limits (ICDCS 2025)

## Install dependencies
Wainscot is built on Baechi (Baechi: Fast Device Placement of Machine Learning Graphs, SOCC 2020. https://github.com/beomyeol/baechi). It has the same dependency requirements as Baechi. 

* Install dependencies 
```
$ conda install -y python=3.6 numpy=1.16 tensorflow-gpu=1.12 bazel=0.20.0 \
      networkx future matplotlib cvxopt scikit-learn
```
* Mosek
```
$ pip install -f https://download.mosek.com/stable/wheel/index.html Mosek==8.1.82
```
One of Baechi's placement algorithm m-sct requires [MOSEK](https://www.mosek.com/) as an LP solver. Wainscot also needs it for basic Baechi run or Wainscot-Inc which balances Baechi's placement.
MOSEK provides a free personal academic license which can be requested at https://www.mosek.com/products/academic-licenses.
Put the license file (`mosek.lic`) at `$HOME/mosek`.

* Build the project
```
$ bazel build :train
```


## Example usage
Wainscot related paramters and most Baechi parameters locate in define_flags.py. 

Several example flags are:

1. Balancer.
```
tf.app.flags.DEFINE_enum(
    'balancer', 'w_inc', ['w_tf', 'w_clu', 'w_inc'], 'Wainscot balancer type')
```

2. Pestco-Clu. Pestco-Clu has a different work flow as Wainscot. Therefore, set is_pesto flag as True if you want to run Pesto-Clu.
```
tf.app.flags.DEFINE_boolean(
    'is_pesto', False, 'Pestco-Clu.')
```

3. Model name.  
```
tf.app.flags.DEFINE_string(
    'model_name', 'gnmt_v2', 'The name of the architecture to train.')
```
4. Batch size.
```
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'The number of samples in each batch.')
```
Plead check define_flags.py for a more complete parameter list. After setting flags with desired values, you can run the code as

```
$ ./bazel-bin/train
```

Alternatively, directly as
```
python train.py
```

Changing flag values with command line assignments is also supported. Example usage:
```
$ ./bazel-bin/train \
    --balancer=w_tf
```
or 
```
python train.py --balancer=w_tf
```

Intermediate files are in './data' by default (Pesto related experiments in '/pesto' and './data/pesto' folders). After running Wainscot, device peak memories and steptime will be saved to a file whose location will be indicated by the last output line, i.e., "file has been written to ./data/gnmt_v2_16steptime_memories.csv". 

The default setting in define_flags.py uses Wainscot-Clu as the balancer, and a 4-layer GNMT v2 (batch size 128, maximum sequence length 40, and vocabulary size 30000) as its model.

## License
University of Illinois/NCSA Open Source License
