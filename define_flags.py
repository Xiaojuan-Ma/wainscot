import tensorflow as tf
from transformer import model_params

# -------------- wainscot control -----------------
tf.app.flags.DEFINE_enum(
    'balancer', 'w_clu', ['w_tf', 'w_clu', 'w_inc'], 'Wainscot balancer type')

tf.app.flags.DEFINE_enum(
    'from_scratch_metric', 'group_op_num', ['group_op_num', 'group_comp_time'], 'metric for from scratch methods')

# for memory adjustment iteration
tf.app.flags.DEFINE_integer(
    'balance_limit', 7, 'The number of ADT iterations.')

tf.app.flags.DEFINE_float('var_threshold', 0.15, 'Balance condition: diff ratio w.r.t. average memory')

##### Reallocator  ######
tf.app.flags.DEFINE_boolean('baechi_algo_run', False, 'Run baechi algorithm to generate placement')

tf.app.flags.DEFINE_enum(
    'reallocator_type', 'metric_balance', ['metric_balance', 'baechi'], 'Wainscot balancer type')


# ----------------- for pesto exp ------------
# if doing pesto experiment, set this flag as True. otherwise False
tf.app.flags.DEFINE_boolean(
    'is_pesto', False, 'Pesto-Clu.')

tf.app.flags.DEFINE_boolean(
    'pesto_collect_data', True, 'collecting op_graph from baechi run.')

tf.app.flags.DEFINE_boolean(
    'pesto_collect_data_force', True, 'collecting op_graph from baechi run.')

tf.app.flags.DEFINE_string(
    'pesto_dir', 'pesto/', 'pesto/')


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ for main.py $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
tf.app.flags.DEFINE_boolean(
    'log_device_placement', False, 'Logging device placement.')

tf.app.flags.DEFINE_boolean(
    'colocate_grads_with_ops', True, 'Colocate gradient with ops.')

tf.app.flags.DEFINE_enum(
    'optimizer', 'adam',
    ['adadelta', 'adagrad', 'adam', 'ftrl', 'momentum', 'sgd', 'rmsprop'],
    'The name of the optimizer')

tf.app.flags.DEFINE_string(
    'model_name', 'gnmt_v2', 'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'batch_size', 128, 'The number of samples in each batch.')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_string(
    'logdir', '', 'Path to log dir.')

tf.app.flags.DEFINE_string(
    'cost_path', './tmp/cost.pkl', 'Path to the cost file.')

tf.app.flags.DEFINE_boolean(
    'est_commi_func',False, 'Generate cost dict.')

tf.app.flags.DEFINE_boolean(
    'costgen',True, 'Generate cost dict.')

# Wainscot only uses only_forward = True
tf.app.flags.DEFINE_boolean(
    'only_forward', True, 'Consider only forward ops.')

tf.app.flags.DEFINE_float('memory_fraction', 1.0, 'GPU memory fraction')

tf.app.flags.DEFINE_string(
    'comm_cost_coeffs', '0.0001754,134',
    'Comma-separated linear communication cost function coefficients')

tf.app.flags.DEFINE_float(
    'comm_cost_factor', 1.0, 'Communication cost function factor.')

tf.app.flags.DEFINE_float(
    'cost_factor', 1.0, 'Factor that applies to all costs')

###### Image classifier ######
tf.app.flags.DEFINE_enum(
    'data_format', 'NHWC', ['NHWC', 'NCHW'], 'Image data format')

##### NMT ######
tf.app.flags.DEFINE_integer('vocab_size', 30000, 'Vocabulary size.')
tf.app.flags.DEFINE_integer('max_seq_length', 40, 'Max. sequence length.')
tf.app.flags.DEFINE_integer('rnn_units', 512, 'RNN units.')
tf.app.flags.DEFINE_integer('num_layers', 4, 'RNN # layers.')
tf.app.flags.DEFINE_enum(
    'rnn_unit_type', 'lstm', ['lstm', 'gru'], 'RNN unit type.')
tf.app.flags.DEFINE_enum(
    'encoder_type', 'gnmt', ['bi', 'uni', 'gnmt'], 'Encoder type.')
tf.app.flags.DEFINE_boolean(
    'residual', True, 'Add residual connections to RNN.')
tf.app.flags.DEFINE_integer('num_gpus', 4, 'Number of gpus for NMT.')
tf.app.flags.DEFINE_boolean('disable_nmt_colocation', False,
                            'Disable the NMT ops colocation.')

###### Transformer #####
## cshetty2 added
tf.app.flags.DEFINE_enum(
    'transformer_type', 'big',['tiny', 'base', 'big'] ,'Type of Transformer')




##### Grappler ######
tf.app.flags.DEFINE_boolean('grappler', False, 'Use Grappler.')
tf.app.flags.DEFINE_integer(
    'grappler_time', 3600, 'Allotted time in seconds for Grappler.')




# $$$$$$$$$$$$$$$$$$$$$$$$ for communication cost $$$$$$$$$$$$$$$$$$$$$$$$$$$
# xiao: to unify the runing
tf.app.flags.DEFINE_integer(
    'from_gpu_id', 0, 'From GPU ID')
tf.app.flags.DEFINE_integer(
    'to_gpu_id', 1, 'To GPU ID')
tf.app.flags.DEFINE_integer(
    'exponent', 30, 'Max tensor size. 2^(exponent).')




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ for placer/placer_lib.py $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
tf.app.flags.DEFINE_enum("placement_method", "m_sct",
                         ["m_sct", "m_sct_reserve", "m_topo",
                          "m_topo_nonuniform", "m_etf", "m_etf_reserve"],
                         "Placement method for placer.")

tf.app.flags.DEFINE_enum("placer_type", "fusion",
                         ["default", "colocation", "fusion"],
                         "Placer type.")

tf.app.flags.DEFINE_boolean(
    "only_important_ops", False, "Use only important ops for the placement.")

tf.app.flags.DEFINE_float(
    "placer_memory_fraction", 1.0,
    "Device memory fraction that is used by the placer.")

##### Placement graph building #####
tf.app.flags.DEFINE_string(
    "stats_path", "./stats.log",
    "Profiling result path to use for placement.")
tf.app.flags.DEFINE_string(
    "device_info_path", "./device_info.json",
    "Path to the JSON file where device information is stored.")

##### Logging ######
tf.app.flags.DEFINE_boolean(
    "log_placer_device_placement", False,
    "Log the device placement in the placement graph.")
tf.app.flags.DEFINE_boolean(
    "log_colocation_graph", False,
    "Log a graph consisting of colocation groups")

##### Colocation group placer #####
tf.app.flags.DEFINE_boolean(
    "resolve_cycle", False,
    "Resolve a cycle if exists by removing a single edge.")
tf.app.flags.DEFINE_boolean(
    "consider_all_edges_in_grouping", False,
    "Using all edges in generating all groups")

##### Fused op placer #####
tf.app.flags.DEFINE_boolean(
    "fusion_check_disjoint_paths", False,
    "Enable disjoint path check to find more fusion chances")

tf.app.flags.DEFINE_boolean(
    "fusion_allow_cycle", False, "Allow cycles in operator fusion.")

##### SCT flags #####
tf.app.flags.DEFINE_float(
    "sct_threshold", 0.1,
    "Threshold to transform relaxed SCT solutions to integers.")



#$$$$$$$$$$$$$$$$$$$$$$$$ for placer/grouper.py $$$$$$$$$$$$$$$
tf.app.flags.DEFINE_enum('grouper', 'coplace', ['tf', 'coplace'],
                         'Grouping algorithm')


# $$$$$$$$$$$$$$$$$$$$$$$$ for placer/adjuster.py $$$$$$$$$$$$$$$$$
#
# tf.app.flags.DEFINE_enum(
#     "adjustment_method", "noop", list(_ADJUSTER_CLASSES.keys()),
#     "Method to adjust placement for colocation groups")
# tf.app.flags.DEFINE_boolean(
#     'adjustment_with_memory_limit', False,
#     'In adjusting the placement, the device memory is considered.')