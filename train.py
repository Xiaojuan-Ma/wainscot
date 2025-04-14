# Copyright 2025 University of Illinois Board of Trustees. All Rights Reserved.
# Author: DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Wainscot, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Runs training."""
from __future__ import absolute_import, division, print_function

import bisect
import collections
import multiprocessing
import os
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.grappler import cluster as gcluster

from image_classifier.networks import nets_factory
from nmt import model_factory
from placer import placer_lib, cost as cost_lib
from placer.placer_utils import humanize_num_bytes
from placer.reallocator import Reallocator
# from placer.reallocator_experiments import Reallocator_Experiment
from third_party.grappler import graph_placer as grappler_graph_placer
from utils import logger

# for communication cost estimation
from utils import communication_benchmark

# all flags are defined in the define_flags.py
import define_flags

# for Pesto experiments, see readme.txt for details in /pesto/scripts/
from pesto.scripts.pesto_clustering import run_pesto_clustering
from pesto.scripts.satisfy_colocation_constraint import run_pesto_colocation_constraint

# for transformer models
from transformer import transformer
from transformer import metrics
from transformer import model_params
PARAMS_MAP = {
    "tiny": model_params.TINY_PARAMS,
    "base": model_params.BASE_PARAMS,
    "big": model_params.BIG_PARAMS,
}

_LOGGER = logger.get_logger(__file__)

_NUM_CLASSES = {
    'cifarnet': 10,
    'inception_v3': 1000,
    'nasnet_mobile': 1000,
    'nasnet_large': 1000,
    'nasnet_cifar': 1000,
    'pnasnet_mobile': 1000,
    'pnasnet_large': 1000,
}

ModelSpec = collections.namedtuple('ModelSpec', ['cls', 'image_size'])


def _configure_optimizer(optimizer_name, learning_rate):
    """Configures the optimizer used for training.

    Args:
        learning_rate: A scalar or `Tensor` learning rate.

    Returns:
        An instance of an optimizer.

    Raises:
        ValueError: if optimizer_name is not recognized.
    """
    if optimizer_name == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer_name == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif optimizer_name == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, name='Momentum')
    elif optimizer_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError(
            'Optimizer [%s] was not recognized' % optimizer_name)
    return optimizer


def _get_gpu_devices(sess_config):
    with tf.Session(config=sess_config) as sess:
        return [
            {"name": device.name,
             "memory_size": device.memory_limit_bytes,
             "type": device.device_type}
            for device in sess.list_devices()
            if device.device_type == 'GPU']
            # if device.device_type == 'GPU' or device.device_type == 'XLA_GPU']


def build_image_classifier_model(inputs, model_name, data_format):
    """Builds a image classifier with the given specs."""
    # pylint: disable=too-many-locals
    _LOGGER.info('data format: %s', data_format)

    images, labels = inputs

    num_classes = _NUM_CLASSES[model_name]
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes)

    logits, _ = network_fn(images, data_format=data_format)

    with tf.variable_scope('loss'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_sum(losses) / tf.to_float(images.shape[0])

    return loss


def build_nmt_model(inputs, model_name, **kwargs):
    """Builds NMT with the given specs."""
    # pylint: disable=too-many-locals
    # log NMT spec.
    _LOGGER.info(', '.join(['{}={}'.format(*item) for item in kwargs.items()]))

    src_input, target_input, target_output = inputs

    vocab_size = kwargs.pop('vocab_size')

    # remove the transformer_type parameter for a nmt model
    kwargs.pop('transformer_type')

    # replicate vocab size
    kwargs['src_vocab_size'] = vocab_size
    kwargs['tgt_vocab_size'] = vocab_size

    model_fn = model_factory.get_model_fn(model_name, **kwargs)
    _, loss = model_fn(src_input, target_input, target_output)

    return loss


def build_transformer_model(inputs, **kwargs):
    """Builds transformer with the given specs."""
    # pylint: disable=too-many-locals
    # log tranformer spec.
    _LOGGER.info(', '.join(['{}={}'.format(*item) for item in kwargs.items()]))

    with tf.variable_scope("model"):
        inputs, targets = inputs

        # Create model and get output logits.

        params = PARAMS_MAP[kwargs.pop('transformer_type')]
        model = transformer.Transformer(params, True)

        logits = model(inputs, targets)

        logits.set_shape(targets.shape.as_list() + logits.shape.as_list()[2:])

        # Calculate model loss.
        # xentropy contains the cross entropy loss of every nonpadding token in the
        # targets.
        # params["label_smoothing"] = 0.1
        # params["vocab_size"] = 33708
        xentropy, weights = metrics.padded_cross_entropy_loss(
            logits, targets, 0.1, 33708)
        loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
        return loss


def build_model(inputs, model_name, data_format, **kwargs):
    """Builds a model with the given specs."""
    if model_name in _NUM_CLASSES:
        return build_image_classifier_model(inputs, model_name, data_format)
    elif model_name == 'transformer':
        return build_transformer_model(inputs, **kwargs)
    return build_nmt_model(inputs, model_name, **kwargs)


def run_op(target_op, warmup_count=5, num_measurement=10,
           profile_every_n_steps=None, logdir=None, config=None, device_memories=[]):
    """Runs the given graph."""
    # pylint: disable=too-many-locals, too-many-arguments
    with tf.Session(config=config) as sess:
        if logdir:
            writer = tf.summary.FileWriter(logdir=logdir,
                                           graph=tf.get_default_graph())
        else:
            writer = None

        sess.run(tf.global_variables_initializer())

        warmup_start_time = time.time()

        for _ in range(warmup_count):
            sess.run(target_op)

        warmup_end_time = time.time()
        _LOGGER.info('Warmup time: %s',
                     str(warmup_end_time - warmup_start_time))

        runtimes = []
        run_metadata_list = []

        # for memory test
        memory_temp = []

        for step in range(1, num_measurement + 1):
            # print('the run step number:', step)
            if profile_every_n_steps and step % profile_every_n_steps == 0:
                _LOGGER.info('Profiling step %d...', step)
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                sess.run(target_op,
                         options=run_options,
                         run_metadata=run_metadata)

                if writer:
                    writer.add_run_metadata(
                        run_metadata, 'step-{}'.format(step))
                    # pylint: disable=invalid-name
                    metadata_out_path = os.path.join(
                        logdir, 'run_metadata-{}.pbtxt'.format(step))
                    with open(metadata_out_path, 'wb') as f:
                        f.write(run_metadata.SerializeToString())

                run_metadata_list.append(run_metadata)
            else:
                start_time = time.time()
                sess.run(target_op)
                end_time = time.time()
                runtimes.append(end_time - start_time)

                # # for memory test:
                #
                # for device in sess.list_devices():
                #     print("device:", device.name)
                #     if "CPU" in device.name:
                #         continue
                #     with tf.device(device.name):
                #         max_m_op = tf.contrib.memory_stats.MaxBytesInUse()
                #         max_m = sess.run(max_m_op)
                #         print(humanize_num_bytes(max_m), max_m)
                #         memory_temp.append(max_m)

        # # open a file, where you ant to store the data
        # file = open('re_data/memories_for_runs/memories_for_runs.pkl', 'wb')
        #
        # # dump information to that file
        # pickle.dump(memory_temp, file)
        #
        # # close the file
        # file.close()
        # # exit()

        _LOGGER.info('Profile run time: %s',
                     str(time.time() - warmup_end_time))

        avg_step_time = np.average(runtimes)

        _LOGGER.info('Graph execution stats. #samples=%d, median=%s, mean=%s',
                     len(runtimes),
                     np.median(runtimes),
                     np.average(runtimes))

        memory = []
        for device in sess.list_devices():
            print("device:", device.name)
            if "CPU" in device.name:
                continue
            with tf.device(device.name):
                max_m_op = tf.contrib.memory_stats.MaxBytesInUse()
                max_m = sess.run(max_m_op)
                print(humanize_num_bytes(max_m), max_m)
                memory.append(max_m)
        print('Average runtime: {}'.format(avg_step_time))
        device_memories.append(memory)
        return avg_step_time, run_metadata_list


def get_costs(target_op, warmup_count=5, num_measurement=50,
              profile_every_n_steps=5, sess_config=None, logdir=None):
    """Generates costs with tf.Session."""
    # pylint: disable=too-many-arguments
    avg_step_time, run_metadata_list = run_op(
        target_op,
        warmup_count=warmup_count,
        num_measurement=num_measurement,
        profile_every_n_steps=profile_every_n_steps,
        logdir=logdir,
        config=sess_config)
    cost_dict = cost_lib.build_cost_dict(run_metadata_list)
    return avg_step_time, cost_dict


def generate_cost(target_op, cost_path, sess_config=None, logdir=None):
    """Generates cost data for the graph at the given path."""
    if not cost_path:
        raise ValueError('cost_path is required.')
    print("what is the cost path", cost_path)

    # copy graphdef since get_costs will create init_op.
    graphdef = tf.get_default_graph().as_graph_def()

    start_time = time.time()
    step_time, cost_dict = get_costs(
        target_op, sess_config=sess_config, logdir=logdir)

    _LOGGER.info('Original runtime: %f', step_time)

    cost_dir_path = os.path.dirname(cost_path)
    if cost_dir_path:
        os.makedirs(cost_dir_path, exist_ok=True)
    # pylint: disable=invalid-name
    with open(cost_path, 'wb') as f:
        _LOGGER.info('Saving to %s...', cost_path)
        cost_data = {'graphdef': graphdef,
                     'cost_dict': cost_dict}
        pickle.dump(cost_data, f)

    _LOGGER.info('Profile run costs: %s', str(time.time() - start_time))


def run_placement(target_op, cost_path, comm_cost_coeffs, cost_factor,
                  logdir=None, sess_config=None):
    """Runs the placement."""
    # pylint: disable=too-many-locals
    if not cost_path:
        raise ValueError('cost_path is required.')

    print("cost path for placement:", cost_path)

    # pylint: disable=invalid-name
    with open(cost_path, 'rb') as f:
        cost_data = pickle.load(f)

    graph = tf.get_default_graph()

    assert cost_data['graphdef'] == graph.as_graph_def()

    devices = _get_gpu_devices(sess_config)

    cost_dict = cost_data['cost_dict']

    # adjust costs for sensitivity experiments.
    if cost_factor != 1.0:
        cost_dict, comm_cost_coeffs = cost_lib.adjust_costs(
            cost_factor, cost_dict, comm_cost_coeffs)

    start_time = time.time()
    placer = placer_lib.get_placer(
        graph,
        devices=devices,
        cost_dict=cost_dict,
        comm_cost_coeffs=comm_cost_coeffs)
    placer.run()
    _LOGGER.info('Entire placement time: %s', str(time.time() - start_time))
    # xiao: for reallocator outside the placer
    return placer


def _build_image_classifier_inputs(model_name, batch_size, data_format):
    num_classes = _NUM_CLASSES[model_name]
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes)

    if data_format == 'NHWC':
        input_shape = (batch_size,
                       network_fn.default_image_size,
                       network_fn.default_image_size,
                       3)
    else:
        input_shape = (batch_size,
                       3,
                       network_fn.default_image_size,
                       network_fn.default_image_size)

    images = np.ones(input_shape, dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.int32)

    element = (images, labels)

    with tf.variable_scope('dataset'):
        dataset = tf.data.Dataset.from_tensors(element).repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def _build_transformer_inputs(batch_size, max_seq_length):
    input_shape = (batch_size, max_seq_length)

    src_input = np.ones(input_shape, dtype=np.int32)
    #target_input = np.ones(input_shape, dtype=np.int32)
    target_output = np.ones(input_shape, dtype=np.int32)

    element = (src_input, target_output)

    with tf.variable_scope('dataset'):
        dataset = tf.data.Dataset.from_tensors(element).repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def _build_nmt_inputs(batch_size, max_seq_length):
    input_shape = (batch_size, max_seq_length)

    src_input = np.ones(input_shape, dtype=np.int32)
    target_input = np.ones(input_shape, dtype=np.int32)
    target_output = np.ones(input_shape, dtype=np.int32)

    element = (src_input, target_input, target_output)

    with tf.variable_scope('dataset'):
        dataset = tf.data.Dataset.from_tensors(element).repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def build_inputs(model_name, batch_size, data_format, max_seq_length):
    """Generates dummy inputs."""
    if model_name in _NUM_CLASSES:
        return _build_image_classifier_inputs(
            model_name, batch_size, data_format)
    elif model_name == 'transformer':
        return _build_transformer_inputs(batch_size, max_seq_length)
    return _build_nmt_inputs(batch_size, max_seq_length)


def build_train_op(loss, optimizer_name, learning_rate,
                   colocate_grads_with_ops):
    """Builds a train op."""
    optimizer = _configure_optimizer(optimizer_name, learning_rate)
    grads_and_vars = optimizer.compute_gradients(
        loss, colocate_gradients_with_ops=colocate_grads_with_ops)
    global_step = tf.train.create_global_step()
    return optimizer.apply_gradients(grads_and_vars,
                                     global_step=global_step)


def run_grappler(target_op, allotted_time, logdir, sess_config):
    """Runs Grappler placement."""
    tf.logging.set_verbosity(tf.logging.INFO)

    # need to create a session here with memory fraction.
    # otherwise, memory fraction flag is not correctly set due to a session
    # created by cluster
    with tf.Session(config=sess_config):
        pass

    graph = tf.get_default_graph()

    cluster = gcluster.Cluster()
    metagraph = tf.train.export_meta_graph(graph=graph,
                                           clear_extraneous_savers=True)

    _LOGGER.info('Grappler allotted time: %d', allotted_time)

    placed_metagraph_list = grappler_graph_placer.PlaceGraph(
        metagraph,
        cluster=cluster,
        allotted_time=allotted_time,
        verbose=True,
        sess_config=sess_config,
        gpu_only=True)

    _LOGGER.info('# found metagraph: %d', len(placed_metagraph_list))

    if len(placed_metagraph_list) == 0:
        _LOGGER.info('No feasible placement is found.')
        return

    if logdir:
        metagraph_dir = os.path.join(logdir, 'metagraph')
        os.makedirs(metagraph_dir, exist_ok=True)
        for i, metagraph in enumerate(placed_metagraph_list):
            metagraph_path = os.path.join(
                metagraph_dir, 'metagraph-%d.pbtxt' % i)
            # pylint: disable=invalid-name
            with open(metagraph_path, 'wb') as f:
                f.write(metagraph.SerializeToString())

    # use the last element because it is the best placement that is found.
    placed_metagraph = placed_metagraph_list[-1]

    # assign device placement
    for node in placed_metagraph.graph_def.node:
        tf_op = graph.get_operation_by_name(node.name)
        # pylint: disable=protected-access
        tf_op._set_device(node.device)

    step_time = run_op(
        target_op, warmup_count=10, num_measurement=21,
        profile_every_n_steps=21, logdir=logdir,
        config=sess_config)[0]

    _LOGGER.info('Average runtime: {}'.format(step_time))


def parse_comm_cost_coeffs(coeffs_str, factor=1.0):
    comm_cost_coeffs = coeffs_str.split(',')
    assert len(comm_cost_coeffs) == 2

    comm_cost_coeffs[0] = float(comm_cost_coeffs[0])
    comm_cost_coeffs[1] = int(comm_cost_coeffs[1])

    if factor != 1.0:
        _LOGGER.info('Communication cost factor: %s', str(factor))
        comm_cost_coeffs = tuple(
            [value * factor for value in comm_cost_coeffs])

    return comm_cost_coeffs


def setup_model():
    inputs = build_inputs(
        model_name=FLAGS.model_name,
        batch_size=FLAGS.batch_size,
        # image classifier
        data_format=FLAGS.data_format,
        # NMT
        max_seq_length=FLAGS.max_seq_length,
    )

    # build graph
    loss = build_model(
        inputs=inputs,
        model_name=FLAGS.model_name,
        # image classifier
        data_format=FLAGS.data_format,
        # NMT
        vocab_size=FLAGS.vocab_size,
        rnn_units=FLAGS.rnn_units,
        num_layers=FLAGS.num_layers,
        rnn_unit_type=FLAGS.rnn_unit_type,
        encoder_type=FLAGS.encoder_type,
        residual=FLAGS.residual,
        num_gpus=FLAGS.num_gpus,
        colocation=not FLAGS.disable_nmt_colocation,
        transformer_type=FLAGS.transformer_type)
    # add to the train op collections to support important ops identification
    tf.add_to_collection(tf.GraphKeys.TRAIN_OP, loss)

    target_op = loss
    # only_forward = FLAGS.only_forward
    if not only_forward:
        train_op = build_train_op(
            loss,
            optimizer_name=FLAGS.optimizer,
            learning_rate=FLAGS.learning_rate,
            colocate_grads_with_ops=colocate_grads_with_ops)
        target_op = train_op

    if not FLAGS.costgen and FLAGS.grappler:
        run_grappler(
            target_op,
            allotted_time=FLAGS.grappler_time,
            logdir=FLAGS.logdir,
            sess_config=sess_config)
        return

    return loss, target_op

def profile(target_op):
    generate_cost(target_op,
                  cost_path=FLAGS.cost_path,
                  sess_config=sess_config,
                  logdir=FLAGS.logdir)

def run_baechi_algo(loss, target_op):
    tf.add_to_collection(tf.GraphKeys.TRAIN_OP, loss)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement)
    if FLAGS.memory_fraction != 1.0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = \
            FLAGS.memory_fraction

    placer = run_placement(
        target_op,
        cost_path=FLAGS.cost_path,
        comm_cost_coeffs=comm_cost_coeffs,
        cost_factor=FLAGS.cost_factor,
        logdir=FLAGS.logdir,
        sess_config=sess_config)
    return placer


def train_with_placement(loss, target_op, op_allocations):
    if only_forward:
        # build train op
        train_op = build_train_op(
            loss,
            optimizer_name=FLAGS.optimizer,
            learning_rate=FLAGS.learning_rate,
            colocate_grads_with_ops=colocate_grads_with_ops)
        target_op = train_op
    step_time = run_op(
        target_op, warmup_count=10, num_measurement=51,
        profile_every_n_steps=51, logdir=FLAGS.logdir,
        config=sess_config, device_memories=device_memories)[0]

    _LOGGER.info('Average runtime: {}'.format(step_time))
    step_times.append(step_time)

def sub_main(**args):
    """Main function."""
    # pylint: disable=invalid-name
    # FLAGS = tf.app.flags.FLAGS
    # pylint: enable=invalid-name
    print("args in submain", args)
    global step_times
    global device_memories
    global ex_results
    global reallocator

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement)

    # specify the number of GPUs tensorflow could use
    sess_config.gpu_options.visible_device_list = ','.join(str(i) for i in range(FLAGS.num_gpus))

    if FLAGS.memory_fraction != 1.0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = \
            FLAGS.memory_fraction
    # disable TF optimizer
    sess_config.graph_options.optimizer_options.opt_level = -1
    _LOGGER.debug('Session config: %s', str(sess_config))


    inputs = build_inputs(
        model_name=FLAGS.model_name,
        batch_size=FLAGS.batch_size,
        # image classifier
        data_format=FLAGS.data_format,
        # NMT
        max_seq_length=FLAGS.max_seq_length,
    )

    # build graph
    loss = build_model(
        inputs=inputs,
        model_name=FLAGS.model_name,
        # image classifier
        data_format=FLAGS.data_format,
        # NMT
        vocab_size=FLAGS.vocab_size,
        rnn_units=FLAGS.rnn_units,
        num_layers=FLAGS.num_layers,
        rnn_unit_type=FLAGS.rnn_unit_type,
        encoder_type=FLAGS.encoder_type,
        residual=FLAGS.residual,
        num_gpus=FLAGS.num_gpus,
        colocation=not FLAGS.disable_nmt_colocation,
        transformer_type=FLAGS.transformer_type)

    # only_forward = FLAGS.only_forward
    _LOGGER.info('Only consider forward ops: %s', str(only_forward))
    colocate_grads_with_ops = FLAGS.colocate_grads_with_ops
    _LOGGER.info('Coloate grads with ops: %s' % str(colocate_grads_with_ops))

    comm_cost_coeffs = parse_comm_cost_coeffs(
        FLAGS.comm_cost_coeffs, FLAGS.comm_cost_factor)

    if only_forward:
        assert colocate_grads_with_ops

    # add to the train op collections to support important ops identification
    tf.add_to_collection(tf.GraphKeys.TRAIN_OP, loss)

    target_op = loss

    if FLAGS.costgen:
        if not only_forward:
            train_op = build_train_op(
                loss,
                optimizer_name=FLAGS.optimizer,
                learning_rate=FLAGS.learning_rate,
                colocate_grads_with_ops=colocate_grads_with_ops)
            target_op = train_op

        generate_cost(target_op,
                      cost_path=FLAGS.cost_path,
                      sess_config=sess_config,
                      logdir=FLAGS.logdir)
    else:
        if not only_forward:
            train_op = build_train_op(
                loss,
                optimizer_name=FLAGS.optimizer,
                learning_rate=FLAGS.learning_rate,
                colocate_grads_with_ops=colocate_grads_with_ops)
            target_op = train_op

        if FLAGS.grappler:
            run_grappler(
                target_op,
                allotted_time=FLAGS.grappler_time,
                logdir=FLAGS.logdir,
                sess_config=sess_config)
            return

        global reallocator
        if not reallocator:
            placer = run_placement(
                target_op,
                cost_path=FLAGS.cost_path,
                comm_cost_coeffs=comm_cost_coeffs,
                cost_factor=FLAGS.cost_factor,
                logdir=FLAGS.logdir,
                sess_config=sess_config)
            reallocator = Reallocator(placer.op_index, placer.op_graph, placer.device_graph)
            f_name = "reallocator.pkl"
            with open(f_name, "wb") as outfile:
                # "wb" argument opens the file in binary mode
                pickle.dump(reallocator, outfile)
            print("reallocator has been written to:", f_name)

        if only_forward:
            # build train op
            train_op = build_train_op(
                loss,
                optimizer_name=FLAGS.optimizer,
                learning_rate=FLAGS.learning_rate,
                colocate_grads_with_ops=colocate_grads_with_ops)
            target_op = train_op

        step_time = run_op(
            target_op, warmup_count=10, num_measurement=51,
            profile_every_n_steps=51, logdir=FLAGS.logdir,
            config=sess_config, device_memories=device_memories)[0]

        _LOGGER.info('Average runtime: {}'.format(step_time))
        step_times.append(step_time)


class PestoClu:
    def __init__(self, model_name):
        """
        all data need to be precalculated and put into the corresponding folder with specfic name
        Args:
            model_name:
        """
        filename =  PestoClu.get_pesto_filename(pesto_info_dir='pesto/info/', model_name=model_name)
        df = pd.read_csv(filename)
        self.op_num = df['op_nums'].tolist()
        self.sorted_groups = df['colocation_group'].tolist()  
        df['op_ids'] = df['op_ids'].apply(lambda x: set(map(int, x.strip('{}').split(','))))
        self.group_to_op_ids = df.set_index('colocation_group')['op_ids'].to_dict()


    @classmethod
    def get_pesto_filename(cls, pesto_info_dir, model_name):
        filename = pesto_info_dir + model_name + '_pesto_ordered_info.csv'
        return filename

    def get_pesto_ordered_op_num_list(self):
        return self.op_num

    def get_pesto_ordered_merged_colocation_list(self):
        return self.sorted_groups

    def get_pesto_merged_colocation_to_op_ids_dict(self):
        return self.group_to_op_ids



class Exps:

    def write_reallocator_to_file(self, reallocator, f_name="reallocator.pkl"):
        with open(f_name, "wb") as outfile:
            # "wb" argument opens the file in binary mode
            pickle.dump(reallocator, outfile)
        print("reallocator has been written to:", f_name)

    def _update_tf_op_assignment(self, tf_graph, reallocator, op_allocations):
        # this is method will update the default tf graph's op assignment according to the op_allocations dictionary
        # parameter: op_allocations: a dictionary with op allocation info. key: op_id (by baechi), value: device id
        # need to convert op_id and device_id into op_name and device_name that used by tensorflow
        # global reallocator
        op_names = {}
        for op_id, device_id in op_allocations.items():
            op_names[reallocator.index_op[op_id]] = reallocator.devices[device_id]

        for tf_op in tf_graph.get_operations():
            # tf_op may not be in the op_index if it is not an important op
            if tf_op.name in op_names:
                tf_op._set_device(op_names[tf_op.name])

    def child_process_run_func(self, func, **kwargs):
        p = multiprocessing.Process(target=func, kwargs=kwargs)
        p.start()
        p.join()

    def _run_profiler(self):
        target_op = setup_model()
        profile(target_op)

    def read_reallocator_from_file(self, f_name="reallocator.pkl"):
        # global reallocator
        with open(f_name, "rb") as infile:
            print("reading reallocator from file:", f_name)
            reallocator = pickle.load(infile)
        return reallocator

    def _run_baechi_algo(self, **kwargs):
        # kwargs: dictionaty. key: 'filename', val: the filename to write the reallocator
        print('running Baechi algorithm')
        tf.reset_default_graph()
        loss, target_op = setup_model()
        placer = run_baechi_algo(loss, target_op)

        print('--------------starts creating reallocator-----------')
        reallocator_start = time.time()
        reallocator = Reallocator(placer.op_index, placer.op_graph, placer.device_graph)
        reallocator_end = time.time()
        print('--------------ends creating reallocator-----------, total time in seconds', reallocator_end - reallocator_start)

        if not kwargs or not kwargs['filename']:
            f_name = "reallocator.pkl"
        else:
            f_name = kwargs['filename']
        self.write_reallocator_to_file(reallocator, f_name=f_name)
        # print('what is the reallocator file?', f_name)
        print('the reallocator from baechi run has been written to ', f_name)

    def _run_allocation(self, **kwargs):
        # the kwargs here is the op_allocation dictionary for the placement
        # the model is built based on the current FLAGS values. Any modification of FLAGs should be done before this call.
        # the placement is in reallocator's ops, reallocation need to be done before this
        op_allocations = kwargs['allocations']
        reallocator = kwargs['allocator']
        tf.reset_default_graph()
        # op_allocations = kwargs
        loss, target_op = setup_model()
        graph = tf.get_default_graph()

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement)
        if FLAGS.memory_fraction != 1.0:
            sess_config.gpu_options.per_process_gpu_memory_fraction = \
                FLAGS.memory_fraction
        # need to update the op allocations, even for basic baechi, since the target_op here is a newly built model
        # may reduce this cost in the future
        self._update_tf_op_assignment(graph, reallocator, op_allocations)
        train_with_placement(loss, target_op, op_allocations)

    def _get_op_allocations(self, reallocator, rebalance_metrics=[], allocation_type=None,
                            need_generate_data=True, level_type='two_levels'):
        # parameter: True: need_generate_data: generate necessary data for reallocation from reallocator
        # False: read file to initiate the variables. placer/reallocator.py -> metric_balance->_get_data -> _read_metric_data
        #  need to be rewritten for clarity
        """
        if no allocation_type provided, use FLAGS.reallocator_type
        """
        if not allocation_type:
            allocation_type = FLAGS.reallocator_type

        if allocation_type == 'metric_balance':
            _, op_allocations = reallocator.ex_type_dic[allocation_type](rebalance_metrics=rebalance_metrics,
                                                                     need_generate_data=need_generate_data,
                                                                     level_type=level_type)
        else:
            # for now, the other possible value is 'baechi', which directly read op_allocations from the reallocator.
            _, op_allocations = reallocator.ex_type_dic[allocation_type]()
        return op_allocations


    def baechi(self):
        print('running Baechi')
        FLAGS.costgen = True
        if FLAGS.costgen:
            self.child_process_run_func(self._run_profiler)

        # run baechi algorithm and write the reallocator to file.
        FLAGS.baechi_algo_run = True
        if FLAGS.baechi_algo_run:
            self.child_process_run_func(self._run_baechi_algo)


    @staticmethod
    def write_csv(filename, data, header, mode='w'):
        df = pd.DataFrame(data, columns=header)
        df.to_csv(filename, mode=mode, index=False)

  
    @staticmethod
    def is_balanced(memories, threshold, M=4):
        print('what is the threshold in is_balanced', threshold, M)
        if not memories:
            return False
        values = [e for e in memories[:M]]
        print('values', values)
        ave = np.average(values)
        dif = [m/ave for m in values]
        _balanced = True
        for d in dif:
            if abs(1 - d) > threshold:
                _balanced = False
        if _balanced:
            print('balanced in oom and balance check, memories', memories)
            print('ave and dif', [d - 1 for d in dif])
        return _balanced

    @staticmethod
    def is_placement_stable(values):
        if not values or len(values) <= 1:
            return False
        cur = values[-1]
        pre = values[-2]
        cur.sort()
        pre.sort()
        for i in range(len(cur)):
            if abs(cur[i] - pre[i]) > cur[i] * 0.1:
                return False
        return True

    @staticmethod
    def presum(data_list):
        result = [0] * len(data_list)
        result[0] = data_list[0]
        for i in range(1, len(data_list)):
            result[i] = result[i - 1] + data_list[i]
        return result

    @staticmethod
    def write_op_allocations_to_file(op_allocations, f_name):
        with open(f_name, "wb") as outfile:
            # "wb" argument opens the file in binary mode
            pickle.dump(op_allocations, outfile)
        print("op_allocations has been written to:", f_name)

    @staticmethod
    def get_filename(dir='./data/', raw_name='', placement_method='', grouper='', extra_info=''):

        # filename = dir + FLAGS.model_name + '_'
        # for debug purpose
        # dir += 'old_data/'
        # extra_info = 'extra_run'
        if 'pesto' in extra_info:
            dir += 'pesto/'
        filename = dir + extra_info + FLAGS.model_name + '_'
        # filename = dir +  FLAGS.model_name + '_'

        if FLAGS.model_name == 'transformer':
            filename += FLAGS.transformer_type + '_'
        filename += str(FLAGS.batch_size)

        if placement_method:
            filename += '_' + FLAGS.placement_method
        if grouper:
            filename += '_' + FLAGS.grouper

        # raw_name: 'filename.suffix'
        # filename += '_dfs_topo_'

        # for_debuggering
        # filename += '_debug'

        if raw_name:
            name, suffix = raw_name.split('.')
            filename += raw_name
        return filename

    def oom_balance_check(self, cnt, threshold=0.15, M=4):   # M is the number of available devices, by default is 4
        # check balance only when oom does not happen
        print('waht is the threshod in oom check')
        oom, balanced = False, False
        if len(device_memories) == cnt:
            # self.write_op_allocations_to_file(op_allocations, 'post_balance_op_allocation.pkl')
            balanced = Exps.is_balanced(memories=device_memories[-1][1:], threshold=threshold, M=M)
            memories = [humanize_num_bytes(e) for e in device_memories[-1][1:]]
            print('device memroies:', memories)
        else:
            # case of OOMs
            oom = True
            step_times.append(-1)
            device_memories.append([])
            print('OOM happens for round {}, after append placetaker, the length of step_times and memories are {} '
                  .format(cnt, len(step_times), len(device_memories)))
        return oom, balanced

    def append_data_clean_up(self, exp_type, steptime_memory_filename):
        print('data in append and clean up', device_memories)
        humanrized_memories = []
        total_memories = []
        rows = []
        max_peak_memories = []
        for row in device_memories:
            if not row:
                humanrized_memories.append([])
                total_memories.append(0)
                rows.append([])
                max_peak_memories.append(0)
            else:
                row = row[1:]
                max_val = np.max(row)
                rows.append(row)
                max_peak_memories.append(humanize_num_bytes(max_val))
                total_memories.append(humanize_num_bytes(sum(row)))
                humanrized_memories.append([humanize_num_bytes(e) for e in row])
        print(humanrized_memories[-1], humanize_num_bytes(sum(row)))
        # data = list(zip(step_times, device_memories[1:], humanrized_memories, total_memories))
        # header = [FLAGS.grouper + exp_type + '_steptime', 'raw_memories', 'humanwrized_memories', 'total_memory']
        data = list(zip(step_times, rows, humanrized_memories, max_peak_memories, total_memories))
        first_col_header = FLAGS.grouper + '_' + FLAGS.placement_method + '_' + exp_type + '_steptime'
        header = [first_col_header, 'raw_memories', 'humanwrized_memories', 'max_peak_memory', 'total_memory']

        # append data: filename, data, header, mode='w'
        self.write_csv(steptime_memory_filename, data, header, 'a')
        # clean data
        step_times[:] = []
        device_memories[:] = []


    def balance_from_scratch(self, reallocator,  steptime_memory_filename, metric='group_op_num', sys_type='wainscot', pesto_clu=None, targets=None, threshold=0.15):
        """
        :param reallocator: reallocator
        :param metric: the metric used for balancing purpose. default 'group_op_num'. possible metrics: 'group_memory', 'group_op_num', 'group_comp_time'
        :param steptime_memory_filename: as named
        :param exp_type: wainscot or pesto, flow is little bit different
        :param pesto_clu: if running pesto exp, a PestoClu object is needed
        :param targets: memory raios, if None, balancing
        :threshold: threshold for m-sct. Notice all exps use default 0.15, if want change, need check if the assignment of the new value is passed properly or not.
        
        :return : updated data in files
        """
        # todo: double check if the reallocator keeps the same

        def get_metrics_from_reallocator(reallocator):
            """
            return the metric_values based on the specified metric_type
            """

            cgroup_names_topo_order = reallocator._group_topo_order
            cgroup_names_2_op_info = {}
            # the index in the topo-order is the group's topo order
            sorted_group_info = {'gname': cgroup_names_topo_order}
            group_memory = []
            group_op_num = []
            group_comp_time = []
            for i in range(len(cgroup_names_topo_order)):
                gname = cgroup_names_topo_order[i]
                group = reallocator._co_groups[gname]
                # currently, group object does not have the op computation time. get it here
                # probably need to add it to the group object
                comp_time = 0
                for op_id, op_object in group.group_ops.items():
                    if 'end_ts' and 'start_ts' in op_object.op_info:
                        comp_time += op_object.op_info['end_ts'] - op_object.op_info['start_ts']
                    else:
                        print('no start/end ts for this op')
                # cgroup_names_2_op_info[gname] = {'gname': gname, 'group_topo_order': i, 'op_num': len(group.group_ops), 'comp_time': comp_time,
                #                                  'simple_sum_memory': group.group_memory}
                group_op_num.append(len(group.group_ops))
                group_memory.append(group.group_memory)
                group_comp_time.append(comp_time)
            sorted_group_info['group_memory'] = group_memory
            sorted_group_info['group_op_num'] = group_op_num
            sorted_group_info['group_comp_time'] = group_comp_time
            return sorted_group_info

        def get_cut_indices(presum):
            # this method is for general memory distribution control
            """
            return: cut_indices [0, first_breakpoint, secondpoint, ... len(ops)], has a length of len(ops) + 1
            """
            nonlocal targets
            print('targets', targets)
            if not targets:
                return get_cut_indices_balance(presum)
            else:
                total_ops = presum[-1]
                print('ops sum', total_ops)
                breakpoints = [e * total_ops for e in targets]
                print('targets need to find', breakpoints)
                cut_indices = []
                for i in range(len(breakpoints)):  # leftovers will be assigned to the last one
                    cut_indices.append(bisect.bisect_left(presum, breakpoints[i]))
                cut_indices.append(len(presum))
                return cut_indices

        def get_cut_indices_balance(presum):
            cut_indices = []
            ave = presum[-1] / M
            for i in range(M):
                cut_indices.append(bisect.bisect_left(presum, ave * i))
            cut_indices.append(len(presum))
            print('cut in get_cut', cut_indices)
            return cut_indices

        def get_op_allocations_from_cut_indices(reallocator, cut_indices, sorted_group_info):
            print('cut_indices', cut_indices, len(cut_indices))
            op_allocations = {}
            for i in range(len(cut_indices) - 1):
                # print('device id in get op allocation', i)
                # assert i < 4, 'device id cannot be 4 or larger'
                gnames = sorted_group_info['gname'][cut_indices[i]: cut_indices[i+1]]
                for gname in gnames:
                    for op_id in reallocator._co_groups[gname].group_ops:
                        op_allocations[op_id] = i
            return op_allocations

        def update_values(pre_peak_memories,  cut_indices):
            nonlocal targets, ops_presum, targets, values
            print('what is the indices', cut_indices)
            print('targets', targets)
            # exit(3)
            if not targets:
                sum_peak_memory = sum(pre_peak_memories)
                weights = [e/sum_peak_memory for e in pre_peak_memories]
            else:
                weights = []
                for i in range(len(pre_peak_memories)):
                    op_num = ops_presum[cut_indices[i+1]-1] - ops_presum[cut_indices[i]]
                    weights.append(pre_peak_memories[i]/op_num)
            print('what is the weights', weights)
            for i in range(len(cut_indices) - 1):
                for j in range(cut_indices[i], cut_indices[i + 1]):
                    values[j] *= weights[i]
            # return values

        def get_op_allocations_from_cut_indices_pesto(group_to_op_ids, cut_indices, sorted_groups):
            print('cut_indices', cut_indices, len(cut_indices))
            op_allocations = {}
            for i in range(len(cut_indices) - 1):
                # print('device id in get op allocation', i)
                # assert i < 4, 'device id cannot be 4 or larger'
                gnames = sorted_groups[cut_indices[i]: cut_indices[i + 1]]
                for gname in gnames:
                    for op_id in group_to_op_ids[gname]:
                        op_allocations[op_id] = i
            return op_allocations



        if sys_type == 'pesto':
            assert pesto_clu is not None, 'pesto exp requires a PestoClu object'
        
        # targets = [2, 5, 3, 8]
        # targets = None   # memory ratio, by default balancing

        if targets:
            targets_sum = sum(targets)
            targets = [e / targets_sum for e in targets]
            print('target ratio', targets)
            targets = self.presum(targets)
            print('accumulated ratios for finding breakpoints', targets)
            targets.insert(0,0) #to match the average case format
            targets.pop() # the last one will be inferred from the previous ones.
            print('adjusted targets', targets)

        M = FLAGS.num_gpus

        for k in range(1): 
            if sys_type == 'pesto':
                sorted_group_info = pesto_clu.sorted_groups
                values = pesto_clu.op_num
                ops_presum = Exps.presum(values)
            else:
                sorted_group_info = get_metrics_from_reallocator(reallocator)
                values = sorted_group_info[metric]
                ops_presum = Exps.presum(values)

            if M == 1:
                print('only one GPU, should all be allocated on the same device')
                cut_indices = [0, len(sorted_group_info)]
                if sys_type == 'pesto':
                    op_allocations = get_op_allocations_from_cut_indices_pesto(pesto_clu.group_to_op_ids, cut_indices,
                                                                           sorted_group_info)
                else:
                    op_allocations = get_op_allocations_from_cut_indices(reallocator, cut_indices, sorted_group_info)
                self.child_process_run_func(self._run_allocation, allocator=reallocator, allocations=op_allocations)
                break    # if only one GPU, no need for any further placement adjustment

            # M is now defined above as the FLAGS.gpu_nums
            # M = len(reallocator.devices)

            # the length of the cut_indices is M + 1
            # the first item is 0, last is len(pre_sum),ith segment is defined by [indices[i], indices[i+1])
            cut_indices = get_cut_indices(Exps.presum(values))

            # balance the placement
            cnt = 1
            balance_limit = FLAGS.balance_limit  # to limit the number of calls of the balance algo
            balanced = False
            assert not device_memories, 'device_memory should be empty before a new type of exp'
            # len(device_memories) should always be the same cnt, since will insert (-1, []) for OOMs
            while cnt < balance_limit and not balanced:
                print('from scratch balancing round', cnt)

                print('-------starts balancing scheduling, op_num ---------')
                balance_scheduling_start = time.time()

                if sys_type == 'pesto':
                    op_allocations = get_op_allocations_from_cut_indices_pesto(pesto_clu.group_to_op_ids, cut_indices,
                                                                           sorted_group_info)
                else:
                    op_allocations = get_op_allocations_from_cut_indices(reallocator, cut_indices, sorted_group_info)

                run_starts = time.time()


                self.child_process_run_func(self._run_allocation, allocator=reallocator, allocations=op_allocations)

                run_ends = time.time()

                # if OOMs, will add (-1 (step time), [](memories)), therefore, device_memories will not be empty
                oom, balanced = self.oom_balance_check(cnt, threshold=threshold, M=M)
                print('is balanced? in main', balanced)
                if balanced:
                    print('peak memory balanced')
                    break
                if oom:
                    print('oom happens for from scratch, no further way to adjust')
                    break
                # stable check
                if Exps.is_placement_stable(values=device_memories):
                    # for balancing from scratch, once oom happens, no way to further adjust
                    print('compare with the previous run, the placement is stable')
                    break
                cnt += 1

                update_values(device_memories[-1][1:],  cut_indices)
                ops_presum = Exps.presum(values)
                cut_indices = get_cut_indices(ops_presum)

                balance_scheduling_end = time.time()
                total_learning_exclude_run_placement = balance_scheduling_end - balance_scheduling_start
                total_learning_exclude_run_placement -= (run_ends - run_starts)
                print('-------end balancing scheduling, op_num---------, time in seconds',
                      total_learning_exclude_run_placement)
                print('details: ', balance_scheduling_start, run_starts, run_ends, balance_scheduling_end)

        # print('step times: baechi and {}'.format(FLAGS.reallocator_type))
        # print(step_times, step_times[-1] / step_times[0])
        # print('memory usage, baechi and {}'.format(FLAGS.reallocator_type))
        # # print(device_memories)
        # for row in device_memories:
        #     if not row: continue
        #     total = sum(row[1:])
        #     ms = [humanize_num_bytes(e) for e in row]
        #     print(ms[1:], humanize_num_bytes(total))
        exp_type = f'scratch_{metric}_{sys_type}'
        self.append_data_clean_up(exp_type, steptime_memory_filename)


    def baechi_profilifing_scheduling_run(self, steptime_memory_filename, baechi_op_allocation_filename=''):
        """
        beachi reallocator and op_allocations will be written to files in this method
        return: baechi's step time and device_memories
        the global step times and device memories will be cleaned up to empty.
        """
        if not baechi_op_allocation_filename:
            baechi_op_allocation_filename = Exps.get_filename('baechi_op_allocation.pkl')
        print('this is the call of baechi_profilifing_scheduling_run')
        # control parameters
        FLAGS.costgen = True  # if need run profiler to get costs for ops and tensors
        # FLAGS.baechi_algo_run = True  # if need to run baechi's placement algorithm
        # baechi_train = True  # if need run training using baechi's placement
        cost_filename = Exps.get_filename(raw_name='cost.pkl')
        FLAGS.cost_path = cost_filename
        self.child_process_run_func(self._run_profiler)

        # filename = 'reallocator_baechi.pkl'
        baechi_reallocator_filename = Exps.get_filename('reallocator_baechi.pkl')
        self.child_process_run_func(self._run_baechi_algo, filename=baechi_reallocator_filename)

        reallocator = self.read_reallocator_from_file(baechi_reallocator_filename)

        # if specifies 'baechi', directly read from the reallocator's info, no other algo runs
        op_allocations = self._get_op_allocations(reallocator, 'baechi')

        self.write_op_allocations_to_file(op_allocations, baechi_op_allocation_filename)

        # will weite step_times and device_memories in this call
        self.child_process_run_func(self._run_allocation, allocator=reallocator, allocations=op_allocations)

        baechi_step_time, baechi_memories = -1, []
        if device_memories:
            baechi_step_time, baechi_memories = step_times[-1], device_memories[-1]
            print('metrics from baechi run:', baechi_step_time, baechi_memories)
        else:
            print('OOM happens for baechi placement')
        self.append_data_clean_up(exp_type='baechi', steptime_memory_filename=steptime_memory_filename)
        return baechi_step_time, baechi_memories

    def balance_from_baechi(self,  reallocator, steptime_memory_filename, threshold=0.15, level_type='two_levels', baechi_memories=[], **kwargs):

        FLAGS.reallocator_type = 'metric_balance'
        need_generate_data = True
        for k in range(1):
            # balance the placement
            cnt = 1
            balance_limit = FLAGS.balance_limit  # to limit the number of calls of the balance algo
            balanced = False
            rebalance_metrics = baechi_memories

            while cnt < balance_limit and not balanced:
                baechi_balance_starts = time.time()
                if Exps.is_balanced(rebalance_metrics, threshold=threshold):
                    print('the rebalance metric is balanced', rebalance_metrics)
                    break
                op_allocations = self._get_op_allocations(reallocator=reallocator, rebalance_metrics=rebalance_metrics,
                                                          allocation_type=FLAGS.reallocator_type,
                                                          need_generate_data=need_generate_data, level_type=level_type)

                run_starts = time.time()
                self.child_process_run_func(self._run_allocation, allocator=reallocator, allocations=op_allocations)

                run_ends = time.time()

                oom, balanced = self.oom_balance_check(cnt, threshold=threshold)
                # print('is balanced? in main', balanced)
                if balanced:
                    print('peak memory balanced')
                    break
                if oom:
                    print('oom happens for from scratch, no further way to adjust')
                    break
                # stable check
                if Exps.is_placement_stable(values=device_memories):
                    # for balancing from scratch, once oom happens, no way to further adjust
                    print('compare with the previous run, the placement is stable')
                    break
                cnt += 1
                need_generate_data = False
                rebalance_metrics = device_memories[-1][1:]

                baechi_balance_ends = time.time()
                learning_exclude_run = baechi_balance_ends - baechi_balance_starts
                learning_exclude_run -= run_ends - run_starts
                print('-------baechi balance exclude run allocation, time in seconds', learning_exclude_run)
                print('details:', baechi_balance_starts, run_starts, run_ends, baechi_balance_ends)
        exp_type = 'baechi_' + level_type
        self.append_data_clean_up(exp_type, steptime_memory_filename)



    def comprehensive_exp(self, **kwargs):
        """
        for Baechi based exps

        """
        def get_baechi_run_information(baechi_op_allocation_filename, reallocator=None, reallocator_file_name=''):
            print('-----calling get_baechi_run_information-----')
            if not reallocator and reallocator_file_name:
                print('no reallocator or reallocator_file_name is given, start from cost generation')
                return self.baechi_profilifing_scheduling_run(
                    baechi_op_allocation_filename)
            else:
                if not reallocator:
                    reallocator = self.read_reallocator_from_file(reallocator_file_name)
                op_allocations = self._get_op_allocations(reallocator, allocation_type='baechi')
                self.write_op_allocations_to_file(op_allocations, baechi_op_allocation_filename)
                self.child_process_run_func(self._run_allocation, allocator=reallocator, allocations=op_allocations)
                print('this is the original baechi run')
                if device_memories:
                    baechi_step_time, baechi_memories = step_times[-1], device_memories[-1][1:]
                    print('metrics from baechi run:', baechi_step_time, baechi_memories)
                else:
                    print('OOM happens for baechi placement')
                    baechi_step_time, baechi_memories = -1, []

                self.append_data_clean_up(exp_type='baechi', steptime_memory_filename=steptime_memory_filename)
                return baechi_step_time, baechi_memories

        print('@@@@@ comprehensive exp run @@@@@@@@@')


        exp_start = time.time()

        # todo: haven't taken care of same paramemter situation, need to remind or rewrite or increase the file name
        # this name will be used for record all data based on the same cost profiling. same model, same batch size
        steptime_memory_filename = Exps.get_filename(raw_name='steptime_memories.csv')
        header = ['steptimes', 'raw_memories', 'humanrized_memories', 'max_peak_memory', 'humanrized_total_memory']
        data = []
        Exps.write_csv(steptime_memory_filename, data, header, 'a')
        print('-----steptime and memories data will be written to-----', steptime_memory_filename)

        # threshold = FLAGS.var_threshold
        run_profiler = FLAGS.costgen

        if FLAGS.balancer == 'w_tf':
            FLAGS.grouper = 'tf'
            print('w_tf requires tf as grouper, ignore original FLAGS.grouper setting')
        elif FLAGS.balancer == 'w_clu':
            FLAGS.grouper = 'coplace'
            print('w_clu requires coplace as grouper, ignore original FLAGS.grouper setting')


        path = Exps.get_filename(raw_name='cost.pkl')
        if not os.path.isfile(path):
            print('no cost file, ignore the run_profiler variable and do profiling')
            run_profiler = True

        # for general use
        if run_profiler:
            # Run profiling, the cost profiling will be written to firl cost_filename
            print('-----running baechi profiler-----')
            cost_filename = Exps.get_filename(raw_name='ngpus_cost.pkl')
            FLAGS.cost_path = cost_filename
            self.child_process_run_func(self._run_profiler)
            FLAGS.costgen = False  # read cost from the above file
            print('-----cost file has been written to-----', cost_filename)
        else:
            path = Exps.get_filename(raw_name='cost.pkl')
            assert os.path.isfile(path), 'not running profilier but no cost file'
            FLAGS.cost_path = path


        print('----grouper and placement method are {}, {}'.format(FLAGS.grouper, FLAGS.placement_method))
        # run baechi scheduling algorithm, the filename contains grouper info
        baechi_reallocator_filename = Exps.get_filename(raw_name='reallocator_baechi.pkl',
                                                    placement_method=FLAGS.placement_method, grouper=FLAGS.grouper)
        # the reallocator will be written to the baechi_reallocator_filename
        print('----runing baechi algorithm-----')
        self.child_process_run_func(self._run_baechi_algo, filename=baechi_reallocator_filename)

        # read the reallocator from the file
        print('----reading reallocator from file', baechi_reallocator_filename)
        reallocator = self.read_reallocator_from_file(baechi_reallocator_filename)

        step_times[:] = []
        device_memories[:] = []
        if FLAGS.balancer in ['w_tf', 'w_clu']:
            metric = FLAGS.from_scratch_metric
            print('---from scratch methods-----', metric)
            # step_times[:] = []
            # device_memories[:] = []
            self.balance_from_scratch(reallocator,  steptime_memory_filename, metric=FLAGS.from_scratch_metric, sys_type='wainscot', pesto_clu=None, targets=None, threshold=0.15)
        else:  # 'w_inc'
            baechi_op_allocation_filename = Exps.get_filename(raw_name='baechi_op_allocation.pkl')
            baechi_step_time, baechi_memories = \
                get_baechi_run_information(baechi_op_allocation_filename, reallocator=reallocator)
            level_type = 'two_levels' #  'cc_level' 'two_levels', 'group_level'
            print('-----from baechi methods-----', level_type)
            self.balance_from_baechi(reallocator, steptime_memory_filename, level_type=level_type,
                                        baechi_memories=baechi_memories)
        exp_end = time.time()
        print('the end to end for whole experiment in seconds:', exp_end-exp_start)
        print(f'file has been written to {steptime_memory_filename}')


    def pesto_exp(self, **kwargs):
        """
        for now the model name and batch size is given from the command line
        will generate one cost file, and do comprehensive exps based on this cost file,
        4 reallocator files (tf + [sct, etf] and coplace + [sct, etf])

        """

        def get_baechi_run_information(baechi_op_allocation_filename, reallocator=None, reallocator_file_name=''):
            print('-----calling get_baechi_run_information-----')
            if not reallocator and reallocator_file_name:
                print('no reallocator or reallocator_file_name is given, start from cost generation')
                return self.baechi_profilifing_scheduling_run(
                    baechi_op_allocation_filename)
            else:
                if not reallocator:
                    reallocator = self.read_reallocator_from_file(reallocator_file_name)
                op_allocations = self._get_op_allocations(reallocator, allocation_type='baechi')
                self.write_op_allocations_to_file(op_allocations, baechi_op_allocation_filename)
                self.child_process_run_func(self._run_allocation, allocator=reallocator, allocations=op_allocations)
                print('this is the original baechi run')
                if device_memories:
                    baechi_step_time, baechi_memories = step_times[-1], device_memories[-1][1:]
                    print('metrics from baechi run:', baechi_step_time, baechi_memories)
                else:
                    print('OOM happens for baechi placement')
                    baechi_step_time, baechi_memories = -1, []

                self.append_data_clean_up(exp_type='baechi', steptime_memory_filename=steptime_memory_filename)
                return baechi_step_time, baechi_memories


        pesto_clu = PestoClu(model_name=FLAGS.model_name)

        exp_start = time.time()

        # haven't taken care of same paramemter situation, need to remind or rewrite or increase the file name
        # this name will be used for record all data based on the same cost profiling. same model, same batch size
        steptime_memory_filename = Exps.get_filename(raw_name='steptime_memories.csv', extra_info=f'pesto_ngpu{FLAGS.num_gpus}')
        header = ['steptimes', 'raw_memories', 'humanrized_memories', 'max_peak_memory', 'humanrized_total_memory']
        data = []
        Exps.write_csv(steptime_memory_filename, data, header, 'a')
        print('-----steptime and memories data will be written to-----', steptime_memory_filename)

        run_profiler = False
        run_scratch_methods = True
        run_baechi = False

        path = Exps.get_filename(raw_name='cost.pkl')
        if not os.path.isfile(path):
            print('no cost file, ignore the run_profiler variable and do profiling')
            run_profiler = True
        else:
            run_profiler = False
            print('cost file exits, skip profiling')

        # for general use
        if run_profiler:
            # Run profiling, the cost profiling will be written to firl cost_filename
            print('-----running baechi profiler-----')
            cost_filename = Exps.get_filename(raw_name='cost.pkl')
            FLAGS.cost_path = cost_filename
            self.child_process_run_func(self._run_profiler)
            FLAGS.costgen = False  # read cost from the above file
            print('-----cost file has been written to-----', cost_filename)
        else:
            path = Exps.get_filename(raw_name='cost.pkl')
            assert os.path.isfile(path), 'not running profilier but no cost file'
            FLAGS.cost_path = path

        groupers = ['tf'] # for Pestco, we use tf
        placement_methods = [FLAGS.placement_method] #, 'm_etf'
        from_scratch_metrics = ['group_op_num']  #, 'group_op_num', 'group_comp_time'


        for grouper in groupers:
            FLAGS.grouper = grouper
            print('#############groupers:############', groupers)
            print('-------starting grouper-------', grouper)
            need_to_run_scrach = True # for from scratch, only need to run it once for each grouper
            for placement_method in placement_methods:
                FLAGS.placement_method = placement_method
                print('----grouper and placement method are {}, {}'.format(FLAGS.grouper, FLAGS.placement_method))
                # run baechi scheduling algorithm, the filename contains grouper info
                baechi_reallocator_filename = Exps.get_filename(raw_name='reallocator_baechi.pkl',
                                                           placement_method=placement_method, grouper=grouper)
                # the reallocator will be written to the baechi_reallocator_filename
                print('----runing baechi algorithm-----')
                self.child_process_run_func(self._run_baechi_algo, filename=baechi_reallocator_filename)

                # read the reallocator from the file
                print('----reading reallocator from file', baechi_reallocator_filename)
                reallocator = self.read_reallocator_from_file(baechi_reallocator_filename)

                if run_scratch_methods and need_to_run_scrach:
                    for metric in from_scratch_metrics:
                        print('---from scratch methods-----', metric)
                        step_times[:] = []
                        device_memories[:] = []
                        self.balance_from_scratch(reallocator,  steptime_memory_filename, metric='group_op_num', sys_type='pesto', pesto_clu=pesto_clu, targets=None, threshold=0.15)
                    need_to_run_scrach = False


                if grouper == 'tf': continue

                print('what is the gpu number', FLAGS.num_gpus)
                if run_baechi:
                    step_times[:] = []
                    device_memories[:] = []

                    baechi_op_allocation_filename = Exps.get_filename(raw_name='baechi_op_allocation.pkl')
                    baechi_step_time, baechi_memories = \
                        get_baechi_run_information(baechi_op_allocation_filename, reallocator=reallocator)

        exp_end = time.time()
        print('the end to end for whole experiment in seconds:', exp_end-exp_start)
        print(f'file has been saved to {steptime_memory_filename}')


    def pesto_collect_data(self, **kwargs):
        """
        This method collects  Baechi op_graph for Pesto clustering algorithm (see pesto/scripts/readme.txt)

        """
        print('@@@@@ collecting Baechi op_graph for Pesto @@@@@@@@@')

        exp_start = time.time()

        run_profiler = False
        # for collecting data, use tf colocation group
        groupers = ['tf'] 
        placement_methods = ['m_sct']  # placement_methods does not matter since we are not using it for Pesto, serves as a placeholder for Baechi to work
        
        # for Pesto data collection, we also use FLAGS.only_forward = True 
        assert FLAGS.only_forward and FLAGS.colocate_grads_with_ops, 'for now, use forward if need raw, comment this line'

        path = Exps.get_filename(raw_name='cost.pkl')
        if not os.path.isfile(path):
            print('no cost file, ignore the run_profiler variable and do profiling')
            run_profiler = True


        # for general use
        if run_profiler:
            # Run profiling, the cost profiling will be written to firl cost_filename
            print('-----running baechi profiler-----')
            cost_filename = Exps.get_filename(raw_name='cost.pkl')
            FLAGS.cost_path = cost_filename
            self.child_process_run_func(self._run_profiler)
            FLAGS.costgen = False  # read cost from the above file
            print('-----cost file has been written to-----', cost_filename)
        else:
            path = Exps.get_filename(raw_name='cost.pkl')
            assert os.path.isfile(path), 'not running profilier but no cost file'
            FLAGS.cost_path = path

        for grouper in groupers:
            FLAGS.grouper = grouper
            print('#############groupers:############', groupers)
            print('-------starting grouper-------', grouper)
            for placement_method in placement_methods:
                FLAGS.placement_method = placement_method
                print('----grouper and placement method are {}, {}'.format(FLAGS.grouper, FLAGS.placement_method))
                # run baechi scheduling algorithm, the filename contains grouper info
                baechi_reallocator_filename = Exps.get_filename(raw_name='reallocator_baechi.pkl',
                                                           placement_method=placement_method, grouper=grouper)
                # the reallocator will be written to the baechi_reallocator_filename
                print('----runing baechi algorithm for data collecting-----')
                self.child_process_run_func(self._run_baechi_algo, filename=baechi_reallocator_filename)


def main(unparsed_args):
    """Main function."""
    # if all flags values are corrected parsed, the only unparsed should be the file name.
    # otherwise, some flags are passed from the command line but not have been defined in define_flags.py file
    if len(unparsed_args) > 1:
        raise RuntimeError('Unparsed args: {}'.format(unparsed_args[1:]))

    global sess_config
    global only_forward
    global colocate_grads_with_ops
    global comm_cost_coeffs

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement)

    # specify the number of GPUs tensorflow can use
    sess_config.gpu_options.visible_device_list = ','.join(str(i) for i in range(FLAGS.num_gpus))

    # specify the portion of GPU memory tensorflow can use
    if FLAGS.memory_fraction != 1.0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = \
            FLAGS.memory_fraction

    colocate_grads_with_ops = FLAGS.colocate_grads_with_ops

    # only_forward = FLAGS.only_forward
    _LOGGER.info('Only consider forward ops: %s', str(only_forward))

    _LOGGER.info('Coloate grads with ops: %s' % str(colocate_grads_with_ops))

    if only_forward:
        assert colocate_grads_with_ops

    # if need to estimate the communication function, call communication benchmark
    if FLAGS.est_commi_func:
        FLAGS.comm_cost_coeffs = communication_benchmark.main()
   
    comm_cost_coeffs = parse_comm_cost_coeffs(
        FLAGS.comm_cost_coeffs, FLAGS.comm_cost_factor)
    

    exp = Exps()

    def pesto_prepare_data(surfix=''):
        if FLAGS.pesto_collect_data_force:
            print('Pesto Collecting Data Run......')
            exp.pesto_collect_data() # op_graph will be written to file in this method call. 
            # surfix = 'test'   # for testing
            run_pesto_clustering(FLAGS.model_name, FLAGS.pesto_dir, surfix=surfix)
            run_pesto_colocation_constraint(FLAGS.model_name, FLAGS.pesto_dir, surfix=surfix)
        else:
            # reuse some file, extra care may needed to file match
            FLAGS.pesto_collect_data   # this flag here is suppressed by existing file, change code if want override
            data_file = FLAGS.pesto_dir + 'info/' + f'{FLAGS.model_name}_pesto_ordered_info{surfix}.csv'
            merged_G_pkl = FLAGS.pesto_dir + f'{FLAGS.model_name}_merged_G_forward{surfix}.pkl'
            original_G_pkl = FLAGS.pesto_dir + f'{FLAGS.model_name}_colocation_graph_for_pesto_forwardTrue_colocatebackTrue.pkl'
            
            if os.path.exists(data_file):
                print(f"{data_file} exists — ignore FLAGS.pesto_collect_data, uses existing file.")
            elif os.path.exists(merged_G_pkl):
                print(f"{merged_G_pkl} exists — ignore FLAGS.pesto_collect_data, uses existing file.")
                run_pesto_colocation_constraint(FLAGS.model_name, FLAGS.pesto_dir, surfix=surfix)
            elif os.path.exists(original_G_pkl):
                print(f"orignal graph file for pesto exists — ignore FLAGS.pesto_collect_data, uses existing file.")
                run_pesto_clustering(FLAGS.model_name, FLAGS.pesto_dir, surfix=surfix)
                run_pesto_colocation_constraint(FLAGS.model_name, FLAGS.pesto_dir, surfix=surfix)
            else:
                print('Pesto Collecting Data Run......')
                exp.pesto_collect_data() # op_graph will be written to file in this method call. 
                # surfix = 'test'
                run_pesto_clustering(FLAGS.model_name, FLAGS.pesto_dir, surfix=surfix)
                run_pesto_colocation_constraint(FLAGS.model_name, FLAGS.pesto_dir, surfix=surfix)


    if FLAGS.is_pesto:
        # surfix = 'test'  # for testing purpose
        print('running Pesto exp')
        pesto_prepare_data()
        exp.pesto_exp()
    else:
        print('running Baechi based Wainscot')
        exp.comprehensive_exp()



# get flags
FLAGS = tf.app.flags.FLAGS
# for cross processes data collection
step_times = multiprocessing.Manager().list([])
device_memories = multiprocessing.Manager().list([])
ex_results = multiprocessing.Manager().list([-1])
group_assign_dic = multiprocessing.Manager().dict()
# Wainscot by default uses only_forward = True, suppresses FLAGS.only_forward
only_forward = True
comm_cost_coeffs = None
sess_config = None


if __name__ == "__main__":
    tf.app.run(main)

