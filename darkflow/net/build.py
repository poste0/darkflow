import tensorflow as tf
import time
from . import help
from . import flow
from .ops import op_create, identity
from .ops import HEADER, LINE
from .framework import create_framework
from ..dark.darknet import Darknet
import json
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TFNet(object):

	_TRAINER = dict({
		'rmsprop': tf.train.RMSPropOptimizer,
		'adadelta': tf.train.AdadeltaOptimizer,
		'adagrad': tf.train.AdagradOptimizer,
		'adagradDA': tf.train.AdagradDAOptimizer,
		'momentum': tf.train.MomentumOptimizer,
		'adam': tf.train.AdamOptimizer,
		'ftrl': tf.train.FtrlOptimizer,
		'sgd': tf.train.GradientDescentOptimizer
	})

	# imported methods
	_get_fps = help._get_fps
	say = help.say
	train = flow.train
	camera = help.camera
	predict = flow.predict
	return_predict = flow.return_predict
	to_darknet = help.to_darknet
	build_train_op = help.build_train_op
	load_from_ckpt = help.load_from_ckpt

	def __init__(self, FLAGS, darknet = None):
		self.ntrain = 0

		if isinstance(FLAGS, dict):
			from ..defaults import argHandler
			newFLAGS = argHandler()
			newFLAGS.setDefaults()
			newFLAGS.update(FLAGS)
			FLAGS = newFLAGS

		self.FLAGS = FLAGS
		if self.FLAGS.tensor:
			with open(self.FLAGS.metaLoad, 'r') as fp:
				self.meta = json.load(fp)
			self.framework = create_framework(self.meta, self.FLAGS)
			MODEL_FILE = '/home/sergey/darkflow/built_graph/tiny-yolo-voc.uff'
			INPUT_NAME = "input"
			INPUT_SHAPE = (3, 416,416)
			OUTPUT_NAME = "BiasAdd_8"
			TRT_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
			from tensorrt.parsers import uffparser
			#builder = trt.Builder(TRT_LOGGER)
			#network = builder.create_network()
			parser = uffparser.create_uff_parser()
			parser.register_input("input", (3, 416,416), 0)
			parser.register_output(OUTPUT_NAME)
			#parser.parse(MODEL_FILE, network)
			import uff
			m = uff.from_tensorflow_frozen_model(MODEL_FILE, ["BiasAdd_8"])
			engine = trt.utils.uff_to_trt_engine(TRT_LOGGER, m, parser,1,1<<20)
			print(engine)
			inputs = []
			outputs = []
			bindings = []
			stream = cuda.Stream()
			for binding in engine:
				size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
				dtype = trt.nptype(engine.get_binding_dtype(binding))
				# Allocate host and device buffers
				host_mem = cuda.pagelocked_empty(size, dtype)
				device_mem = cuda.mem_alloc(host_mem.nbytes)
				# Append the device buffer to device bindings.
				bindings.append(int(device_mem))
				# Append to the appropriate list.
				if engine.binding_is_input(binding):
					inputs.append(HostDeviceMem(host_mem, device_mem))
				else:
					outputs.append(HostDeviceMem(host_mem, device_mem))
			context = engine.create_execution_context()
			img = cv2.imread('/home/sergey/darkflow/sample_img/sample_computer.jpg')
			img = self.framework.resize_input(img)
			img = img.transpose(2, 0, 1)
			h, w, _ = img.shape
			img = img.ravel()
			np.copyto(inputs[0].host,img)
			[cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
			# Run inference.
			context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
			# Transfer predictions back from the GPU.
			[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
			# Synchronize the stream
			stream.synchronize()

			[result] = [out.host for out in outputs]
			print(result)

			file = open('/home/sergey/q.txt' , 'w')
			np.savetxt('/home/sergey/q.txt' , result.ravel())
			out = np.ndarray(shape=(13,13,125) ,dtype=np.float32)
			out = result.reshape((13,13,125))
			boxes = self.framework.findboxes(out)
			threshold = self.FLAGS.threshold
			boxesInfo = list()
			for box in boxes:
				tmpBox = self.framework.process_box(box, h, w, threshold)
				if tmpBox is None:
					continue
				boxesInfo.append({
					"label": tmpBox[4],
					"confidence": tmpBox[6],
					"topleft": {
						"x": tmpBox[0],
						"y": tmpBox[2]},
					"bottomright": {
						"x": tmpBox[1],
						"y": tmpBox[3]}
				})
			print(boxesInfo)
			return
		if self.FLAGS.pbLoad and self.FLAGS.metaLoad:
			self.say('\nLoading from .pb and .meta')
			self.graph = tf.Graph()
			device_name = FLAGS.gpuName \
				if FLAGS.gpu > 0.0 else None
			with tf.device(device_name):
				with self.graph.as_default() as g:
					self.build_from_pb()
			return

		if darknet is None:	
			darknet = Darknet(FLAGS)
			self.ntrain = len(darknet.layers)

		self.darknet = darknet
		args = [darknet.meta, FLAGS]
		self.num_layer = len(darknet.layers)
		self.framework = create_framework(*args)
		
		self.meta = darknet.meta

		self.say('\nBuilding net ...')
		start = time.time()
		self.graph = tf.Graph()
		device_name = FLAGS.gpuName \
			if FLAGS.gpu > 0.0 else None
		with tf.device(device_name):
			with self.graph.as_default() as g:
				self.build_forward()
				self.setup_meta_ops()
		self.say('Finished in {}s\n'.format(
			time.time() - start))
	
	def build_from_pb(self):
		graph_def = tf.GraphDef()
		with tf.gfile.FastGFile(self.FLAGS.pbLoad, "rb") as f:
			graph_def.ParseFromString(f.read())
		import tensorflow.contrib.tensorrt as t
		graph_deff = t.create_inference_graph(
			input_graph_def=graph_def,
			outputs=['BiasAdd_22'],
			max_batch_size=1,
			max_workspace_size_bytes=512,
			precision_mode='FP32')

		
		tf.import_graph_def(
			graph_deff,
			name=""
		)
		with open(self.FLAGS.metaLoad, 'r') as fp:
			self.meta = json.load(fp)
		self.framework = create_framework(self.meta, self.FLAGS)

		# Placeholders
		self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
		self.feed = dict() # other placeholders
		self.out = tf.get_default_graph().get_tensor_by_name('BiasAdd_22:0')
		
		self.setup_meta_ops()
	
	def build_forward(self):
		verbalise = self.FLAGS.verbalise

		# Placeholders
		inp_size = [None] + self.meta['inp_size']
		self.inp = tf.placeholder(tf.float32, inp_size, 'input')
		self.feed = dict() # other placeholders

		# Build the forward pass
		state = identity(self.inp)
		roof = self.num_layer - self.ntrain
		self.say(HEADER, LINE)
		for i, layer in enumerate(self.darknet.layers):
			scope = '{}-{}'.format(str(i),layer.type)
			args = [layer, state, i, roof, self.feed]
			state = op_create(*args)
			mess = state.verbalise()
			self.say(mess)
		self.say(LINE)

		self.top = state
		self.out = tf.identity(state.out, name='output')

	def setup_meta_ops(self):
		cfg = dict({
			'allow_soft_placement': False,
			'log_device_placement': False
		})

		utility = min(self.FLAGS.gpu, 1.)
		if utility > 0.0:
			self.say('GPU mode with {} usage'.format(utility))
			cfg['gpu_options'] = tf.GPUOptions(
				per_process_gpu_memory_fraction = utility)
			cfg['allow_soft_placement'] = True
		else: 
			self.say('Running entirely on CPU')
			cfg['device_count'] = {'GPU': 0}

		if self.FLAGS.train: self.build_train_op()
		
		if self.FLAGS.summary:
			self.summary_op = tf.summary.merge_all()
			self.writer = tf.summary.FileWriter(self.FLAGS.summary + 'train')
		
		self.sess = tf.Session(config = tf.ConfigProto(**cfg))
		self.sess.run(tf.global_variables_initializer())

		if not self.ntrain: return
		self.saver = tf.train.Saver(tf.global_variables(), 
			max_to_keep = self.FLAGS.keep)
		if self.FLAGS.load != 0: self.load_from_ckpt()
		
		if self.FLAGS.summary:
			self.writer.add_graph(self.sess.graph)

	def savepb(self):
		"""
		Create a standalone const graph def that 
		C++	can load and run.
		"""
		darknet_pb = self.to_darknet()
		flags_pb = self.FLAGS
		flags_pb.verbalise = False
		
		flags_pb.train = False
		# rebuild another tfnet. all const.
		tfnet_pb = TFNet(flags_pb, darknet_pb)		
		tfnet_pb.sess = tf.Session(graph = tfnet_pb.graph)
		# tfnet_pb.predict() # uncomment for unit testing
		name = 'built_graph/{}.pb'.format(self.meta['name'])
		os.makedirs(os.path.dirname(name), exist_ok=True)
		#Save dump of everything in meta
		with open('built_graph/{}.meta'.format(self.meta['name']), 'w') as fp:
			json.dump(self.meta, fp)
		self.say('Saving const graph def to {}'.format(name))
		graph_def = tfnet_pb.sess.graph_def
		tf.train.write_graph(graph_def,'./', name, False)
