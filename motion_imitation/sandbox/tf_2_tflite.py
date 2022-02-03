from pickletools import optimize
import tensorflow as tf
import numpy as np

# #https://github.com/tensorflow/tensorflow/issues/23809
# sess=tf.compat.v1.Session()
# metagraph=tf.compat.v1.saved_model.loader.load(sess,['serve'],'model2_tf') 
# sig=metagraph.signature_def['serving_default']
# input_dict=dict(sig.inputs)
# output_dict=dict(sig.outputs)
# print(input_dict,'\n',output_dict)
# input_obs_label_0=input_dict['obs'].name
# output_stochastic_label_0=output_dict['action'].name

# # {'obs': name: "input/Ob:0"
# # dtype: DT_FLOAT
# # tensor_shape {
# #   dim {
# #     size: -1
# #   }
# #   dim {
# #     size: 120
# #   }
# # }
# # }
# #  {'action': name: "Placeholder_14:0"
# # dtype: DT_FLOAT
# # tensor_shape {
# #   dim {
# #     size: -1
# #   }
# #   dim {
# #     size: 8
# #   }
# # }
# # }

# actual_input={input_obs_label_0:np.random.randn(1,120)} # fill here with your actual input 
# out=sess.run(output_stochastic_label_0,feed_dict=actual_input)
# print(out)

# def print_layers(graph_def):
#     def _imports_graph_def():
#         tf.compat.v1.import_graph_def(graph_def, name="")

#     wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
#     import_graph = wrapped_import.graph

#     print("-" * 50)
#     print("Frozen model layers: ")
#     layers = [op.name for op in import_graph.get_operations()]
#     ops = import_graph.get_operations()
#     print(ops[0])
#     print("Input layer: ", layers[0])
#     print("Output layer: ", layers[-1])
#     print("-" * 50)

# Load frozen graph using TensorFlow 1.x functions
# with tf.io.gfile.GFile("model2_tf/saved_model.pb", "rb") as f:
#     graph_def = tf.compat.v1.GraphDef()
#     loaded = graph_def.ParseFromString(f.read())

# frozen_func = print_layers(graph_def=graph_def)


#Assert:
# o = np.array([-0.03255312490386112, 0.04498604936452228, 0.44527336955070496, -1.2682040929794312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3183445525169373, 0.7678501498699188, 0.30798907995224, 1.366463918685913, 0.47884428545832636, 1.1910549592971802, 0.5739896301925183, 1.4169372510910034, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3459574233498874, 0.48675533772791924, -0.09871433195112012, 0.5871817392349538, 0.33003323470532364, 0.5622692392720565, 0.3010381091055335, 0.5716181287889274, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.08610070171098971, -0.00720271391288404, -0.007752998024744829, -0.038624271936682546, 0.01699184332054309, -0.7060793552451812, 0.7068743784962649, 0.224757039030773, 0.52, -0.18436251337275722, 0.52, 0.44050110176889173, 0.52, 0.23866509176225975, 0.52, 0.17100797944397417, -0.018361059366639874, -0.013303055610888848, -0.03920996170277488, 0.02891045391932716, -0.7089045132512306, 0.7036200364184361, 0.2837230201142283, 0.52, -0.11805427033996507, 0.52, 0.5970026325172565, 0.52, 0.0945582988343758, 0.52, 0.7629840709278087, -0.1323366299492109, 0.003265732728951365, -0.00670121625834154, 0.03990700778704027, -0.7427338870637621, 0.6683628486363519, 0.6700331036779094, 0.52, 0.27905820072466403, 0.52, 0.24307775585643515, 0.52, -0.1718126689644586, 0.52, 2.2532738862245267, -0.3597721366641932, -0.0019318320714872828, -0.01223931404628258, 0.040548257542990226, -0.7601965987087094, 0.6483110127968006, 0.6160323199419572, 0.52, 0.17731371179230862, 0.52, 0.2921078444612278, 0.52, -0.12078605432356174, 0.52])
# a = np.array([-0.33234656, 0.09565406, -0.020421324, 0.6233484, 0.40746462, 0.5946801, 0.17966142, 0.54043645])

# #Convert the model
# export_dir = 'model2_tf/saved_model.pb'
# converter = tf.lite.TFLiteConverter.from_frozen_graph(
#     graph_def_file=export_dir,
#     input_arrays=['input'],
#     input_shapes={'input' : [1, 128, 128, 3]},
#     output_arrays=['MobilenetV1/Predictions/Softmax'],
# )
# converter.optimizations = {tf.lite.Optimize.DEFAULT}
# converter.change_concat_input_ranges = True
# tflite_model = converter.convert()

# import pathlib
# tflite_model_file = pathlib.Path('model2.tflite')
# tflite_model_file.write_bytes(tflite_model)

# # #Load TFLite Model and Allocate Tensors
# interpreter = tf.lite.Interpreter(model_content=tflite_model)
# interpreter.allocate_tensors()

# # #Get input and output tensors
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# print(input_details)
# print(output_details)

# to_predict = np.array([[10.0]], dtype=np.float32)
# print(to_predict)
# interpreter.set_tensor(input_details[0]['index'], to_predict)
# interpreter.invoke()
# tflite_results = interpreter.get_tensor(output_details[0]['index'])
# print(tflite_results)


# loaded_model = tf.saved_model.load('model2_tf_TEST2')

#https://stackoverflow.com/questions/51278213/what-is-the-use-of-a-pb-file-in-tensorflow-and-how-does-it-work
#HOW TO LOAD Protobuf
# def load_pb(path_to_pb):
#     with tf.gfile.GFile(path_to_pb, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name='')
#         return graph

# graph = load_pb("model2_tf/saved_model.pb")
# input = graph.get_tensor_by_name('input:0')
# output = graph.get_tensor_by_name('output:0')
# print(input)

# model = tf.keras.models.load_model('model2_tf')

# export_dir = 'model2_tf'
# model = tf.saved_model.load(export_dir)

# export_dir = "model2_tf"
# converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
# tflite_model = converter.convert()