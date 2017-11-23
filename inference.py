'''
Run inference using trained model
'''
import tensorflow as tf
from settings import *
from model import SSDModel
from model import ModelHelper
from model import nms
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import math
import os
import time
from PIL import Image
import cv2
def show(img):
  window = 'window'
  cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
  cv2.imshow(window, img)
  cv2.waitKey()
  cv2.destroyAllWindows()
def run_inference(image, model, sess, mode, sign_map):
	"""
	Run inference on a given image

	Arguments:
		* image: Numpy array representing a single RGB image
		* model: Dict of tensor references returned by SSDModel()
		* sess: TensorFlow session reference
		* mode: String of either "image", "video", or "demo"

	Returns:
		* Numpy array representing annotated image
	"""
	# Save original image in memory
	image = np.array(image)
	image_orig = np.copy(image)

	# Get relevant tensors
	x = model['x']
	is_training = model['is_training']
	preds_conf = model['preds_conf']
	preds_loc = model['preds_loc']
	probs = model['probs']

	# Convert image to PIL Image, resize it, convert to grayscale (if necessary), convert back to numpy array
	image = Image.fromarray(image)
	orig_w, orig_h = image.size
	if NUM_CHANNELS == 1:
		image = image.convert('L')  # 8-bit grayscale
	image = image.resize((IMG_W, IMG_H), Image.LANCZOS)  # high-quality downsampling filter
	image = np.asarray(image)

	images = np.array([image])  # create a "batch" of 1 image
	if NUM_CHANNELS == 1:
		images = np.expand_dims(images, axis=-1)  # need extra dimension of size 1 for grayscale

	# Perform object detection
	t0 = time.time()  # keep track of duration of object detection + NMS
	preds_conf_val, preds_loc_val, probs_val = sess.run([preds_conf, preds_loc, probs], feed_dict={x: images, is_training: False})
	if mode != 'video':
		print('Inference took %.1f ms (%.2f fps)' % ((time.time() - t0)*1000, 1/(time.time() - t0)))

	# Gather class predictions and confidence values
	y_pred_conf = preds_conf_val[0]  # batch size of 1, so just take [0]
	y_pred_conf = y_pred_conf.astype('float32')
	prob = probs_val[0]

	# Gather localization predictions
	y_pred_loc = preds_loc_val[0]

	# Perform NMS
	boxes = nms(y_pred_conf, y_pred_loc, prob)
	if mode != 'video':
		print('Inference + NMS took %.1f ms (%.2f fps)' % ((time.time() - t0)*1000, 1/(time.time() - t0)))

	# Rescale boxes' coordinates back to original image's dimensions
	# Recall boxes = [[x1, y1, x2, y2, cls, cls_prob], [...], ...]
	scale = np.array([orig_w/IMG_W, orig_h/IMG_H, orig_w/IMG_W, orig_h/IMG_H])
	if len(boxes) > 0:
		boxes[:, :4] = boxes[:, :4] * scale

	# Draw and annotate boxes over original image, and return annotated image
	image = image_orig
	for box in boxes:
		# Get box parameters
		box_coords = [int(round(x)) for x in box[:4]]
		cls = int(box[4])
		cls_prob = box[5]

		# Annotate image
		image = cv2.rectangle(image, tuple(box_coords[:2]), tuple(box_coords[2:]), (0,255,0))
		label_str = '%s %.2f' % (sign_map[cls], cls_prob)
		image = cv2.putText(image, label_str, (box_coords[0], box_coords[1]), 0, 0.5, (0,255,0), 1, cv2.LINE_AA)
	return image


def generate_output(mode):
	"""
	Generate annotated images, videos, or sample images, based on mode
	"""
	# First, load mapping from integer class ID to sign name string
	sign_map = {}
	with open('signnames.csv', 'r') as f:
		for line in f:
			line = line[:-1]  # strip newline at the end
			sign_id, sign_name = line.split(',')
			sign_map[int(sign_id)] = sign_name
	sign_map[0] = 'background'  # class ID 0 reserved for background class

	# Launch the graph
	path = 'model/model.ckpt'
	with tf.Graph().as_default(), tf.Session() as sess:
		# "Instantiate" neural network, get relevant tensors
		model = SSDModel()

		# Load trained model
		saver = tf.train.Saver()
		print('Restoring previously trained model at %s' % path)
		saver.restore(sess, path)
		image_orig = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
		t = time.time()
		image_orig = cv2.resize(image_orig, (int(image_orig.shape[1]/2), int(image_orig.shape[0]/2)))
		image = run_inference(image_orig, model, sess, mode, sign_map)
		print(image.shape)
		print(time.time() - t)
		show(image)
		
generate_output('demo')