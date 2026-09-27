import argparse
import signal
import time
from pathlib import Path

import cv2

from edge_impulse_linux.image import ImageImpulseRunner

BOX_COLOR = (0, 255, 0)  # Green

def _log(msg: str, *, enabled: bool = True):
	if not enabled:
		return
	print(msg, flush=True)


class _InitTimeout(Exception):
	pass


def _init_runner_with_timeout(runner: ImageImpulseRunner, *, seconds: int, debug: bool):
	# edge_impulse_linux.runner can wait forever for the IPC socket if the EIM never
	# creates it (e.g. wrong arch, missing deps). Put a hard cap on init.
	if seconds <= 0:
		return runner.init(debug=debug)

	def _on_alarm(_signum, _frame):
		raise _InitTimeout(f'EIM init timed out after {seconds}s')

	prev = signal.getsignal(signal.SIGALRM)
	signal.signal(signal.SIGALRM, _on_alarm)
	try:
		signal.alarm(int(seconds))
		return runner.init(debug=debug)
	finally:
		signal.alarm(0)
		signal.signal(signal.SIGALRM, prev)


def parse_paddle_ocr_dictionary(file_path: str) -> list[str]:
	return Path(file_path).read_text(encoding='utf-8').split('\n')


def parse_paddle_ocr_predictor(pred_resp: dict, dictionary: list[str]):
	result = pred_resp.get('result', {})
	freeform = result.get('freeform')
	if freeform is None:
		raise RuntimeError('Predictor model did not return freeform results')
	if not isinstance(freeform, list) or len(freeform) == 0:
		raise RuntimeError('Predictor freeform results are empty')

	tensor = freeform[0]
	if not isinstance(tensor, list):
		raise RuntimeError('Predictor freeform[0] was not a list')

	cols = len(tensor) / (len(dictionary) + 1)
	if cols % 1 != 0:
		raise RuntimeError(
			'Invalid output shape (probably incorrect dictionary?). '
			f'Dict size={(len(dictionary) + 1)}, output tensor length={len(tensor)}, '
			f'detected columns={cols} (expected to be an integer)'
		)

	reshaped: list[list[float]] = []
	step = len(dictionary) + 1
	for ix in range(0, len(tensor), step):
		reshaped.append(tensor[ix:ix + step])

	all_confidences = 0.0
	predicted_chars_length = 0
	output = ''

	for col in reshaped:
		hi_val = 0.0
		hi_ix = -1
		for ix, v in enumerate(col):
			if v > hi_val:
				hi_val = float(v)
				hi_ix = ix

		# index 0 is the blank label
		if hi_ix != 0:
			output += dictionary[hi_ix - 1]
			all_confidences += hi_val
			predicted_chars_length += 1

	if predicted_chars_length == 0:
		return None

	return {
		'output': output,
		'avgConfidence': all_confidences / predicted_chars_length,
	}


def _map_prediction_to_original_image(
	resize_mode: str,
	box_norm: dict,
	img: dict,
):
	# Port of ts/bounding-box-scaling.ts (only the modes this app uses).
	# box_norm: x,y,width,height are normalized 0..1 in reference coordinates.
	x = 0.0
	y = 0.0
	width = 0.0
	height = 0.0

	if resize_mode == 'squash':
		scale_x = img['clientWidth'] / img['referenceWidth']
		scale_y = img['clientHeight'] / img['referenceHeight']
		x = max(0.0, min(box_norm['x'] * img['referenceWidth'] * scale_x, img['clientWidth']))
		y = max(0.0, min(box_norm['y'] * img['referenceHeight'] * scale_y, img['clientHeight']))
		width = min(box_norm['width'] * img['referenceWidth'] * scale_x, img['clientWidth'] - x)
		height = min(box_norm['height'] * img['referenceHeight'] * scale_y, img['clientHeight'] - y)

	elif resize_mode == 'fit-short':
		reference_aspect = img['referenceWidth'] / img['referenceHeight']
		client_aspect = img['clientWidth'] / img['clientHeight']

		scale = 1.0
		offset_x = 0.0
		offset_y = 0.0
		if client_aspect > reference_aspect:
			scale = img['clientHeight'] / img['referenceHeight']
			crop_width = img['clientWidth'] - img['referenceWidth'] * scale
			offset_x = crop_width / 2.0
		else:
			scale = img['clientWidth'] / img['referenceWidth']
			crop_height = img['clientHeight'] - img['referenceHeight'] * scale
			offset_y = crop_height / 2.0

		x = max(0.0, min(box_norm['x'] * img['referenceWidth'] * scale + offset_x, img['clientWidth']))
		y = max(0.0, min(box_norm['y'] * img['referenceHeight'] * scale + offset_y, img['clientHeight']))
		width = min(box_norm['width'] * img['referenceWidth'] * scale, img['clientWidth'] - x)
		height = min(box_norm['height'] * img['referenceHeight'] * scale, img['clientHeight'] - y)

	elif resize_mode == 'fit-long':
		reference_aspect = img['referenceWidth'] / img['referenceHeight']
		client_aspect = img['clientWidth'] / img['clientHeight']
		if client_aspect > reference_aspect:
			scale = img['clientWidth'] / img['referenceWidth']
			pad_height = img['clientHeight'] - img['referenceHeight'] * scale
			pad_y = pad_height / 2.0
			x = max(0.0, min(box_norm['x'] * img['referenceWidth'] * scale, img['clientWidth']))
			y = max(0.0, min(box_norm['y'] * img['referenceHeight'] * scale + pad_y, img['clientHeight']))
			width = min(box_norm['width'] * img['referenceWidth'] * scale, img['clientWidth'] - x)
			height = min(box_norm['height'] * img['referenceHeight'] * scale, img['clientHeight'] - y)
		else:
			scale = img['clientHeight'] / img['referenceHeight']
			pad_width = img['clientWidth'] - img['referenceWidth'] * scale
			pad_x = pad_width / 2.0
			x = max(0.0, min(box_norm['x'] * img['referenceWidth'] * scale + pad_x, img['clientWidth']))
			y = max(0.0, min(box_norm['y'] * img['referenceHeight'] * scale, img['clientHeight']))
			width = min(box_norm['width'] * img['referenceWidth'] * scale, img['clientWidth'] - x)
			height = min(box_norm['height'] * img['referenceHeight'] * scale, img['clientHeight'] - y)
	else:
		# Unknown / unsupported mode
		return {'x': 0.0, 'y': 0.0, 'width': 0.0, 'height': 0.0}

	return {'x': x, 'y': y, 'width': width, 'height': height}


def map_bounding_box_back_to_original_image(
	detector_model_info: dict,
	resized_info: dict,
	bb: dict,
):
	resize_mode = detector_model_info.get('model_parameters', {}).get('image_resize_mode')
	# Align naming with the TS implementation.
	if resize_mode == 'fit-shortest':
		resize_mode = 'fit-short'
	elif resize_mode == 'fit-longest':
		resize_mode = 'fit-long'
	elif resize_mode in (None, '', 'not-reported'):
		resize_mode = 'squash'

	mapped = _map_prediction_to_original_image(
		resize_mode,
		{
			'x': bb['x'] / resized_info['newWidth'],
			'y': bb['y'] / resized_info['newHeight'],
			'width': bb['width'] / resized_info['newWidth'],
			'height': bb['height'] / resized_info['newHeight'],
		},
		{
			'clientWidth': resized_info['originalWidth'],
			'clientHeight': resized_info['originalHeight'],
			'referenceWidth': resized_info['newWidth'],
			'referenceHeight': resized_info['newHeight'],
		},
	)

	x = int(mapped['x'] // 1)
	y = int(mapped['y'] // 1)
	width = int(-(-mapped['width'] // 1))  # ceil
	height = int(-(-mapped['height'] // 1))

	if x < 0:
		x = 0
	if y < 0:
		y = 0
	if x + width > resized_info['originalWidth']:
		width = resized_info['originalWidth'] - x
	if y + height > resized_info['originalHeight']:
		height = resized_info['originalHeight'] - y

	return {
		'label': bb.get('label', ''),
		'value': bb.get('value', 0.0),
		'x': x,
		'y': y,
		'width': width,
		'height': height,
	}


def parse_thresholds_arg(thresholds: str | None):
	if not thresholds:
		return []

	overrides = []
	for opt in thresholds.split(','):
		# ${id}.${key}=${value}
		# value may be float/int or boolean
		try:
			left, value_str = opt.split('=', 1)
			block_id_str, key = left.split('.', 1)
			block_id = int(block_id_str)
			if value_str in ('true', 'false'):
				value = value_str == 'true'
			else:
				value = float(value_str)
		except Exception:
			raise ValueError(f'Failed to parse threshold "{opt}" (expected e.g. 4.min_anomaly_score=35)')

		overrides.append({'id': block_id, 'key': key, 'value': value})
	return overrides


def update_threshold_in_model_info(model_info: dict, block_id: int, key: str, value):
	thresholds = model_info.get('model_parameters', {}).get('thresholds')
	if not isinstance(thresholds, list):
		return
	for th in thresholds:
		if isinstance(th, dict) and th.get('id') == block_id:
			th[key] = value
			return


def main():
	parser = argparse.ArgumentParser(description='Edge Impulse OCR demo (Python)')
	parser.add_argument('--detect-file', required=True, help='Path to detector .eim executable')
	parser.add_argument('--predict-file', required=True, help='Path to predictor .eim executable')
	parser.add_argument('--dict-file', required=True, help='Dictionary file (newline separated) for the predictor')
	parser.add_argument('--camera', default='0', help='OpenCV camera device index (default: 0)')
	parser.add_argument('--rtsp-url', default=None, help='RTSP/HTTP stream URL to use instead of a wired camera (e.g. rtsp://10.0.0.124:554/mjpeg/1)')
	parser.add_argument('--interval-ms', default='0', help='Delay between frames (default: 0)')
	parser.add_argument('--max-frames', default='0', help='Stop after N frames (default: 0 = run forever)')
	parser.add_argument('--min-box-area', default='20', help='Discard boxes with area smaller than this (default: 20)')
	parser.add_argument('--display', action='store_true', help='Show live camera preview with overlay (default)')
	parser.add_argument('--no-display', action='store_true', help='Disable live preview window')
	parser.add_argument('--display-width', default='0', help='Optional: resize preview to this width (0 = no resize)')
	parser.add_argument(
		'--thresholds',
		default=None,
		help='Override model thresholds. E.g. --thresholds 4.min_anomaly_score=35. Comma-separate multiple overrides.',
	)
	parser.add_argument('--verbose', action='store_true', help='Verbose app output (does not enable EIM runner debug)')
	parser.add_argument('--runner-debug', action='store_true', help='Enable verbose EIM runner debug (very noisy)')
	parser.add_argument('--init-timeout-s', default='30', help='Timeout (seconds) for EIM runner startup (default: 30)')
	args = parser.parse_args()

	detect_file = args.detect_file
	predict_file = args.predict_file
	dict_file = args.dict_file
	camera_device = int(args.camera) if str(args.camera).isdigit() else 0
	interval_ms = int(args.interval_ms) if str(args.interval_ms).isdigit() else 0
	max_frames = int(args.max_frames) if str(args.max_frames).isdigit() else 0
	min_box_area = int(args.min_box_area) if str(args.min_box_area).isdigit() else 20
	display_width = int(args.display_width) if str(args.display_width).isdigit() else 0
	show_display = (not args.no_display)
	# If user explicitly passed --display, it wins.
	if args.display:
		show_display = True
	threshold_overrides = parse_thresholds_arg(args.thresholds)
	init_timeout_s = int(args.init_timeout_s) if str(args.init_timeout_s).isdigit() else 30

	predictor_dict = parse_paddle_ocr_dictionary(dict_file)

	_log('Initializing detector model...', enabled=True)
	detector_runner = ImageImpulseRunner(detect_file)
	try:
		detector_model_info = _init_runner_with_timeout(
			detector_runner,
			seconds=init_timeout_s,
			debug=args.runner_debug,
		)
	except Exception as ex:
		_log(f'Failed to initialize detector model: {ex}', enabled=True)
		raise

	_log('Initializing predictor model...', enabled=True)
	predictor_runner = ImageImpulseRunner(predict_file)
	try:
		predictor_model_info = _init_runner_with_timeout(
			predictor_runner,
			seconds=init_timeout_s,
			debug=args.runner_debug,
		)
	except Exception as ex:
		_log(f'Failed to initialize predictor model: {ex}', enabled=True)
		raise

	# Apply CLI threshold overrides to detector
	if threshold_overrides:
		for ov in threshold_overrides:
			obj = {'id': ov['id'], ov['key']: ov['value']}
			detector_runner.set_threshold(obj)
			update_threshold_in_model_info(detector_model_info, ov['id'], ov['key'], ov['value'])

	print('Models:')
	det_proj = detector_model_info.get('project', {})
	pred_proj = predictor_model_info.get('project', {})
	print('    Detector: ', f"{det_proj.get('owner')} / {det_proj.get('name')} (v{det_proj.get('deploy_version')})")
	print('    Predictor:', f"{pred_proj.get('owner')} / {pred_proj.get('name')} (v{pred_proj.get('deploy_version')})")
	thresholds = detector_model_info.get('model_parameters', {}).get('thresholds')
	if isinstance(thresholds, list) and len(thresholds) > 0:
		opts = []
		for th in thresholds:
			if not isinstance(th, dict):
				continue
			th_id = th.get('id')
			th_type = th.get('type')
			for k, v in th.items():
				if k in ('id', 'type'):
					continue
				if isinstance(v, bool):
					opts.append(f'{th_id}.{k}={str(v).lower()}')
				elif isinstance(v, (int, float)):
					rounded = round(float(v), 3)
					opts.append(f'{th_id}.{k}={rounded}')
		if opts:
			print('    Thresholds:', ','.join(opts), '(override via --thresholds <value>)')
	print('')
	if args.rtsp_url:
		print(f'Opening stream {args.rtsp_url}...', flush=True)
		# FFmpeg backend handles RTSP/HTTP network streams
		cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
		# Minimize buffering for lower latency on network streams
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	else:
		print(f'Opening camera device {camera_device}...', flush=True)
		cap = cv2.VideoCapture(camera_device)
		# Best-effort: match Node sample dimensions
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	if not cap.isOpened():
		raise RuntimeError(f'Cannot open video source: {args.rtsp_url or camera_device}')

	if show_display:
		print('Running OCR pipeline. Press Ctrl+C or press q/ESC in the preview window to stop.', flush=True)
	else:
		print('Running OCR pipeline. Press Ctrl+C to stop.', flush=True)
	frame_ix = 0

	try:
		while True:
			ok, frame_bgr = cap.read()
			if not ok:
				time.sleep(0.01)
				continue

			frame_ix += 1
			start_ms = int(time.time() * 1000)

			# OpenCV gives BGR; Edge Impulse helpers expect RGB
			frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

			# Stage 1: detector
			features, _ = detector_runner.get_features_from_image_auto_studio_settings(frame_rgb)
			det_resp = detector_runner.classify(features)
			detector_time_ms = int(time.time() * 1000) - start_ms

			original_h, original_w = frame_rgb.shape[:2]
			detector_w = detector_model_info['model_parameters']['image_input_width']
			detector_h = detector_model_info['model_parameters']['image_input_height']
			original_info = {
				'originalWidth': int(original_w),
				'originalHeight': int(original_h),
				'newWidth': int(detector_w),
				'newHeight': int(detector_h),
			}

			# Stage 2: predictor per bounding box
			predictor_start_ms = int(time.time() * 1000)
			results: list[str] = []
			overlay_items: list[dict] = []
			bbs = det_resp.get('result', {}).get('bounding_boxes') or []
			if isinstance(bbs, list):
				for orig_bb in bbs:
					if not isinstance(orig_bb, dict):
						continue

					bb = map_bounding_box_back_to_original_image(detector_model_info, original_info, orig_bb)
					if bb['width'] <= 0 or bb['height'] <= 0:
						continue
					if bb['width'] * bb['height'] < min_box_area:
						continue

					x, y, w, h = bb['x'], bb['y'], bb['width'], bb['height']
					crop_rgb = frame_rgb[y:y + h, x:x + w]
					if crop_rgb.size == 0:
						continue

					pred_features, _ = predictor_runner.get_features_from_image_auto_studio_settings(crop_rgb)
					pred_resp = predictor_runner.classify(pred_features)
					predicted = parse_paddle_ocr_predictor(pred_resp, predictor_dict)
					if not predicted:
						continue

					text = str(predicted['output'])
					conf = float(predicted['avgConfidence'])
					results.append(f"{text} (conf={conf:.2f})")
					overlay_items.append({
						'x': int(x),
						'y': int(y),
						'w': int(w),
						'h': int(h),
						'text': text,
						'conf': conf,
						'score': float(bb.get('value', 0.0)),
					})

			predictor_time_ms = int(time.time() * 1000) - predictor_start_ms

			if results:
				ts = time.strftime('%H:%M:%S')
				print(
					f"[{ts}] frame={frame_ix} detector={detector_time_ms}ms predictor={predictor_time_ms}ms: "
					+ ' | '.join(results),
					flush=True,
				)
			elif args.verbose:
				ts = time.strftime('%H:%M:%S')
				print(
					f"[{ts}] frame={frame_ix} detector={detector_time_ms}ms predictor={predictor_time_ms}ms: (no text)",
					flush=True,
				)

			if max_frames > 0 and frame_ix >= max_frames:
				break

			if show_display:
				display_frame = frame_bgr.copy()
				# Draw boxes + labels
				for item in overlay_items:
					x1 = max(0, item['x'])
					y1 = max(0, item['y'])
					x2 = max(0, item['x'] + item['w'])
					y2 = max(0, item['y'] + item['h'])
					# green box
					cv2.rectangle(display_frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
					label = f"{item['text']} ({item['conf']:.2f})"
					# background for text
					(font_w, font_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
					bg_x2 = min(display_frame.shape[1] - 1, x1 + font_w + 6)
					bg_y1 = max(0, y1 - font_h - baseline - 6)
					cv2.rectangle(display_frame, (x1, bg_y1), (bg_x2, y1), BOX_COLOR, -1)
					cv2.putText(
						display_frame,
						label,
						(x1 + 3, y1 - 4),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.5,
						(0, 0, 0),
						1,
						cv2.LINE_AA,
					)

				# Top-left status line
				status = f"frame {frame_ix} | det {detector_time_ms}ms | pred {predictor_time_ms}ms"
				cv2.putText(
					display_frame,
					status,
					(10, 25),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					(255, 255, 255),
					2,
					cv2.LINE_AA,
				)

				if display_width and display_width > 0 and display_frame.shape[1] != display_width:
					scale = display_width / float(display_frame.shape[1])
					new_h = int(display_frame.shape[0] * scale)
					display_frame = cv2.resize(display_frame, (display_width, new_h), interpolation=cv2.INTER_AREA)

				cv2.imshow('Edge Impulse OCR (Python)', display_frame)
				key = cv2.waitKey(1) & 0xFF
				if key in (27, ord('q')):
					break

			if interval_ms > 0:
				time.sleep(interval_ms / 1000.0)

	except KeyboardInterrupt:
		print('\nStopping...', flush=True)
	finally:
		try:
			cap.release()
		except Exception:
			pass
		try:
			if show_display:
				cv2.destroyAllWindows()
		except Exception:
			pass
		try:
			detector_runner.stop()
		except Exception:
			pass
		try:
			predictor_runner.stop()
		except Exception:
			pass


if __name__ == '__main__':
	main()