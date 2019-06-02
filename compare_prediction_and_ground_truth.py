import copy
import csv

def parse_ground_truth_file(gt_file_path):
	with open(gt_file_path) as gt_file:
		bboxes = []
		bbox_reader = csv.reader(gt_file, delimiter=' ')
		for row in bbox_reader:
			bbox = {}
			bbox['class_name'] = row[0]
			bbox['x_min'] = int(row[1])
			bbox['y_min'] = int(row[2])
			bbox['x_max'] = int(row[3])
			bbox['y_max'] = int(row[4])

			bboxes.append(bbox)

		return bboxes

def parse_prediction_file(prediction_file_path):
	with open(prediction_file_path) as prediction_file:
		bboxes = []
		bbox_reader = csv.reader(prediction_file, delimiter=' ')
		for row in bbox_reader:
			bbox = {}
			bbox['class_name'] = row[0]
			bbox['conf'] = float(row[1])
			bbox['x_min'] = int(row[2])
			bbox['y_min'] = int(row[3])
			bbox['x_max'] = int(row[4])
			bbox['y_max'] = int(row[5])

			bboxes.append(bbox)

		return bboxes

def compute_box_area(box):
	return max(0, box['x_max'] - box['x_min'] + 1) * max(0, box['y_max'] - box['y_min'] + 1)

def compute_iou(box1, box2):
	# find the intersection
	interbox = {}
	interbox['x_min'] = max(box1['x_min'], box2['x_min'])
	interbox['y_min'] = max(box1['y_min'], box2['y_min'])
	interbox['x_max'] = min(box1['x_max'], box2['x_max'])
	interbox['y_max'] = min(box1['y_max'], box2['y_max'])

	inter_area = compute_box_area(interbox)

	box1_area = compute_box_area(box1)
	box2_area = compute_box_area(box2)

	union_area = box1_area + box2_area - inter_area

	iou = inter_area / float(union_area)

	return iou, inter_area, union_area



def compare_precition_and_ground_truth(predictions, ground_truths, iou_threshold=0.5, conf_threshold=0.3):
	# remove low confidence predictions
	filtered_predictions = []
	for pred_bbox in predictions:
		if pred_bbox['conf'] >= conf_threshold:
			filtered_predictions.append(pred_bbox)
	predictions = filtered_predictions

	# greedily match the predicted bboxes and actual bboxes
	matches = []
	prediction_indices = set(range(len(predictions)))
	ground_truth_indices = set(range(len(ground_truths)))

	while len(prediction_indices) > 0 and len(ground_truth_indices) > 0:
		# find the maximal match
		max_iou = -1.0
		max_match = {}
		for pr_idx in prediction_indices:
			for gt_idx in ground_truth_indices:
				iou, inter_area, union_area = compute_iou(predictions[pr_idx], ground_truths[gt_idx])
				if iou > max_iou:
					max_iou = iou
					max_match['pr_idx'] = pr_idx
					max_match['gt_idx'] = gt_idx
					max_match['conf'] = predictions[pr_idx]['conf']
					max_match['iou'] = iou
					max_match['inter_area'] = inter_area
					max_match['union_area'] = union_area

		# stop if max_iou is below iou_threshold
		if max_iou < iou_threshold:
			break


		matches.append(copy.deepcopy(max_match))
		prediction_indices.remove(max_match['pr_idx'])
		ground_truth_indices.remove(max_match['gt_idx'])


	num_predictions = len(predictions)
	num_ground_truths = len(ground_truths)
	num_matches = len(matches)

	total_inter_area = 0
	for match in matches:
		total_inter_area += match['inter_area']

	total_pred_area = 0
	for pred_bbox in predictions:
		total_pred_area += compute_box_area(pred_bbox)

	total_gt_area = 0
	for gt_bbox in ground_truths:
		total_gt_area += compute_box_area(gt_bbox)

	total_union_area = total_pred_area + total_gt_area - total_inter_area

	iou = total_inter_area / float(total_union_area)

	summary = {'iou': iou, 'total_pred_area': total_pred_area, 'total_gt_area': total_gt_area,
	           'num_matches': num_matches, 'num_preds': num_predictions, 'num_gts': num_ground_truths}


	return matches, summary




if __name__ == '__main__':

	summary_file_path = '/home/yuan/Learning/cs230/data/mAP/summary.txt'
	with open(summary_file_path, 'a+') as summary_file:

	#TODO: update this to not hardcode the number of files
		for i in xrange(2064):
			gt_bboxes = parse_ground_truth_file('/home/yuan/Learning/cs230/data/mAP/ground-truth/{}.txt'.format(i))
			pr_bboxes = parse_prediction_file('/home/yuan/Learning/cs230/data/mAP/predicted/{}.txt'.format(i))
			matches, summary = compare_precition_and_ground_truth(pr_bboxes, gt_bboxes)

			# print('ground truth')
			# for gt_bbox in gt_bboxes:
			# 	print(gt_bbox)

			# print('predictions')
			# for pr_bbox in pr_bboxes:
			# 	print(pr_bbox)

			# print('matches')
			# for match in matches:
			# 	print(match)

			print('summary')
			for key, val in summary.iteritems():
				print('{}: {:.2f}'.format(key, val))

			summary_file.write(','.join([str(i), '{:.2f}'.format(summary['iou']), str(summary['total_pred_area']), 
				str(summary['total_gt_area']), str(summary['num_matches']), str(summary['num_preds']), 
				str(summary['num_gts'])]) + "\n")
