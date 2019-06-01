import argparse
import csv
import os
from PIL import Image

# v2 considers only license plate 
label_name_to_class_id_dict = {
    '/m/01jfm_': 0,  # Vehicle Registration Plate
}

image_filename_to_size_dict_train = {}
image_filename_to_size_dict_validation = {}
image_filename_to_size_dict_test = {}


# Construct the width and height for each image
def construct_image_id_to_size_dict(mode, image_path_prefix):
    image_filename_to_size_dict = switcher(mode)
    for filename in os.listdir(image_path_prefix):
        if filename.endswith('.jpg'):
            image_path = image_path_prefix+filename
            im = Image.open(image_path)
            width, height = im.size
            image_filename_to_size_dict[filename] = [width, height]
    print("construct_image_id_to_size_dict done for %s!" % mode)


# Reformat row based on image_path x_min, y_min, x_max, y_max, class_id format
def convert_row(mode, row, image_path_prefix):
    image_filename_to_size_dict = switcher(mode)
    image_id = str(row[0])
    image_filename = image_id + '.jpg'
    if image_filename not in image_filename_to_size_dict:
        print('ERROR: This image %s is not in the directory' % image_filename)
        width, height = [1, 1]
    else:
        width, height = image_filename_to_size_dict[image_filename]

    x_min = '%d' % (float(row[4]) * width)
    x_max = '%d' % (float(row[5]) * width)
    y_min = '%d' % (float(row[6]) * height)
    y_max = '%d' % (float(row[7]) * height)
    label_name = str(row[2])
    class_id = str(label_name_to_class_id_dict[label_name])
    return [image_path_prefix + image_filename, x_min, y_min, x_max, y_max, class_id]


# Group the converted_rows by image_path
def group_by_image_id(mode, converted_rows):
    output_dict = {}
    for row in converted_rows:
        cur_image_path = row[0]
        if cur_image_path in output_dict:
            output_dict[cur_image_path] += ' ' + ','.join(row[1:])
        else:
            output_dict[cur_image_path] = cur_image_path + ' ' + ','.join(row[1:])
    print("group_by_image_id done for %s!" % mode)
    return [row for row in output_dict.values()]


def switcher(mode):
    if mode == 'train':
        return image_filename_to_size_dict_train
    elif mode == 'validation':
        return image_filename_to_size_dict_validation
    elif mode == 'test':
        return image_filename_to_size_dict_test
    else:
        print('Fatal Error!!!!')


def convert_annotations(mode, image_path_prefix, input_csv_file_path, output_file_path):
    with open(input_csv_file_path) as input_csv_file:
        converted_rows = []
        input_annotation_reader = csv.reader(input_csv_file, delimiter=',')
        for row in input_annotation_reader:
            if (row[2] in label_name_to_class_id_dict):
                converted_rows.append(convert_row(mode, row, image_path_prefix))

        converted_rows = group_by_image_id(mode, converted_rows)

        with open(output_file_path, 'w') as output_file:
            for row in converted_rows:
                output_file.write(row + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/open_image_1000G/")
    parser.add_argument("--train_annotation", default="./data/dataset/open_image_train_v2.txt")
    # parser.add_argument("--validation_annotation", default="./data/dataset/open_image_validation_v2.txt")
    parser.add_argument("--test_annotation", default="./data/dataset/open_image_test_v2.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):
        os.remove(flags.train_annotation)
    # if os.path.exists(flags.validation_annotation):
    #     os.remove(flags.validation_annotation)
    if os.path.exists(flags.test_annotation):
        os.remove(flags.test_annotation)

    construct_image_id_to_size_dict('train', os.path.join(flags.data_path, 'train/'))
    # construct_image_id_to_size_dict('validation', os.path.join(flags.data_path, 'validation/'))
    construct_image_id_to_size_dict('test', os.path.join(flags.data_path, 'test/'))

    convert_annotations('train', os.path.join(flags.data_path, 'train/'),
                        os.path.join(flags.data_path, 'train-annotations-bbox.csv'),
                        flags.train_annotation)
    # convert_annotations('validation', os.path.join(flags.data_path, 'validation/'),
    #                     os.path.join(flags.data_path, 'validation-annotations-bbox.csv'),
    #                     flags.validation_annotation)
    convert_annotations('test', os.path.join(flags.data_path, 'test/'),
                        os.path.join(flags.data_path, 'test-annotations-bbox.csv'),
                        flags.test_annotation)
