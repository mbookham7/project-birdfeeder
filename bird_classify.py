# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/python3

"""
Coral Smart Bird Feeder

Uses ClassificationEngine from the EdgeTPU API to analyze animals in
camera frames. Sounds a deterrent if a squirrel is detected.

Users define model, labels file, storage path, deterrent sound, and
optionally can set this to training mode for collecting images for a custom
model.

"""
import argparse
import time
import logging
import json
from PIL import Image
from playsound import playsound

from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters.classify import get_classes

import gstreamer


def save_data(image, results, path, ext='png'):
    """Saves camera frame and model inference results
    to user-defined storage directory."""
    tag = '%010d' % int(time.monotonic()*1000)
    name = '%s/img-%s.%s' % (path, tag, ext)
    image.save(name)
    print('Frame saved as: %s' % name)
    logging.info('Image: %s Results: %s', tag, results)
    # Save results as JSON
    results_name = '%s/img-%s.json' % (path, tag)
    with open(results_name, 'w') as f:
        json.dump(results, f)


def print_results(start_time, last_time, end_time, results):
    """Print results to terminal for debugging."""
    inference_rate = ((end_time - start_time) * 1000)
    fps = (1.0/(end_time - last_time))
    print('\nInference: %.2f ms, FPS: %.2f fps' % (inference_rate, fps))
    for label, score in results:
        print(' %s, score=%.2f' % (label, score))


def do_training(results, last_results, top_k):
    """Compares current model results to previous results and returns
    true if at least one label difference is detected. Used to collect
    images for training a custom model."""
    new_labels = [label[0] for label in results]
    old_labels = [label[0] for label in last_results]
    shared_labels = set(new_labels).intersection(old_labels)
    if len(shared_labels) < top_k:
        print('Difference detected')
        return True
    return False


def user_selections():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='.tflite model path')
    parser.add_argument('--labels', required=True,
                        help='label file path')
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of classes with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='class score threshold')
    parser.add_argument('--storage', required=True,
                        help='File path to store images and results')
    parser.add_argument('--sound', required=True,
                        help='File path to deterrent sound')
    parser.add_argument('--print', default=False, required=False,
                        help='Print inference results to terminal')
    parser.add_argument('--training', action='store_true',
                        help='Training mode for image collection')
    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='File path of .tflite file.')
    parser.add_argument('--labels', required=True, help='File path of labels file.')
    parser.add_argument('--videosrc', required=True, help='Device path of video source.')
    parser.add_argument('--storage', required=True, help='Directory to store captured images.')
    parser.add_argument('--sound', required=True, help='File path of deterrent sound.')
    parser.add_argument('--print', type=bool, default=False, help='Print results to terminal.')
    args = parser.parse_args()

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = common.input_size(interpreter)

    # Define the labels for animals to deter
    DETER_LABELS = [
        'fox squirrel, eastern fox squirrel, Sciurus niger',
        'heron',
        'otter',
        'cat',
        'mink'
    ]

    def user_callback(image, svg_canvas):
        start_time = time.monotonic()
        common.set_input(interpreter, image)
        interpreter.invoke()
        results = get_classes(interpreter, top_k=1)
        end_time = time.monotonic()

        if args.print:
            print_results(start_time, last_time, end_time, results)

        for result in results:
            if labels[result.id] in DETER_LABELS:
                playsound(args.sound)
                save_data(image, results, args.storage)
                break

    gstreamer.run_pipeline(user_callback, src=args.videosrc, appsink_size=inference_size)

if __name__ == '__main__':
    main()
