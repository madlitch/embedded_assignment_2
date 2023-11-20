import os
import time

from matplotlib import pyplot as plt
from imageai.Detection import VideoObjectDetection

execution_path = os.getcwd()
input_file_name = "traffic.mp4"
output_file_name = "video_frame_analysis"

color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow',
               'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure', 'refrigerator': 'gold',
               'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry',
               'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 'apple': 'cobalt',
               'parking meter': 'aliceblue', 'sandwich': 'skyblue', 'skis': 'deepskyblue', 'microwave': 'peacock',
               'knife': 'cadetblue', 'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey',
               'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen',
               'remote': 'forestgreen', 'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki',
               'skateboard': 'palegoldenrod', 'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood',
               'orange': 'melon', 'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown',
               'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 'car': 'silver',
               'bowl': 'maroon', 'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink',
               'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple',
               'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue',
               'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue', 'teddy bear': 'peacock',
               'clock': 'lightcyan', 'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream',
               'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 'person': 'honeydew',
               'surfboard': 'palegreen', 'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow',
               'toaster': 'ivory', 'stop sign': 'beige', 'couch': 'khaki'}

resized = False

start = time.time()


def for_frame(frame_number, output_array, output_count, returned_frame):
    plt.clf()

    this_colors = []
    labels = []
    sizes = []

    for item in output_count:
        labels.append(item + " = " + str(output_count[item]))
        sizes.append(output_count[item])
        this_colors.append(color_index[item])

    global resized

    if not resized:
        manager = plt.get_current_fig_manager()
        manager.resize(1000, 500)
        resized = True

    plt.subplot(1, 2, 1)
    plt.title("Frame : " + str(frame_number))
    plt.axis("off")
    plt.imshow(returned_frame, interpolation="none")

    plt.subplot(1, 2, 2)
    plt.title("Analysis: " + str(frame_number))
    plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct="%1.1f%%")

    plt.pause(0.01)
    now = time.time()
    print("processed frame {frame} : time elapsed {time:.2f}s".format(frame=frame_number, time=(now - start)))


video_detector = VideoObjectDetection()
video_detector.setModelTypeAsRetinaNet()
video_detector.setModelPath(os.path.join(execution_path, "retinanet.pth"))
video_detector.loadModel()

print("Starting Analysis")

video_detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "traffic.mp4"),
                                      output_file_path=os.path.join(execution_path, "video_frame_analysis"),
                                      frames_per_second=20,
                                      per_frame_function=for_frame,
                                      minimum_percentage_probability=30,
                                      return_detected_frame=True)

end = time.time()
print("Finished Analysis in {time:.2f} seconds".format(time=(end - start)))
