
# YOLO Training on Custom Data

This repository contains the implementation of YOLOv3 trained on a custom data containing 4 classes:
* Hayabusa
* Iphone
* Pistol
* Football

A) OpenCV Yolo:  [SOURCE](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)
1. Run this above code on your laptop or Colab. 
2. Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
3. Run this image through the code above. 
4. Upload the link to GitHub implementation of this
5. Upload the annotated image by YOLO. 

B) Training Custom Dataset on Colab for YoloV3
1. Refer to this Colab File:  [LINK](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS)
2. Refer to this GitHub [Repo](https://github.com/theschoolofai/YoloV3)
3. Download this [dataset](https://drive.google.com/file/d/1sVSAJgmOhZk6UG7EzmlRjXfkzPxmpmLy/view?usp=sharing). This was annotated by EVA5 Students. Collect and add 25 images for the following 4 classes into the dataset shared:
\
a) class names are in custom.names file. 
\
b) you must follow exact rules to make sure that you can train the model. Steps are explained in the README.md file on github repo link above.
\
c) Once you add your additional 100 images, train the model

4. Once done:
\
a) [Download](https://www.y2mate.com/en19) a very small (~10-30sec) video from youtube which shows your classes. 
\
b) Use [ffmpeg](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence) to extract frames from the video. 
\
c) Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
\
d) Infer on these images using detect.py file. **Modify** detect.py file if your file names do not match the ones mentioned on GitHub. 
\
python detect.py --conf-three 0.3 --output output_folder_name
\
e) Use  [ffmpeg](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence)  to convert the files in your output folder to video
\
f) Upload the video to YouTube. 
\
g) Also run the model on 16 images that you have collected (4 for each class)
\
h) Share the link to your GitHub project with the steps mentioned above - 1000 pts (only if all the steps were done, and it resulted in a trained model that you could run on video/images)
\
i) Share the link to your YouTube video - 500 pts
\
j) Share the link of your YouTube video shared on LinkedIn, Instagram, medium, etc! You have no idea how much you'd love people complimenting you! [TOTALLY OPTIONAL] - bonus points
\
k) Share the link to the readme file where we can find the result of your model on YOUR 16 images. - 500 pts
