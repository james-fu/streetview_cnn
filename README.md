SVHN Classification & Detection using Convolutional Neural Networks

Classifying unconstrained natural photographs requires a pipeline that that pre-processes the
images so that the neural network can classify the 48 by 48 BGR image correctly. First we must
process the entire image by subtracting the mean and creating a pyramid. For each layer of the pyramid
we divide the image by half. Three layers were created and then a 48 by 48 sliding window was applied
to each. The stride was reduced on each subsequent layer since it represented a larger movement in the
original image. The sliding window represented bounding boxes of sizes 48, 96 and 192 in the original
image. These window images were passed to the CNN model of our choosing. In this case we used the
VGG16 Transfer Model since it produced the best results.

Preproccessed SVHN Dataset and model weights can be found on google drive. Place model weights in "saved" folder and dataset in "data" folder.The training data consists of 230k multi-digit images and 78k negative examples. The
validation set contains 5k positive samples and the testing set contains 13k positive samples.

Link: https://drive.google.com/drive/folders/1a8akNFYAdycMfGJTW9NcBqzh1p5G5GfR?usp=sharing



To run the file just open run_v3.py and click run. It will grab the 5 images in the current folder and classify them. The first image takes a while but then the other 4 are rather quick. It will place these images in the graded_imgs folder. You can see the results by checking the folder now. Delete the images and run the file if you want to see new images. As long as keras, numpy, scipy and other common libraries are implemented then the file should run without a problem.

Only the detection.py, processing.py, classification.py are ran. The custom_model.py & vgg_model.py are just showing the training files for those models.

