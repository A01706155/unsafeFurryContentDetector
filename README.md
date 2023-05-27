# A01706155 - Manolo Ramírez Pintor
## Unsafe Furry Content Detector - CNN

## Dataset link:
**IMPORTANT NOTICE: ACCESS THE AUGMENTED AND NSFW FOLDERS UNDER YOUR OWN RISK**

[Furry - Google Drive Folder](https://drive.google.com/drive/folders/1r-uJWHH_A7MWnHDd6ZcGFRKvapNyBV8Z?usp=share_link)

### Description
This repository will be used to create an AI model using Convolutional Neural Networks which is going to be able to detect if some content is SFW (Safe For Work) or NSFW (Not Safe For Work / Explicit).

This CNN will determine if a digital art furry piece is safe or not safe content, some use cases for this could be using it for filtering content on safe artist platforms or any other social media platforms.

### Getting the dataset
In order to get the data, the software [e621_downloader](https://github.com/McSib/e621_downloader) is used to harvest image data from the site. This site can be used by anyone if their age is +18 since the content in that website contains lots of mature art about the furry fandom.

When using the downloader software, some filters were created to try to get the best images possible to train the convolutional neural network. The main requirements for the images are:
* Only JPG and PNG
* Ordered by best score
* No text
* No black and white / monochome
* No sketches
* No 3D art

With those requirements, the following filters were used to download image data:

**SFW (Safe For Work) images:**

```rating:s type:jpg type:png order:score -human -photo -big_belly -overweight -text -inflation -pool_toy -black_and_white -monochrome -sketch -3d -babyfur -diaper -shitpost -aged_down -reaction_image -hyper_muscles -seductive -patreon_logo -young -wide_hips -doom_(series) -food_creature -looking_at_viewer -big_breasts -convenient_censorship -not_furry```

**NSFW (Not Safe For Work) images:**

```rating:e -rating:s type:jpg type:png order:score -human -photo -overweight -text -scat -watersports -inflation -pool_toy -black_and_white -monochrome -sketch -3d -babyfur -shitpost -aged_down -reaction_image -hyper_muscles -patreon_logo -young -doom_(series) -food_creature -big_breasts -not_furry```

For both the SFW and NSFW filters, some content that I considered extreme was removed since there's some art with lots of unethical or very gross subjects.

After the content was downloaded, I used [Caesium Image Compressor](https://saerasoft.com/caesium/), which is open source, to compress all of the png images and get them uploaed to my Google Account without issues since I have limited space and I don't want to use the university's cloud storage. All of the files went from 9 GB to around 600 MB. 

After that, I created the following file stucture to store the images:
```
images:
    train:
        sfw
        nsfw
    test:
        sfw
        nsfw
```

Now, I reviewed both SFW and NSFW images manually since I noticed there were some memes and unwanted content that could add noise to the dataset. I edited and deleted some contents to make it more clean.

What I had to do was the following:
* Cut the images to focus the main aspects of the art.
* Deteleted the images that had text and other unwanted noisy things that could affect the training.

After the manual review, I counted all of the SFW and NSFW images using ```Ctrl + A``` on the file explorer and then calculated the 10% of the file count to manually select and move images to the test folders.

### Data augmentation
Now, I needed to get data augmentation for the dataset. I used the code [NSFW_Furry_Image_Detector_A01706155.ipynb](./NSFW_Furry_Image_Detector_A01706155.ipynb) to create the datagen and run a very simple model to start the generation of the images. 

The following code fragment was used for data augmentation. 
```py
train_datagen = ImageDataGenerator(
rescale = 1./255,
rotation_range = 25,
width_shift_range = 0.1,
height_shift_range = 0.1,
shear_range = 0.2,
zoom_range = 0.15,
horizontal_flip = True)
```

I set a rotation range of 25 because I don't feel like big rotations are going to make the model learn correctly. 
The shift ranges are minimal in order to avoid the model learning from very stretched images that could add noise. 
The shear range is moderated so the model can still know how things should look like. 
The zoom range is minimal too so we don't lose important details of the images. 

I think most of these transformations are good but in moderated values, it really depends on the kind of images you want your model to learn from too.

I zipped all of the files and uploaded them [here](https://drive.google.com/drive/folders/1r-uJWHH_A7MWnHDd6ZcGFRKvapNyBV8Z?usp=share_link). I will uncompress the contents of the zips later. 

### Model
A model was trained for the detection of the safe and unsafe furry images. The train accuracy got to 70% but the accuracy with the test data was of 65% meaning. This means my current model is overfitting and not learning properly.

We will use this convolutional neural network model based on how the [VGG16 model](https://datagen.tech/guides/computer-vision/vgg16/) was made to take advantage of having a base and modify it depending on what we see while training.

Based on [VGG16](https://datagen.tech/guides/computer-vision/vgg16/), I took a high resolution input like they did and added convolutional filters using using (3,3) kernel sizes and ReLu activations because that function only gives us positive numbers and if there's any negatives we just get zero, so that's good for our model training process.

With the usage of Pooling layers, we reduce the number of parameters created by the convolutional steps.

Finally, we get our output doing a dense of only 1 parameter using a sigmoid function to get 0 to 1 values.

This is the model I created for training:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 254, 254, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 127, 127, 32)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 125, 125, 64)      18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 62, 62, 64)       0         
 2D)                                                             
                                                                 
 dropout (Dropout)           (None, 62, 62, 64)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 60, 60, 128)       73856     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 30, 30, 128)      0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 28, 28, 256)       295168    
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 14, 14, 256)      0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 14, 14, 256)       0         
                                                                 
 flatten (Flatten)           (None, 50176)             0         
                                                                 
 dense (Dense)               (None, 128)               6422656   
                                                                 
 dense_1 (Dense)             (None, 1)                 129       
                                                                 
=================================================================
Total params: 6,811,201
Trainable params: 6,811,201
Non-trainable params: 0
```

### Model conclusions
When doing manual/interactive testing with the model, I saw it learned real explicit things from the unsafe content, but since there are some suggestive images in the SFW folder, I think it creates some confusion. To try to make this content detector a bit better, I will move some of the "safe" suggestive images to the "nsfw" folder and see what it does. The last thing I will be doing is using transfer learning to see how the learning of the model changes.

For now, this is everything I've been doing these days. Thanks! 😊
