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
* No black and white / monochrome
* No sketches
* No 3D art

With those requirements, the following filters were used to download image data:

**SFW (Safe For Work) images:**

```rating:s type:jpg type:png order:score -human -photo -big_belly -overweight -text -inflation -pool_toy -black_and_white -monochrome -sketch -3d -babyfur -diaper -shitpost -aged_down -reaction_image -hyper_muscles -seductive -patreon_logo -young -wide_hips -doom_(series) -food_creature -looking_at_viewer -big_breasts -convenient_censorship -not_furry```

**NSFW (Not Safe For Work) images:**

```rating:e -rating:s type:jpg type:png order:score -human -photo -overweight -text -scat -watersports -inflation -pool_toy -black_and_white -monochrome -sketch -3d -babyfur -shitpost -aged_down -reaction_image -hyper_muscles -patreon_logo -young -doom_(series) -food_creature -big_breasts -not_furry```

For both the SFW and NSFW filters, some content that I considered extreme was removed since there's some art with lots of unethical or very gross subjects.

After the content was downloaded, I used [Caesium Image Compressor](https://saerasoft.com/caesium/), which is open source, to compress all the PNG images and get them uploaded to my Google Account without issues since I have limited space and I don't want to use the university's cloud storage. All the files went from 9 GB to around 600 MB. 

After that, I created the following file structure to store the images:
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
* Deleted the images that had text and other unwanted noisy things that could affect the training.

After the manual review, I counted all the SFW and NSFW images using ```Ctrl + A``` on the file explorer and then calculated the 10% of the file count to manually select and move images to the test folders.

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
The shear range is moderated, so the model can still know how things should look like. 
The zoom range is minimal too, so we don't lose important details of the images. 

I think most of these transformations are good but in moderated values, it really depends on the kind of images you want your model to learn from too.

I zipped all the files and uploaded them [here](https://drive.google.com/drive/folders/1r-uJWHH_A7MWnHDd6ZcGFRKvapNyBV8Z?usp=share_link). Furthermore, I will decompress the contents of the zips later. 

### Models
During the project, I made 8 different models that had different results, but I will be showing the 3 best ones I got and how they did perform after the VGG16 explanation.

### Things that made the models perform better
* I did 3 versions of the zips to remove images that could cause noise.

* The main change between the zip versions was moving suggestive images to the NSFW side, so it could be more "friendly" about being SFW.

* The second change between the zip versions was cutting what is not important, like most of the background of an image.

* I changed constantly the number of image batches for the data generator from 8 to 16, 32 and 64. Those changes had an impact on how fast it trained but also on the level of noise it generated, being more noisy with 8 or less and less noisy but not so accurate with values to up to 64. I decided that the batch size could be 16 as a base with the changed I observed during many trial and error cases.

* For the optimizer, after trial and error, I saw Adam performed better when making dynamic changes to the learning rate

* For the learning rate, I found out that for this case, values between 2e-4 and 4e-4 were working good. I stuck with the 2e-4 value at the end.

## **Custom CNN Model** (first model):
For this model, I had a version 1 of the image dataset which contained raw SFW and NSFW data downloads with a bit of human review. ```images_v1.zip``` *(currently deleted, my GDrive storage is almost full)*

The main problem was the data: "SFW" suggestive images were affecting the model's learning.

Another problem is that this model was too simple for the complexity of the data shown in the images, like lots of art styles, colors, species and more characteristics the furry art has in general.

*Retrained with the v3 version of the images zip to check differences.

Testing with outside the dataset images, the model is able to detect some NSFW characteristics of images, but it fails at detecting very obvious things that cover almost half of the image and tiny things that are part of a few pixels in size.

```py
CNN1 = models.Sequential()
CNN1.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape = (256,256, 3)))
CNN1.add(layers.MaxPooling2D((2, 2)))
CNN1.add(layers.Flatten())
CNN1.add(layers.Dense(1, activation="sigmoid"))
```

**The test accuracy for this model *(using ```images_v1.zip```)* was 45%**
*When retrained with ```images_v3.zip``` it got to 56%*

## **CNN Model with more layers** (second model):

This second model was able to detect better big and some small characteristics of NSFW images, now, there's a difficulty with detecting explicit parts that are the same color as the character's fur.

This tends to usually fail with dark environment images and when there is lack of detail or too much detail.

For example, when there's a simple drawing of a SFW and a NSFW character, the model gets confused and doesn't do a well prediction. The same happens with super realistic drawings that show too much detail and the model gets confused too.

This was initially trained with the zip ```images_ver2.zip``` file located in the Google Drive folder.*

*Retrained with the v3 zip of the images to check differences.

```py
CNN2 = models.Sequential()
CNN2.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape = (256,256, 3)))
CNN2.add(layers.MaxPooling2D((2, 2)))
CNN2.add(layers.Dropout(0.3))
CNN2.add(layers.Conv2D(32, (3, 3), activation='relu'))
CNN2.add(layers.MaxPooling2D((2, 2)))
CNN2.add(layers.Dropout(0.3))
CNN2.add(layers.Conv2D(64, (3, 3), activation='relu'))
CNN2.add(layers.MaxPooling2D((2, 2)))
CNN2.add(layers.Flatten())
CNN2.add(layers.Dense(64, activation='relu'))
CNN2.add(layers.Dropout(0.3))
CNN2.add(layers.Dense(1, activation="sigmoid"))
```
**The test accuracy for this model *(using ```images_v2.zip```)* was 64%**
*When retrained with ```images_v3.zip``` it got to 67%*

## Quick investigations for the VGG16 model

Based on the [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) paper, I took a high resolution input like they did and added convolutional filters using (3,3) kernel sizes and ReLu activations because that function only gives us positive numbers and if there are any negatives we just get zero, so that's good for our model training process.

With the usage of Pooling layers, we reduce the number of parameters created by the convolutional steps.

Finally, we get our output doing a dense layer of only 1 parameter using a sigmoid function to get 0 to 1 values.

**This is the model I created for training:**

## **VGG16 model** (third model):

Before I started making this model, I generated a new version of the image dataset that had a better classification of the possible suggestive images to could them as NSFW to try to make the model perform even better. ```images_v3.zip```

This model performed better than the other ones on detecting more particular explicit characteristics of the furry characters with more accuracy and situations that may put the characters into a NSFW situation, like beds and dark environments with two or more characters.

When making the final version of this model, I was playing with the non-trainable weights by enabling them and changing things, but I didn't find it reliable. So I started playing with the number of neurons that the dense layers had since VGG16 is not that easy to modify and mess up by accident.

```py
modelWith_VGG16 = models.Sequential()
modelWith_VGG16.add(VGG16_model)
modelWith_VGG16.add(layers.Flatten())
modelWith_VGG16.add(layers.Dense(32,activation='relu')) 
modelWith_VGG16.add(layers.Dropout(0.35))
modelWith_VGG16.add(layers.Dense(64,activation='relu')) 
modelWith_VGG16.add(layers.Dropout(0.35))
modelWith_VGG16.add(layers.Dense(1,activation='sigmoid'))
```

**The test accuracy for this model *(using ```images_v3.zip```)* was 69%** 

## Model results summary
* **Model 1:** 45% test acc *(using ```images_v1.zip```)*
* **Model 2:** 64% test acc *(using ```images_v2.zip```)*
* **Model 3:** 69% test acc *(using ```images_v3.zip```)*


## Conclusions and future work

Surprisingly, the VGG16 base model training performed better at detecting more characteristics of the images that can make predictions to define if is SFW or NSFW.

Sadly, all the test folder results are always below our training accuracy.

I did manual checks with each model with the section that is below this part to check what did actually change. Here's what changed:

**Model 1:** It predicted many NSFW images as SFW because the first version of the images was confusing the model with suggestive images.

**Model 2:** It got better at predicting NSFW images, but it wasn't able to detect some characteristics.

**Model 3:** It got better at predicting if the images are NSFW by detecting more particular characteristics such as bigger and smaller things and a specific type of liquid.

I was thinking at first the model was only overfitting, but there was more...

* As I explained before, all the models were not learning well because of problems with similar colors between the important parts that we need to detect, the complexity and simplicity of many images that we have and more.

* To get a better accuracy, we could continue modifying our dataset and model, adding more images and making better classifications.

* The other thing we can do it make a main model that can detect the complexity of the image, then, two separate models that train on complex images and simple images in order to detect if there's a SFW or a NSFW image present.

* For the problem with similar color on the parts we want to detect, it could be possible to solve it by training on grayscale values and making the model learn specifically from the shapes of the drawings to detect if there is something SFW or NSFW.

So, this is what I learned about making an image classifier for the dark side of the furry fandom.

**I really want to give a big thanks to my professor in guiding me on how to make my model perform better. He was always there to help. 😊**

To all the GitHub watchers, **thank you for following my project!**

I have a question for you too...

What are you doing here? Can you guys email me at manoloramirezpintor@gmail.com ?

Alright. That's all. Bye, bye! 👋
