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

Now, I needed to get data augmentation for the dataset. I used the code [NSFW_Furry_Image_Detector_A01706155.ipynb](./NSFW_Furry_Image_Detector_A01706155.ipynb) to create the datagen and run a very simple model to start the generation of the images. I'm not sure if this is the only way but for now I took this approach to make the new images generation.

I zipped all of the files and uploaded them [here](https://drive.google.com/drive/folders/1r-uJWHH_A7MWnHDd6ZcGFRKvapNyBV8Z?usp=share_link). I will uncompress the contents of the zips later.

For now, this is everything I've been doing these days. Thanks! 😊
