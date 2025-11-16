# Computer vision

Here are detailed notes on Computer Vision (CV), its primary tasks, and the key models used for each, including their pros and cons.

#### 📜 What is Computer Vision?

Computer Vision (CV) is a field of artificial intelligence (AI) that enables computers and systems to "see," interpret, and understand visual information from the world, just as humans do. It aims to extract meaningful information from digital images, videos, and other visual inputs.

At its core, CV is about pattern recognition.

* Classic CV: In the past, this was done with hand-crafted features. Experts would design "filters" to find specific things like edges (e.g., Sobel, Canny operators), corners, or specific textures (e.g., SIFT, SURF).
* Modern (Deep Learning) CV: This approach has almost completely taken over. Instead of hand-crafting features, we use Convolutional Neural Networks (CNNs). A CNN _learns_ the best features automatically.
  * The first layers of a CNN learn to recognize simple edges and colors.
  * Middle layers combine these to find textures and simple shapes.
  * Deep layers combine those to recognize complex objects or parts of objects (like an "eye," a "wheel," or a "face").

***

#### Models Based on Use Case

Here are the primary tasks in CV, the models that solve them, and their trade-offs.

#### 1. 🖼️ Image Classification

This is the most fundamental CV task. The goal is to assign a single label to an entire image.

* Task: "Is this picture a 'Cat,' 'Dog,' or 'Ship'?"

| **Model**                     | **Pros**                                                                                                                                                                                                                       | **Cons**                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| LeNet-5, AlexNet (Historical) | Pioneers. LeNet proved CNNs work for digits. AlexNet (2012) won the ImageNet competition and started the deep learning revolution.                                                                                             | Outdated. By modern standards, they are "shallow" and not very accurate.                                                    |
| VGGNet (e.g., VGG16)          | Simple & Uniform. Proved that just stacking _deep_ layers of simple 3x3 filters works very well. Easy to understand.                                                                                                           | Very heavy. It has a huge number of parameters (weights), making it slow and large to store.                                |
| ResNet (Residual Network)     | The Default Standard. This is the breakthrough model. It introduced "skip connections," which allow the network to be _extremely_ deep (e.g., 50, 101, or 152 layers) without suffering from the "vanishing gradient" problem. | More complex architecture. The skip connections, while effective, make the "block" structure less straightforward than VGG. |
| EfficientNet                  | State-of-the-art balance. It uses a formula to _perfectly balance_ the network's depth, width, and resolution to get the best accuracy for a given computational budget.                                                       | Complex design. The "scaling formula" is not trivial to implement from scratch.                                             |

***

#### 2. 📦 Object Detection

This is the next step up. The goal is to find multiple objects in an image and draw a bounding box around each one.

* Task: "Find all the 'Cars' and 'Pedestrians' in this street view and draw a box around each one."

**Family 1: Two-Stage Detectors (Region-Based)**

These models first propose "regions of interest" (ROIs) and then run a classifier on each region.

| **Model**                       | **Pros**                                                                                                                                                           | **Cons**                                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| Faster R-CNN (Region-based CNN) | Highest Accuracy (Historically). This is the "gold standard" for precision. It's excellent at detecting small objects and getting the bounding boxes _just right_. | Very Slow. The two-stage process makes it unsuitable for real-time applications (e.g., live video). |

**Family 2: One-Stage Detectors (Single-Shot)**

These models predict the bounding boxes and class labels in a single pass.

| **Model**                           | **Pros**                                                                                                                                             | **Cons**                                                                                                                                                                       |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| YOLO (You Only Look Once)           | Extremely Fast. This is the standard for _real-time_ detection. It's fast enough to run on video feeds for self-driving cars, security, or robotics. | Less Accurate (Historically). Can struggle with very small or very close, overlapping objects compared to Faster R-CNN (though new versions like YOLOv8 are closing this gap). |
| SSD (Single Shot MultiBox Detector) | A good middle ground. Faster than Faster R-CNN and often more accurate than early YOLO versions.                                                     | Less popular now. YOLO's development has been faster, making it the more common choice today.                                                                                  |

***

#### 3. 🎨 Image Segmentation

This is the most detailed CV task. The goal is to classify every single pixel in an image.

* Task 1 (Semantic): "Label every pixel as 'Road,' 'Sky,' 'Building,' or 'Car'."
* Task 2 (Instance): "Label every pixel of 'Car 1' as blue, 'Car 2' as green, and 'Car 3' as red."

| **Model**                              | **Pros**                                                                                                                                                                                                                                                                                      | **Cons**                                                                                                                                                                          |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| U-Net (For Semantic Segmentation)      | The Standard for Segmentation. It has a "U-shaped" encoder-decoder architecture. The "skip connections" from the encoder to the decoder allow it to combine deep feature information ("what") with shallow, high-resolution information ("where"), making it excellent at precise boundaries. | Can't distinguish instances. It will label all "cars" as one color. It doesn't know where one car ends and another begins if they are touching.                                   |
| Mask R-CNN (For Instance Segmentation) | The Standard for Instance Segmentation. It's a Faster R-CNN with an extra branch. It first _detects_ an object (box) and then runs a _mask generator_ inside that box to segment the object.                                                                                                  | Complex and Slow. It's even slower than Faster R-CNN, as it's doing detection _and_ segmentation. The training data is also very expensive to create (needs pixel-perfect masks). |

***

#### 4. ✨ Generative Models

This is a different branch of CV. The goal is not to _understand_ an image, but to create new images.

* Task: "Create a new, photorealistic image of a 'dog playing in a park'."

| **Model**                                             | **Pros**                                                                                                                                                                                                                                             | **Cons**                                                                                                                                                                      |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Autoencoders (AE / VAE)                               | Good at compression. An Autoencoder learns to "compress" an image into a small latent code and then "decompress" it. A Variational Autoencoder (VAE) is generative and has a smooth, well-structured latent space (good for "morphing" faces, etc.). | Blurry Images. VAEs are famously known for producing fuzzy, less-realistic images compared to other methods.                                                                  |
| GANs (Generative Adversarial Networks)                | Produces Sharp, Realistic Images. GANs (like StyleGAN) were the king for years. A "Generator" and "Discriminator" network compete, forcing the generator to produce incredibly realistic (but fake) images.                                          | Extremely Hard to Train. GANs are notoriously "unstable." They can suffer from "mode collapse" (only learning to generate one type of face) and are difficult to tune.        |
| Diffusion Models (e.g., DALL-E 2/3, Stable Diffusion) | State-of-the-Art Quality & Control. This is the newest breakthrough. They work by "de-noising" a random static image into a coherent picture, often guided by text. The quality is astounding, and they are highly controllable.                     | Very Slow Inference. Generating one image can take several seconds or even minutes (compared to a GAN, which is near-instant) because it's a multi-step "de-noising" process. |
