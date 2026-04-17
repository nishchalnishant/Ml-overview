# Computer Vision

For vision interviews, strong answers explain the task definition, the typical pipeline, and the constraint that usually breaks naive solutions: data quality, invariance, latency, or label cost.

---

# Q1: What is computer vision, and why is it important?

**Interview-ready answer**

Computer vision is the field of teaching machines to extract meaning from visual data such as images and video. It matters because so much real-world information is visual: documents, medical scans, traffic scenes, retail shelves, industrial inspection, and user-generated media. A strong answer should also mention that vision is not only classification; it includes localization, segmentation, tracking, retrieval, and scene understanding.

---

# Q2: What is image segmentation, and what are its applications?

**Interview-ready answer**

Image segmentation assigns a label to each pixel or region in an image, so instead of saying only what is in the image, it says where it is. Semantic segmentation labels each pixel by class, instance segmentation separates individual objects, and panoptic segmentation combines both. Applications include medical imaging, autonomous driving, satellite analysis, manufacturing inspection, and background removal.

---

# Q3: What is object detection, and how does it differ from image classification?

**Interview-ready answer**

Image classification predicts what is present in the image as a whole, while object detection predicts both what objects are present and where they are, usually with bounding boxes and confidence scores. Detection is harder because it must solve localization and classification together, often for multiple objects of different scales in the same image.

---

# Q4: What are the steps to build an image recognition system?

**Interview-ready answer**

I would start by defining the task clearly: classification, detection, segmentation, retrieval, or OCR. Then I would collect and audit data, build labeling guidelines, choose a baseline model, define evaluation metrics and slices, and design preprocessing and augmentation. After training, I would analyze failure cases, optimize for deployment constraints such as latency or memory, and set up monitoring for drift, camera changes, and data quality. The strongest answer here is process-oriented, not just model-oriented.

---

# Q5: What are the challenges in real-time object tracking?

**Interview-ready answer**

Real-time tracking must maintain object identity across frames under tight latency constraints. The hard parts are occlusion, motion blur, changing viewpoints, similar-looking objects, missed detections, and the tradeoff between detector accuracy and frame rate. A good interview answer should also mention that tracking quality depends heavily on detection quality, so the system is often a detect-then-associate pipeline rather than a standalone tracker.

---

# Q6: What is feature extraction in computer vision?

**Interview-ready answer**

Feature extraction means converting raw pixels into representations that are more useful for a downstream task. Historically this meant hand-crafted descriptors like SIFT, HOG, or color histograms. In modern vision, the features are usually learned by CNNs or vision transformers, where lower layers capture simple patterns like edges and textures and deeper layers capture higher-level object structure.

---

# Q7: What is OCR, and what are its main applications?

**Interview-ready answer**

OCR, or optical character recognition, is the task of converting text in images or scanned documents into machine-readable text. It is used in document processing, ID verification, invoice automation, digitizing archives, license plate reading, and assistive technology. In practice, OCR pipelines often include detection of text regions, correction for layout or rotation, recognition, and post-processing with language models or rules.

---

# Q8: How does CNN differ from traditional neural networks in computer vision?

**Interview-ready answer**

Traditional fully connected networks ignore the spatial structure of images and require huge numbers of parameters when applied directly to pixels. CNNs exploit locality and parameter sharing through convolutional filters, which makes them much more data-efficient and better aligned with image structure. That inductive bias is why CNNs were so successful in vision before transformers became competitive.

---

# Q9: What is data augmentation, and what techniques are commonly used?

**Interview-ready answer**

Data augmentation creates additional training variation by applying label-preserving transformations to images. Common techniques include flipping, cropping, rotation, color jitter, random erasing, mixup, cutmix, and geometric transforms. The key point is that augmentation injects invariances you want the model to learn, but it has to respect the task. For example, horizontal flips may be fine for animals but not always for text or medical images.

---

# Q10: What are some popular deep learning frameworks for computer vision?

**Interview-ready answer**

PyTorch is a dominant research and production framework because of its flexibility and ecosystem. TensorFlow and Keras are also widely used, especially in production workflows and mobile deployment. Around those frameworks, people often rely on specialized libraries such as torchvision, OpenCV, Detectron2, MMDetection, or Hugging Face for model implementations and training utilities. The important interview point is that framework choice is usually secondary to data, architecture, and deployment needs.

---

# Q11: How can Transformers be used for computer vision tasks?

**Interview-ready answer**

Transformers can be applied to vision by treating an image as a sequence of patches and using self-attention to model relationships across the entire image. This is the core idea behind vision transformers. They are powerful because they capture global context well and transfer nicely across tasks, especially at scale. But they typically need large datasets or strong pretraining and can be more computationally demanding than CNNs in some regimes. In practice, modern vision systems often combine convolutional and transformer ideas rather than treating them as mutually exclusive.
