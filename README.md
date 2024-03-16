# Cataract-Detection-and-Classification

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/48744487/119628159-77a2e480-be2a-11eb-8557-eb8186d6fe04.png">
</p>


# DESCRIPTION
According to the World Health Organization report, one of the world's leading causes of blindness is reported to be due to cataracts. Even though cataract majorly affects the elderly population however now they can be seen among minors too. Among the various types, the prominently three types of cataract affect masses in high numbers which are nuclear, cortical, and post-subcapsular cataract. Conventional methods of cataract diagnoses include slit lamp image tests by doctors which do not prove to be effective in classifying cataracts in the early stages and can also have inaccuracies in identifying the correct type of cataract. Existing work to automate the process has worked on classification based upon binary detection only or has considered only one type of cataract among the mentioned types for further expanding the system.

# Team members details
  
<p>VAROLLA VASU – 9921004753</p>
<p>PUVVADA SURYA SAI GOWTHAM 9822003003</p>
<p>VANDRASI SNEHITHA 9921004989</p>
<p>JINKA PENCHALAIAH 9822003004</p>

# The problem it solves
Our system works on the detection of cataracts and type of classification on the basis of severity namely; mild, normal, and severe, in an attempt to reduce errors of manual detection of cataracts in the early ages.

The phase 1 implementation has successfully classified images as cataract affected or as a normal eye with an accuracy of 96% using combined feature vectors from the SIFT-GLCM algorithm applied to classifier models of SVM, Random Forest, and Logistic Regression. The effect of using SIFT and GLCM separately has also been studied which leads to comparatively lesser accuracies in the model trained. 

The phase 2 implementation which deals with the type classification, has obtained the maximum validation acurracy of 97.66% using deep convolutional neural network models, in particular SqueezeNet, MobileNet, and VGG16.

The results have been made accessible using web and Flask based user interface.

The phase 1 implementation of the project which works on binary classification of cataract has been compiled into a conference paper and accepted in the “International Conference on Artificial Intelligence: Advances and Applications (ICAIAA 2021).”

# use cases
Algorithms used

PHASE 1

1. SIFT 
2. GLCM
3. SVM
4. LOGISTIC REGRESSION
5. RANDOM FOREST
6. KNN

PHASE 2
1. HOUGH CIRCLE TRANSFORM
2. VGG-16
3. MOBILENET V2
4. SQUEEZENET
5. 
# challenges you ran into
1)Image Quality: Obtaining high-quality images of the eye can be challenging, especially if the patient has difficulty keeping their eye still during imaging. Poor image quality can make it harder to accurately detect cataracts.

2)Variability in Cataract Types: Cataracts can manifest in various forms, including different degrees of opacity and location within the eye. Detecting and characterizing these different types of cataracts accurately can be challenging.

3)Complexity of Eye Anatomy: The eye is a complex organ with various structures, and cataracts can affect different parts of the eye lens. Distinguishing cataracts from other ocular conditions or anomalies requires a deep understanding of eye anatomy.
4)Generalization: Ensuring that the detection algorithm performs well on images from different populations, demographics, and imaging devices is crucial for its clinical utility. Achieving robustness and generalization across diverse datasets can be challenging.

