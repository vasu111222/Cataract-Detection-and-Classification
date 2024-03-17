# Cataract-Detection-and-Classification

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/48744487/119628159-77a2e480-be2a-11eb-8557-eb8186d6fe04.png">
</p>


# DESCRIPTION
According to the World Health Organization report, one of the world's leading causes of blindness is reported to be due to cataracts. Even though cataract majorly affects the elderly population however now they can be seen among minors too. Among the various types, the prominently three types of cataract affect masses in high numbers which are nuclear, cortical, and post-subcapsular cataract. Conventional methods of cataract diagnoses include slit lamp image tests by doctors which do not prove to be effective in classifying cataracts in the early stages and can also have inaccuracies in identifying the correct type of cataract. Existing work to automate the process has worked on classification based upon binary detection only or has considered only one type of cataract among the mentioned types for further expanding the system.

# Team members details
  
<p>VAROLLA VASU – 9921004753@klu.ac.in</p>
<p>PUVVADA SURYA SAI GOWTHAM-9822003003@klu.ac.in</p>
<p>VANDRASI SNEHITHA-9921004989@klu.ac.in</p>
<p>JINKA PENCHALAIAH-9822003004@klu.ac.in</p>

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
# challenges you ran into
1)Image Quality: Obtaining high-quality images of the eye can be challenging, especially if the patient has difficulty keeping their eye still during imaging. Poor image quality can make it harder to accurately detect cataracts.

2)Variability in Cataract Types: Cataracts can manifest in various forms, including different degrees of opacity and location within the eye. Detecting and characterizing these different types of cataracts accurately can be challenging.


3)Complexity of Eye Anatomy: The eye is a complex organ with various structures, and cataracts can affect different parts of the eye lens. Distinguishing cataracts from other ocular conditions or anomalies requires a deep understanding of eye anatomy.
4)Generalization: Ensuring that the detection algorithm performs well on images from different populations, demographics, and imaging devices is crucial for its clinical utility. Achieving robustness and generalization across diverse datasets can be challenging.


you can refer from my youtube video the link is given below 
-https://youtu.be/xlipDGxOrQI?feature=shared

# Procedure to run the code 
step-1 : first you need copy the total code
step-2 : then open the terminal in that particular file location
step-3 : then use need to install the following packages with the versions
absl-py                      2.1.0
aioice                       0.9.0
aiortc                       1.8.0
altair                       4.2.2
astunparse                   1.6.3
attrs                        23.2.0
av                           11.0.0
blinker                      1.7.0
cachetools                   5.3.3
certifi                      2024.2.2
cffi                         1.16.0
charset-normalizer           3.3.2
click                        8.1.7
colorama                     0.4.6
contourpy                    1.2.0
cryptography                 42.0.5
cycler                       0.12.1
dnspython                    2.6.1
entrypoints                  0.4
Flask                        2.3.2
flatbuffers                  24.3.7
fonttools                    4.49.0
gast                         0.5.4
gitdb                        4.0.11
GitPython                    3.1.42
google-auth                  2.28.2
google-auth-oauthlib         1.0.0
google-crc32c                1.5.0
google-pasta                 0.2.0
grpcio                       1.62.1
h5py                         3.10.0
idna                         3.6
ifaddr                       0.2.0
imageio                      2.34.0
importlib_metadata           7.0.2
itsdangerous                 2.1.2
jax                          0.4.25
Jinja2                       3.1.3
joblib                       1.3.2
jsonschema                   4.21.1
jsonschema-specifications    2023.12.1
keras                        2.14.0
kiwisolver                   1.4.5
lazy_loader                  0.3
libclang                     16.0.6
Markdown                     3.5.2
markdown-it-py               3.0.0
MarkupSafe                   2.1.5
matplotlib                   3.8.3
mdurl                        0.1.2
mediapipe                    0.10.11
ml-dtypes                    0.2.0
networkx                     3.2.1
numpy                        1.26.4
oauthlib                     3.2.2
opencv-contrib-python        4.9.0.80
opencv-python                4.8.1.78
opt-einsum                   3.3.0
package-name                 0.1
packaging                    24.0
pandas                       1.5.3
pillow                       10.2.0
pip                          22.3.1
protobuf                     3.20.3
pyarrow                      15.0.1
pyasn1                       0.5.1
pyasn1-modules               0.3.0
pycparser                    2.21
pydeck                       0.8.1b0
pyee                         11.1.0
Pygments                     2.17.2
pylibsrtp                    0.10.0
Pympler                      1.0.1
pyOpenSSL                    24.1.0
pyparsing                    3.1.2
python-dateutil              2.9.0.post0
pytz                         2024.1
referencing                  0.33.0
requests                     2.31.0
requests-oauthlib            1.4.0
rich                         13.7.1
rpds-py                      0.18.0
rsa                          4.9
scikit-image                 0.22.0
scikit-learn                 1.4.1.post1
scikit-plot                  0.3.7
scipy                        1.12.0
semver                       3.0.2
setuptools                   65.5.0
six                          1.16.0
smmap                        5.0.1
sounddevice                  0.4.6
streamlit                    1.20.0
streamlit-webrtc             0.47.6
tensorboard                  2.14.1
tensorboard-data-server      0.7.2
tensorflow                   2.14.0
tensorflow-estimator         2.14.0
tensorflow-intel             2.14.0
tensorflow-io-gcs-filesystem 0.31.0
termcolor                    2.4.0
threadpoolctl                3.3.0
tifffile                     2024.2.12
toml                         0.10.2
toolz                        0.12.1
tornado                      6.4
tqdm                         4.66.2
typing_extensions            4.10.0
tzdata                       2024.1
tzlocal                      5.2
urllib3                      2.2.1
validators                   0.22.0
watchdog                     4.0.0
Werkzeug                     3.0.1
wheel                        0.43.0
wrapt                        1.14.1
zipp                         3.18.0

the above packages are must showd be installed in your computer before you need to run the code.
step-4  : then go to the app.py file
step-5 : and then run the code you will get the interface

![Screenshot 2024-03-16 145926](https://github.com/vasu111222/Cataract-Detection-and-Classification/assets/136715738/93be02dd-5350-4a45-947e-b24a299fe5fb)


