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
# step-1 : first you need copy the total code
# step-2 : then open the terminal in that particular file location
# step-3 : then use need to install the following packages with the versions

<p>absl-py                      2.1.0</p>
<p>aioice                       0.9.0</p>
<p>aiortc                       1.8.0</p>
<p>altair                       4.2.2</p>
<p>astunparse                   1.6.3</p>
<p>attrs                        23.2.0</p>
<p>av                           11.0.0</p>
<p>blinker                      1.7.0</p>
<p>cachetools                   5.3.3</p>
<p>certifi                      2024.2.2</p>
<p>cffi                         1.16.0</p>
<p>charset-normalizer           3.3.2</p>
<p>click                        8.1.7</p>
<p>colorama                     0.4.6</p>
<p>contourpy                    1.2.0</p>
<p>cryptography                 42.0.5</p>
<p>cycler                       0.12.1</p>
<p>dnspython                    2.6.1</p>
<p>entrypoints                  0.4</p>
<p>Flask                        2.3.2</p>
<p>flatbuffers                  24.3.7</p>
<p>fonttools                    4.49.0</p>
<p>gast                         0.5.4</p>
<p>gitdb                        4.0.11</p>
<p>GitPython                    3.1.42</p>
<p>google-auth                  2.28.2</p>
<p>google-auth-oauthlib         1.0.0</p>
<p>google-crc32c                1.5.0</p>
<p>google-pasta                 0.2.0</p>
<p>grpcio                       1.62.1</p>
<p>h5py                         3.10.0</p>
<p>idna                         3.6</p>
<p>ifaddr                       0.2.0</p>
<p>imageio                      2.34.0</p>
<p>importlib_metadata           7.0.2</p>
<p>itsdangerous                 2.1.2</p>
<p>jax                          0.4.25</p>
<p>Jinja2                       3.1.3</p>
<p>joblib                       1.3.2</p>
<p>jsonschema                   4.21.1</p>
<p>jsonschema-specifications    2023.12.1</p>
<p>keras                        2.14.0</p>
<p>kiwisolver                   1.4.5</p>
<p>lazy_loader                  0.3</p>
<p>libclang                     16.0.6</p>
<p>Markdown                     3.5.2</p>
<p>markdown-it-py               3.0.0</p>
<p>MarkupSafe                   2.1.5</p>
<p>matplotlib                   3.8.3</p>
<p>mdurl                        0.1.2</p>
<p>mediapipe                    0.10.11</p>
<p>ml-dtypes                    0.2.0</p>
<p>networkx                     3.2.1</p>
<p>numpy                        1.26.4</p>
<p>oauthlib                     3.2.2</p>
<p>opencv-contrib-python        4.9.0.80</p>
<p>opencv-python                4.8.1.78</p>
<p>opt-einsum                   3.3.0</p>
<p>package-name                 0.1</p>
<p>packaging                    24.0</p>
<p>pandas                       1.5.3</p>
<p>pillow                       10.2.0</p>
<p>pip                          22.3.1</p>
<p>protobuf                     3.20.3</p>
<p>pyarrow                      15.0.1</p>
<p>pyasn1                       0.5.1</p>
<p>pyasn1-modules               0.3.0</p>
<p>pycparser                    2.21</p>
<p>pydeck                       0.8.1b0</p>
<p>pyee                         11.1.0</p>
<p>Pygments                     2.17.2</p>
<p>pylibsrtp                    0.10.0</p>
<p>Pympler                      1.0.1</p>
<p>pyOpenSSL                    24.1.0</p>
<p>pyparsing                    3.1.2</p>
<p>python-dateutil              2.9.0.post0</p>
<p>pytz                         2024.1</p>
<p>referencing                  0.33.0</p>
<p>requests                     2.31.0</p>
<p>requests-oauthlib            1.4.0</p>
<p>rich                         13.7.1</p>
<p>rpds-py                      0.18.0</p>
<p>rsa                          4.9</p>
<p>scikit-image                 0.22.0</p>
<p>scikit-learn                 1.4.1.post1</p>
<p>scikit-plot                  0.3.7</p>
<p>scipy                        1.12.0</p>
<p>semver                       3.0.2</p>
<p>setuptools                   65.5.0</p></p>
<p>six                          1.16.0</p>
<p>smmap                        5.0.1</p>
<p>sounddevice                  0.4.6</p>
<p>streamlit                    1.20.0</p>
<p>streamlit-webrtc             0.47.6</p>
<p>tensorboard                  2.14.1</p>
<p>tensorboard-data-server      0.7.2</p>
<p>tensorflow                   2.14.0</p>
<p>tensorflow-estimator         2.14.0</p>
<p>tensorflow-intel             2.14.0</p>
<p>tensorflow-io-gcs-filesystem 0.31.0</p>
<p>termcolor                    2.4.0</p>
<p>threadpoolctl                3.3.0</p>
<p>tifffile                     2024.2.12</p>
<p>toml                         0.10.2</p>
<p>toolz                        0.12.1</p>
<p>tornado                      6.4</p>
<p>tqdm                         4.66.2</p>
<p>typing_extensions            4.10.0</p>
<p>tzdata                       2024.1</p>
<p>tzlocal                      5.2</p>
<p>urllib3                      2.2.1</p>
<p>validators                   0.22.0</p>
<p>watchdog                     4.0.0</p>
<p>Werkzeug                     3.0.1</p>
<p>wheel                        0.43.0</p>
<p>wrapt                        1.14.1</p>
<p>zipp                         3.18.0</p>


the above packages are must showd be installed in your computer before you need to run the code.
# step-4  : then go to the app.py file
# step-5 : and then run the code you will get the interface

![Screenshot 2024-03-16 145926](https://github.com/vasu111222/Cataract-Detection-and-Classification/assets/136715738/93be02dd-5350-4a45-947e-b24a299fe5fb)
 then click the proceed button then next interface will open
 
![Screenshot 2024-03-16 150021](https://github.com/vasu111222/Cataract-Detection-and-Classification/assets/136715738/dd2f8125-903a-440d-ba97-59e2f5a4194f)
![Uploading Screenshot 2024-03-16 150021.png…]()
# step-6 : then we can choose the one image from folder or you can download and give to the model
then the inter face will open like this

![Screenshot 2024-03-16 150154](https://github.com/vasu111222/Cataract-Detection-and-Classification/assets/136715738/324e8300-c7f5-4541-8ff0-1c2a1facaaed)
then select one image and click submit and 
# step-7 : then you will get the interface like this 

![Screenshot 2024-03-16 150213](https://github.com/vasu111222/Cataract-Detection-and-Classification/assets/136715738/166a047b-1931-4bae-a68c-fe0b20f566b3)

click the show result button 
# step-8: here you will get the output


![Screenshot 2024-03-16 150320](https://github.com/vasu111222/Cataract-Detection-and-Classification/assets/136715738/51851c76-6c91-4ea3-b307-3c607012b3c6)
