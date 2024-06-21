Instruction of programming assignments for CSCE421: Machine Learning


Environment Building
--------------------
Please install python packages tqdm using:
"pip install tqdm" or
"conda install tqdm" (for Anaconda environment)


Installation of PyTorch
--------------------------
If you are using Anaconda environment, simply using:
"conda install -c pytorch pytorch"

For other environments, using:
"pip install torch"

You can refer to https://pytorch.org/get-started/locally/ for more details.

Note: by default, you are installing the CPU version of PyTorch. You are not required 
to install the GPU version.


Dataset Descriptions
--------------------
We will use the Cifar-10 Dataset to do the image classification. The 
CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
with 6000 images per class. There are 50000 training images and 10000 
test images.

Download the CIFAR-10 python version from the website:
https://www.cs.toronto.edu/~kriz/cifar.html
then extract the files.

Follow the instructions in "Dataset layout" to load the data and labels.
The training data and labels are save in 5 files. You will need to integrate
the 5 parts. The image data in each file is a 10000x3072 numpy array. Each 
row of the array stores a 32x32 colour image. The first 1024 entries contain 
the red channel values, the next 1024 the green, and the final 1024 the blue.


Assignment Descriptions
-----------------------
There are total three Python files including 'main.py', 'solution.py' and 
'helper.py'. In this assignment, you only need to add your solution in 
'solution.py' file following the given instruction. However, you might need 
to read all the files to fully understand the requirement. 

The 'helper.py' includes all the helper functions for the assignments.The 'main.py' is used to test your solution. 

Notes: Do not change anything in 'main.py' and 'helper,py' files. Only try 
to add your code to 'solution.py' file and keep function names and parameters 
unchanged.  


APIs you will need
------------------
torch.nn.Conv2d
torch.nn.MaxPool2d
torch.nn.Linear
torch.nn.ReLU

For the honor section, you may also need:
torch.nn.BatchNorm2d
torch.nn.BatchNorm1d
torch.nn.Dropout

Refer to https://pytorch.org/docs/stable/nn.html for more details.


Feel free to email Cong Fu for any assistance.
Email address: congfu@tamu.edu.