# LlaVA-Med 1.5 OpenVINO demo instructions 

## Installation

This document provides step-by-step instructions to install the LlaVA-Med 1.5 OpenVINO demo on Windows.

### Prerequisites 

- Install Python 3.10 or Python 3.11 from python.org. Select the option to add Python to the environment variables and Path.
- Install Git from https://git-scm.com/downloads/win Make sure to select the recommended option to use git from the command line (not just from Git Bash)

### Install the demo

In a command prompt (not PowerShell), in the directory where you want to install the demo, run:

```
curl.exe -O https://raw.githubusercontent.com/helena-intel/LLaVA-Med/refs/heads/main/install_demo.bat 
```
and then
```
install_demo.bat
```

Running `install_demo.bat` will:

- create a llavamed_demo directory 
- create a Python virtual environment llava_env 
- install LLaVA-med and all dependencies in this virtual environment
- download the model, gradio app, notebook and sample images, and copy the gradio app and notebook to the main llavamed_demo directory. 
- create an update.bat script in the llavamed_demo directory

> NOTE: if you want to install the demo manually, or modify the installation, you can or course run all the steps in [install_demo.bat](https://github.com/helena-intel/LLaVA-Med/blob/main/install_demo.bat) manually

## Run the demo 

In the command prompt, go to the llavamed_demo directory and run `launch_app.bat`.  
When the app has loaded, this will be shown in the command prompt. Click on the link in the command prompt, or  go to http://localhost:7788 in your browser.

> NOTE: It is not needed to activate the virtual environment manually, this will be done by the launch_app.bat file. 

### Option: run manually

To run the gradio app manually, activate the virtual environment and run `python app.py`

```
llava_env\scripts\activate
python app.py 
```

Just like with the .bat method, when the app has loaded, this will be shown in the command prompt. Click on the link in the command prompt, or  go to http://localhost:7788 in your browser.


## Update

The installation script creates an update.bat file in the llavamed_demo directory. This can be run to update the app/model/code. 
 

## Troubleshooting 

If the installation script doesn’t complete, for example because git or python are missing, the easiest solution is to delete the llavamed_demo directory, start a new terminal, and run install_demo.bat again.  
