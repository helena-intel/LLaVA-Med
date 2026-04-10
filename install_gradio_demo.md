# LlaVA-Med 1.5 OpenVINO demo instructions 

## Installation

This document provides step-by-step instructions to install the LlaVA-Med 1.5 OpenVINO demo on Windows.

### Prerequisites 

- Install `uv` by opening a PowerShell terminal and running:
  ```
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  Then close and reopen your terminal.
- Install Git from https://git-scm.com/downloads/win Make sure to select the recommended option to use git from the command line (not just from Git Bash)

> NOTE: You do not need to install Python manually. The installer will automatically download and use the correct Python version.

### Install the demo

In a command prompt, in the directory where you want to install the demo, run:

```
curl.exe -O https://raw.githubusercontent.com/helena-intel/LLaVA-Med/refs/heads/helena/uv/install_demo.bat 
```
and then
```
install_demo.bat
```

Running `install_demo.bat` will:

- create a llavamed_demo directory 
- clone the LLaVA-Med repository
- automatically download Python 3.11 (managed by uv) and install all dependencies
- download the model and sample images

> NOTE: if you want to install the demo manually, or modify the installation, you can of course run all the steps in [install_demo.bat](https://github.com/helena-intel/LLaVA-Med/blob/main/install_demo.bat) manually

## Run the demo 

In the command prompt, go to the llavamed_demo directory and run `launch_app.bat`.  
When the app has loaded, this will be shown in the command prompt. Click on the link in the command prompt, or  go to http://localhost:7788 in your browser.

### Option: run manually

To run the gradio app manually, from the llavamed_demo directory run:

```
uv run python app.py 
```

Just like with the .bat method, when the app has loaded, this will be shown in the command prompt. Click on the link in the command prompt, or  go to http://localhost:7788 in your browser.


## Update

The installation script creates an `update.bat` file in the llavamed_demo directory. Running it will pull the latest code, update dependencies, and re-download the model files if they have changed.

## Troubleshooting 

If the installation script doesn't complete, for example because git or uv are missing, the easiest solution is to delete the llavamed_demo directory, start a new terminal, and run install_demo.bat again.

