# Steam Olympics Designathon 2025 Machine Learning Examples

## Instructions
I have included all necessary code and data to run this project. On your machine, you will need to have python v3.0+ and pip installed. Personally I am running 3.11 - I can guarantee it works on that version. It _should_ work on any version 3.0+, but cannot promise as that has not been tested.

In a terminal (command prompt / iterm / git bash / terminal etc), run `pip install -r requirements.txt` from either the `/neuralNetowrk` directory, or the `/regression` directory (depending on which project you are trying to run) to install all software required for the app. 

I personally would recommend doing these installs in a [virtual environment](https://docs.python.org/3/library/venv.html) for isolation from the rest of your machine (this installs all required software in this same directory to avoid version conflicts, and to make cleanup simple for memory conservation)

To run the project, navigate to the directory containing your desired project, and run `python3 classification.py` or `python3 linearRegression.py`. These are written to run correctly from any directory, so if you are comfortable with navigating a file tree with a terminal you can run them from anywhere in your local.

This is intentionally a very watered down and simple implementation of both a classification model using Tensorflow/Keras, and a LinearRegression model using SciKitLearn. There are links and explainations all though the code to help you understand what is going on, as well as to promote diving further into this content if you find this stuff as interesting as I do!

Hopefully you find this helpful, and can pull value from it. If anything is unclear please don't hesitate to open an issue - I will attempt to keep an eye on this repo for a period of time after the presentation.

Thank you!