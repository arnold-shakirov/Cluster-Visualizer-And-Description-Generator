# Image clustering app with description of photos

## Overview

This application focuses on creating a clustered image, which would give the user insights about similar photos and generating descriptions for these photos. User has to provide folder with the files, which will be analyzed. This application uses Tkinter for GUI, OpenAI's CLIP model, and machine learning clustering algorithms to analyze and cluster images based on their visual and contextual similarities. It displays images in a cluster format on a canvas based on t-SNE dimensionality reduction.

## Features

- **Folder Selection**: Users can select a folder containing images for analysis. They should be with PNG, JPG, JPEG extensions
- **Image Processing**: Click the "Process the Data" button to analyzes images in the selected folder using the pretrained CLIP model to generate descriptions and features. 
- **Progress Bar**: While the photo is being analyzed, you can check the progress by looking at a bar, which shows the current progress. Also, feel free to look at the gif of dancing cat.
- **Clustering and Visualization**: Clusters images based on their features and displays them on a canvas according to their t-SNE reduced dimensions.
- **Save Results**: Click the "Save Results" buton to save the canvas as a PNG file in the folder the user wants.

## Requirements

To run this application, you need the following installed:

- Python 3.7+
- PyTorch
- PIL (Pillow)
- numpy
- sklearn
- open_clip
- glob
- threading
- tkinter

You can install the necessary libraries using pip:

```bash
pip install torch pillow numpy scikit-learn glob2 open_clip tkinter
```

## How to use?
To use the application:

- Start the application.
- Click the "Choose the folder" button to select the folder containing the images.
- Click "Process the data" to start the analysis and clustering of images.
- Once processing is complete, the images will be displayed on the canvas in clusters.
- Use the "Save Results" button to save the canvas as a PNG file.

## Contact me
Any questions? Contact me, my email is: arnold.shakirov@gmail.com

03/05/24, Arnold Shakirov
