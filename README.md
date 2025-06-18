# ML-AI_Pneumonia_Prediction
This is an in-progress project focused on classifying chest Xray images, between normal, viral pneumonia, and bacterial pneumonia
The project will focus on applying various types of ML and AI algorithms & solutions to the dataset, and comparing the overall results

The current algorithms are listed below:

Gaussian Naive Bayes: After preprocessing the image via resizing and improving contrast via cv2's CLAHE, the model has an overall accuracy of 78%, with specific scores for precision, recall, and the f1-score listed below:

                        precision   recall  f1-score   support

           Normal          0.80      0.73      0.76       234
           Viral           0.65      0.80      0.72       148
           Bacterial       0.86      0.81      0.84       242

While it seems to disambiguate between normal Xrays and general pneumonia Xrays alright, the viral pneumonia Xray classification has a precision score of only 65% and a recall of 80%. This indicates that some normal Xrays and bacterial ones are being misattributed as viral, showing that while the model can figure out if an Xray generally has pneumonia or not, it has trouble distinguishing viral pneumonia properly
