# gestureRecognition
Program that recognizes a gesture shown by a hand or two hands.
Gestures using one hand: 
- thumbs up
- thumbs down
- fist
- stop
- student answer gesture
- live long
- middle finger
- call me
- rock
- letter 'L'
- finger gun
- peace
- little
- ok

Gestures using two hands:
- heart
- bird
- fireball
- Wakanda salute
- movie frame

It also allows training custom gestures, adding new or creating completely new models with new gestures. To do so:
- In a file keypoint_classifier_label.csv in directory"model/keypoint_classifier/keypoint_classifier_label.csv", if you want to add a gesture to the model with one hand or in a file keypoint_classifier_label2.csv in directory "model2/keypoint_classifier2/keypoint_classifier_label2.csv" if you want to add a gesture to the model with two hands, add a new label
- In file "main.py" add 1 in global variables GESTURES_ONE_HAND, if you want to add a new gesture to the model with one hand; if to the model with two hands - increase GESTURE_TWO_HANDS. Remember to count gestures from zero.
- In file "keypoint_classification.py" add 1 for every added gesture: in global variable NUM_CLASSES_ONE_HAND (model with one hand) or NUM_CLASSES_TWO_HANDS (model with two hands). Remember to count gestures from one.
- Start "main.py" file. Show your new gesture with correct index visible on trackbar, the same as in the label file. If you are satisfied with the gesture and its keypoint are visible press #k on the keyboard. Now your training data is added to "keypoint.csv" file
- If all custom gestures were added, start "keypoint_classification.py" file. Wait for training results. If model accuracy and results of recognizing a gesture using test data is sufficient for you, everything is ready.
- After starting "main.py" file, gestures will be recognized using your data.


