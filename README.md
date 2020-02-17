# Segmentation_evaluation_metric
This file Contains Evaluation metrics, which mostly used for segmentation

Soft Dice Coefficient: dice_coeff(y_true, y_pred)\\
Hard Dice Coefficient: dice_coeff1(y_true, y_pred)
Mean Iou: mean_iou(y_true, y_pred)
Precision: precision(y_true, y_pred)
Recall: recall(y_true, y_pred)
Specificity: specificity(y_true, y_pred)
HausdorffDist: HausdorffDist(y_true, y_pred,Label)
Modeified HausdorffDist: ModifiedHausdorffDist(y_true, y_pred,Label)
Average Symmetric Surface Distance : ASSD(y_true, y_pred,Label)
Soft Dice Coefficient using numpy: dice_coeff_numpy(y_true, y_pred)
Soft Dice Coefficient using numpy: dice_coeff_numpy(y_true, y_pred)
Dice Loss : dice_loss(y_true, y_pred)
Binay Crossentropy Dice Loss : bce_dice_loss(y_true, y_pred)
# Package Required
Tensorflow
cv2
matplotlib
numpy
