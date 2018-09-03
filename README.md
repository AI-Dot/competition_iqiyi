# competition_iqiyi
## ref
- [mtcnn](https://github.com/pangyupo/mxnet_mtcnn_face_detection.git)
- [insightface](https://github.com/deepinsight/insightface)
- [InsightFace_TF](https://github.com/auroua/InsightFace_TF)

## Experiment Report


|  method   |   train loss | TRAIN mAP | VAL mAP | TRAIN Precision | VAL Precision | Test mAP | 
| --------  | -----------: |  ----:    | ------: | --------------: | ---------:    | -------: | 
| mtcnn + insightface + means knn (baseline) | none | none | 0.771192 | none | 0.82217 | none | 
| mtcnn + insightface + our FC + means knn | 2.01 | none | 0.365657 | none | 0.4204 | none| 


## InsightFace+FC Usage:
### Train steps:
1. Git clone the InsightFace_TF reposity.
2. Train the model by train_fc_only.py script in InsightFace repo.
3. Evaluate the validate set by eval_map_fc_only.py script.
