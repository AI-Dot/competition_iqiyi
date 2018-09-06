# competition_iqiyi
## Ref
- [mtcnn](https://github.com/pangyupo/mxnet_mtcnn_face_detection.git)
- [insightface](https://github.com/deepinsight/insightface)
- [InsightFace_TF](https://github.com/auroua/InsightFace_TF)

## Dataset Details

### Training Set Statistical Information (in log scale)
![Training Set](https://raw.githubusercontent.com/AI-Dot/competition_iqiyi/master/imgs/train.png)

### Validation Set Statistical Information (in log scale)
![Validation Set](https://raw.githubusercontent.com/AI-Dot/competition_iqiyi/master/imgs/val.png)

## Experiment

### Overview Report
|  method   |  TRAIN mAP | VAL mAP | TRAIN Precision (infer_2) | VAL Precision | Test mAP | 
| --------  |   ----:    | ------: | --------------: | ---------:    | -------: | 
| mtcnn + insightface + means knn (baseline) | none | 0.771192 | none | 0.82217 | none | 
| mtcnn + insightface + our FC + means knn |  none | 0.365657 | 0.89025 | 0.4204 | none| 

### Partial Training Report
|  method   |  TRAIN Loss | TRAIN Precision (infer) | VAL Precision | 
| --------  |   ------:    | -------------: | -------------:|
| our FC (512in_2048_512out) | 2.01 | 0.904674 | 0.365657 | 




### InsightFace+FC Usage:
1. Git clone the InsightFace_TF reposity.
2. Train the model by train_fc_only.py script in InsightFace repo.
3. Evaluate the validate set by eval_map_fc_only.py script.

### Error Report Usage:
1. Run my report.py 
2. Then we will get a report.md
3. export report.pdf by using the largest report.md printing page size.
