# cfMethyl-Seq codes

## Overview

This code is the implementation of two tasks: (1) multi-feature ensemble learning method, which is a component of an integrated cancer detection system; (2) read-based marker discovery.

For the first task, to maximize the diagnostic power of those diverse types of features, we developed a multi-view, stacked model framework with two layers: Level-1 consists of several independent predictive models, each using only features of one type (i.e., one “view” of the samples). Level-2 “stacks” the predictions of the level-1 models as inputs to make a final prediction (Fig. 2b). This is a form of ensemble learning, an effective strategy to ward against overfitting and improve the regularization of the overall model. More important, even if some level-1 models have weak predictive power, they can still contribute to the accuracy of the level-2 prediction by providing complementary information.

For the second task, the read-based marker discovery framework uses alpha-value, defined as the percent of methylated CpGs out of all CpG sites in a sequencing read. The alpha-value describes the pervasiveness of the methylation in a read. The alpha-values of a sample’s all reads in a genomic region form an alpha-value distributions. Since alpha-value distributions often do not follow any known statistical distributions, we develop a non-parametric method to compare two alpha-value distributions. Taking hypermethylation marker discovery as an example, we introduce an alpha-value threshold, i.e., alpha_hyper, to define those with alpha-values >= alpha_hyper as hypermethylated reads. Given an alpha_hyper, if the fold change of hypermethylated read counts between a tumor and its adjacent normal tissue is significant, then this genomic region carries significant tumor signals. The more pairs of tumor and adjacent normal tissues demonstrate such tumor signals, the more stable is the marker. In addition to identifying tumor signals in tissues, we also need to control the background noise in blood, i.e. in the reference non-cancer cfDNA samples. Take the hypermethylation marker discovery as an example, we need to ensure most reads in most reference non-cancer cfDNA samples are strongly hypomethylated. That is, alpha-values of most reads are less than a threshold alpha_hypo.

## Prerequisite Packages

multi.feature.ensemble.models was developed on UCLA Hoffman2 cluster. All the analyses in this study were done there. The system information of UCLA Hoffman2 cluster is:
Linux n6015 2.6.32-754.14.2.el6.x86_64 #1 SMP Tue May 14 19:35:42 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
There is no non-standard hardware required.

1) python3 version 3.7.3+
2) numpy 1.18.4
3) scipy 1.1.0
4) sklearn 0.23.0

## Installation

multi.feature.ensemble.models algorithm is implemented using python scripts. There is no specific installation needed, if all required packages and tools listed in PREREQUISITE PACKAGES are successfully installed. All the source codes are saved under src folder.

Although the python code was developed in the environment of UCLA Hoffman2 cluster, multi.feature.ensemble.models algorithm can be directly run in other environment.

## Usage

### Inputs
1) configuration summary file, which lists all input parameters of the two-level multi.feature.ensemble.models
   a) all input filenames
   b) all output filenames
   c) all input information, such as number of data views (i.e., feature types), class name of control samples, classifier names of Level-1 and Level-2 in the stacked model, etc.
2) input file of class labels of all training samples
3) input file of class labels of all testing samples
4) input data files of each view for all training samples. Supposing we have 3 feature types (i.e., 3 views), then we have three files in this category.
5) input data files of each view for all testing samples. Supposing we have 3 feature types (i.e., 3 views), then we have three files in this category.

### Outputs
1) prediction file for cancer detection
2) prediction file for Tissue-Of-Origin (TOO)

Please use the following steps to run this code:

Step 1). Manually edit the input data configuration file to fill in all information needed. An example is shown in the folder ./demo/example.input_data.config

Step 2). Since all input information has been put into the input data configuration file, The following commands can be run:

export PYTHONPATH=./src/:$PYTHONPATH
echo "======================"
echo "Perform cancer detection using input data"
echo "======================"
python src/cancer_detection.py <input_data_configuration_file>
echo ""

echo "======================"
echo "Perform TOO prediction using input data"
echo "======================"
python src/TOO.py <input_data_configuration_file>
echo ""

In the above shell scripts, you may modify the path of python scripts files (src/cancer_detection.py and src/TOO.py) accordingly.

#### Input File Format

+++ input data summary configuration file
This file includes all detailed information about the multi-feature ensemble predictive model, for both cancer detection and TOO prediction. Users need to edit this file to put into all detailed information that are needed for running multi-feature ensemble predictive model. This file consists of the following sections, consisting of not only different feature types (so called "views" in this software package) but also specifies which classifer is used for each feature type at Level 1 and ensemble classifier in Level 2. Below is the structure of this file, where "?" is where users need to fill in and all other text cannot be modified. We explain each section of the file one by one:

(Explain Section 1): provide information of feature type (data view) numbers.
[number_data_views]
number_data_views = ? 

(Explain Section 2): provide class name of control samples, which appears as a class label in the input class labels files of training and testing samples.
[class_name_of_controls]
class_name = ?

(Explain Section 3): for training data, provide file names of each feature type's (view's) input data matrix where rows are samples and columns are values of markers within the same feature type or view.
[filename_of_data_views_for_training]
view1 = ?
view2 = ?

(Explain Section 4): for testing data, provide file names of each feature type's (view's) input data matrix where rows are samples and columns are values of markers within the same feature type or view.
[filename_of_data_views_for_testing]
view1 = ?
view2 = ?

(Explain Section 5): for cancer detection, provide classifier name of Level-1 in the multi-feature ensemble model. In this code, we implemented two classifiers that are used in the manuscript: Linear-kernel Support Vector Machine with the L2 penalty function and the C parameter set to C=0.5. It is denoted as a classifier name "LinearSVC_l2_c0.5". Another classifier used in the manuscript is random forest with 2000 trees and mtry=\sqrt{#features}, which is denoted as "RF_ntree2000_mtry5"
[level1_classifiers_of_data_views_for_cancer_detection]
view1 = ?
view2 = ?

(Explain Section 6): for Tissue-Of-Origin (TOO) prediction, provide classifier name of Level-1 in the multi-feature ensemble model. In this code, we implemented two classifiers that are used in the manuscript: Linear-kernel Support Vector Machine based one-vs-rest multi-class classifier with the L2 penalty function and the C parameter C=0.5. It is denoted as a classifier name "ovr_LinearSVC_l2_c0.5". Another classifier used in the manuscript is one-vs-rest multi-class random forest with 2000 trees and mtry=\sqrt{#features}, which is denoted as "ovr_RF_ntree2000_mtry5"
[level1_classifiers_of_data_views_for_TOO]
view1 = ?
view2 = ?

(Explain Section 7): provide the number of folds that are used to split all training data into K folds, then produce the Out-Of-Fold Prediction (OOFP) scores in the training data. These OOFP scores are used as the input data to train Level-2 ensemble model. Detailed explanation of OOFP refers to Method section in the manuscript and Supplementary Figure S3 in the Supplementary Information.
[level1_num_folds_of_training_data_for_OOFP]
num_folds = ?

(Explain Section 8): for cancer detection, provide classifier name of Level-2 in the multi-feature ensemble model. The naming rule of the classifier has been explained in Section 5 above.
[level2_classifier_for_cancer_detection]
classifier = ?

(Explain Section 9): for Tissue-Of-Origin (TOO) prediction, provide classifier name of Level-2 in the multi-feature ensemble model. The naming rule of the classifier has been explained in Section 6 above.
[level2_classifier_for_TOO]
classifier = ?

(Explain Section 10): provide the output file of prediction scores and results, for cancer detection and TOO prediction.
[output_prediction_scores]
file_of_cancer_detection_prediction = ?
file_of_TOO_prediction = ?

An example file see "demo/example.input_data.config"

+++ input file of class labels of all training/testing samples
Each line corresponds to a sample, representing the class label of this sample. Supposing there are 100 samples, this file has 100 lines. For example

class1
class1
...
class2
...

An example file sees "demo/input/example.class_labels_of_samples.train.txt" or "demo/input/example.class_labels_of_samples.test.txt"

+++ input data files of each view for all training/testing samples
Each line corresponds to a sample, representing the feature type's (or view's) profile of this sample. Each column represents the value of a marker in the feature type.  Supposing there are 100 samples and 3 markers for this feature type, this file has 100 lines and 3 columns. Columns are separated by a TAB. For example,

0.7687092209875869	0.7792811175078653	0.8668172263868889
0.7908567124816908	0.8607983439478522	0.7366380642451803
0.7621209016199888	0.7938244548298914	0.794897893932299
...

Note that the sample represented by each line in this data matrix file, must be exactly the same as the sample represented by each line in the above input file of class labels of samples.

An example file sees "demo/input/example.data_view_3.train.txt" or "demo/input/example.data_view_3.test.txt"

#### Output file format

+++ output file of cancer detection prediction for testing samples
This file has a header line to self-explain the file format. It has two columns: sample_index(1-based) and prediction_score. The higher the prediction score is, the larger chance the sample gets cancer.

sample_index(1-based)	prediction_score(higher_scores_indicate_higher_cancer_risk, class1(control) vs class2+class3+class4(cancer))
1	0.00000
2	0.00000
...
21	0.76000
22	0.76000
...
79	1.00000
80	1.00000
...

An example file sees "demo/output/example.test.cancer_detection.preds.txt.reference"

+++ output file of TOO prediction for testing samples
This file has a header line to self-explain the file format. Supposing 'class1' is control sample's class label and 'class2/class3/class4' are class labels of all three cancer types, then this file consists 5 columns:

Column 1: sample_index (1-based)
Column 2: predicted_classname (this sample is predicted as the class with the highest prediction score)
Column 3: prediction score of the first cancer type
Column 4: prediction score of the second cancer type
Column 5: prediction score of the third cancer type
...

An example is below:

sample_index(1-based)	predicted_classname	prediction_score_of_class2	prediction_score_of_class3	prediction_score_of_class4
21	class2	1.00000	0.00000	0.00000
22	class2	0.86799	0.02532	0.10669
...
41	class3	0.02554	0.97446	0.00000
42	class3	0.06191	0.93809	0.00000
...
75	class4	0.12339	0.00000	0.87661
76	class4	0.00000	0.00605	0.99395
...

An example file sees "demo/output/example.test.TOO.preds.txt.reference"


## Demo: Example Data

A demo is provided along with the scripts. Example data and required reference files are saved under demo folder.

This demo can be run with simple command lines. This demo was expected to run about 1 minute. 
Please run the following commands:

### run multi.feature.ensemble.models
cd demo
./run_demo.sh

In this demo, two output prediction files are generated:

1) output/example.test.cancer_detection.preds.txt
2) output/example.test.TOO.preds.txt

Please compare each output file with its reference file, which is already provided in the package:

1) output/example.test.cancer_detection.preds.txt.reference
2) output/example.test.TOO.preds.txt.reference

Although Level-2 classifier "random forest" produces different prediction scores (float numbers) every running due to its intrinsically random feature, you can find that these prediction scores in two times of running are still very similar to each other and the prediction results are not changed. For example, the follow two pieces are from output files of two running for TOO predictions:

== from one running: 'demo/output/example.test.TOO.preds.txt.reference'
21	class2	1.00000	0.00000	0.00000
22	class2	0.86799	0.02532	0.10669

== from another running: './demo/output/example.test.TOO.preds.txt'
21	class2	0.99596	0.00404	0.00000
22	class2	0.86195	0.02655	0.11150

