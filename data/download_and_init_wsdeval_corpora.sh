#!/bin/bash

mkdir -p wsdeval_corpora
cd wsdeval_corpora

wget -nc http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip

echo "Extract corpus files..."
unzip -q WSD_Evaluation_Framework.zip

echo "Move corpus files..."
mv WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt ./
mv WSD_Evaluation_Framework/Evaluation_Datasets/semeval2013/semeval2013.data.xml WSD_Evaluation_Framework/Evaluation_Datasets/semeval2013/semeval2013.gold.key.txt ./
mv WSD_Evaluation_Framework/Evaluation_Datasets/semeval2015/semeval2015.data.xml WSD_Evaluation_Framework/Evaluation_Datasets/semeval2015/semeval2015.gold.key.txt ./
mv WSD_Evaluation_Framework/Evaluation_Datasets/senseval2/senseval2.data.xml WSD_Evaluation_Framework/Evaluation_Datasets/senseval2/senseval2.gold.key.txt ./
mv WSD_Evaluation_Framework/Evaluation_Datasets/senseval3/senseval3.data.xml WSD_Evaluation_Framework/Evaluation_Datasets/senseval3/senseval3.gold.key.txt ./
mv WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt ./

echo "Cleanup..."
rm -R WSD_Evaluation_Framework
