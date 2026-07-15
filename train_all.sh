#!/bin/bash

MAX_ROUND=10
EPOCH=60 #60 
PRE_EPOCH=15 #15  
LR=1e-4
train_T=10

declare -A dataset_mapping=(
    [MD_GarSti]="MD_GarSti MD_GarSma MD_DriSti MD_DriSma" # MD_GarSma MD_DriSti MD_DriSma
    [VIPL]="VIPL"
    [Tokyo]="Tokyo"
    [MI]="MI"
    [Tokyo,MD]="MI"
    [MD,MI]="Tokyo"
    [Tokyo,MI]="MD"
)

for round in $(seq 1 $MAX_ROUND)    #"" # NIR RGB FUSE RGB #
do
    echo Round \#"$round"
    for modality in UFUSE 
    do
        echo "Running for modality: $modality"
        for train_dataset in MD_GarSti MI #MD_GarSti #MI   #MD_GarSti
        do
            bs=2
            echo python3 train_label.py --train_dataset=$train_dataset --bs $bs --epoch $EPOCH --modality $modality --model_S 4 --lr $LR --train_T $train_T --pretrain_epoch $PRE_EPOCH --do_preload
            python3 train_label.py --train_dataset=$train_dataset --bs $bs --epoch $EPOCH --modality $modality --model_S 4 --lr $LR --train_T $train_T --pretrain_epoch $PRE_EPOCH --do_preload

            test_datasets=${dataset_mapping[$train_dataset]}
            echo "test_datasets for $train_dataset: $test_datasets"
            for test_dataset in $test_datasets #MD_GarSma #MI #MD MD_DriSti # MD_DriSti
            do  
                """if [ $test_dataset != $train_dataset ]
                then
                    continue
                fi"""
                bs=1
                echo python3 test_label.py --train_dataset="$train_dataset" --test_dataset $test_dataset --bs $bs --epoch $EPOCH --modality $modality --train_T $train_T --test_T 10 --model_S 4 --pretrain_epoch $PRE_EPOCH --do_preload 
                python3 test_label.py --train_dataset="$train_dataset" --test_dataset $test_dataset --bs $bs --epoch $EPOCH --modality $modality --train_T $train_T --test_T 10 --model_S 4 --pretrain_epoch $PRE_EPOCH --do_preload 

            done
        done
    done
done
'''
#PRE_EPOCH=15
#train_T=10
# cross dataset testing
for round in $(seq 1 $MAX_ROUND)    #"" # NIR RGB FUSE RGB #
do
    echo Round \#"$round"
    for modality in UFUSE #UFUSE SFUSE # # #RGB NIR  #  FUSE #NIR RGB #NIR #RGB   #FUSE #FUSE #RGB #NIR   #FUSE
    do
        echo "Running for modality: $modality"
        for train_dataset in Tokyo,MI   # #MD,MI #Tokyo,MI  #Tokyo,MD MD,MI
        do
            bs=4
            echo python3 train_label.py --train_dataset=$train_dataset --bs $bs --epoch $EPOCH --modality $modality --model_S 4 --lr $LR --train_T --train_T $train_T --pretrain_epoch $PRE_EPOCH --do_preload --is_cross
            python3 train_label.py --train_dataset=$train_dataset --bs $bs --epoch $EPOCH --modality $modality --model_S 4 --lr $LR --train_T $train_T --pretrain_epoch $PRE_EPOCH --do_preload --is_cross

            test_datasets=${dataset_mapping[$train_dataset]}
            echo "test_datasets for $train_dataset: $test_datasets"
            for test_dataset in $test_datasets #MD_GarSma #MI #MD MD_DriSti # MD_DriSti
            do  
                """if [ $test_dataset != $train_dataset ]
                then
                    continue
                fi"""
                bs=1
                echo python3 test_label.py --train_dataset="$train_dataset" --test_dataset $test_dataset --bs $bs --epoch $EPOCH --modality $modality --train_T $train_T --test_T 10 --model_S 4 --pretrain_epoch $PRE_EPOCH --do_preload --is_cross
                python3 test_label.py --train_dataset="$train_dataset" --test_dataset $test_dataset --bs $bs --epoch $EPOCH --modality $modality --train_T $train_T --test_T 10 --model_S 4 --pretrain_epoch $PRE_EPOCH --do_preload --is_cross

            done
        done
    done
done

'''
