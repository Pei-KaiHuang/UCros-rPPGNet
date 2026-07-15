# parameter setting
import argparse


def get_args():
    
    parser = argparse.ArgumentParser()
    
    # ----------------- General ------------------
    parser.add_argument('--train_dataset', default="", type=str,
                        help="""
                        Options => C: COHFACE, P: PURE, U: UBFC, MD: MR-NIRP driving, MI: MR-NIRP indoor, V: VIPL-HR,
                        e.g. --dataset="C"  for intra-training/testing on COHFACE
                             --dataset="UP" for cross-training/testing on PURE and UBFC        
                        """)
    parser.add_argument('--test_dataset', default="", type=str,
                        help="Same as above")

    parser.add_argument('--model_S', default=2, type=int,
                        help="spatial dimension of model")
    parser.add_argument('--conv', default="conv3d", type=str,
                        help="Convolution type for 3DCNN")
        
    parser.add_argument('--bs', default=10, type=int,
                        help="batch size")
    parser.add_argument('--epoch', default=100, type=int,
                        help="training/testing epoch")
    parser.add_argument('--fps', default=30, type=int,
                        help="fps for dataset")
    parser.add_argument('--lr', default=1e-5, type=float,
                        help="learning rate")
    
    parser.add_argument('--high_pass', default=40, type=int)
    parser.add_argument('--low_pass', default=250, type=int)

    parser.add_argument('--modality', type=str, help="modality")
    
    # ----------------- Training -----------------
    parser.add_argument('--train_T', default=6, type=int,
                        help="training clip length(seconds))")
    parser.add_argument('--pretrain_epoch', default=0, type=int,
                        help="pretraining epoch")
    
    # ----------------- Testing -----------------
    parser.add_argument('--test_T', default=10, type=int,
                        help="testing clip length(seconds))")
    #parser.add_argument('--stride', default=300, type=int,
                        #help="testing clip stride")
    
    parser.add_argument('--do_preload', action='store_true')
    parser.add_argument('--do_not_en', action='store_true',help="do not use adapter to enhance modality feature")
    parser.add_argument('--is_cross', action='store_true', help='Enable cross-dataset training')
    
    return parser.parse_args()
