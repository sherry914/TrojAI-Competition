from os import listdir
from os.path import join
import json
import logging
import warnings
import shutil
import jsonschema

from detector import Detector
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")


def test_auc(args):
    """Method to test the accuracy of trojan detector.
    """

    with open(args.metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(args.schema_filepath) as schema_file:
        schema_json = json.load(schema_file)

    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)

    # Create the detector instance and loads the metaparameters.
    detector = Detector(args.metaparameters_filepath, args.learned_parameters_dirpath)

    logging.info("Calling the trojan detector")
    
    # List all available model and limit to the number provided
    model_path_list = sorted(
        [
            join(args.testing_dataset_dirpath, 'models', model)
            for model in listdir(join(args.testing_dataset_dirpath, 'models'))
        ]
    )
    logging.info(f"Loading %d models...", len(model_path_list))

    result_list = []
    ground_truth_list = []
    wrong_pred_list = []
    for model in model_path_list:
        detector.infer(join(model, "model.pt"), args.result_filepath, args.scratch_dirpath, "./model/id-00000001/clean-example-data", "NULL")
        with open(args.result_filepath, "r") as f1:
            result = float(f1.read())
            result_list.append(result)
        with open(join(model, "ground_truth.csv"), "r") as f2:
            truth = float(f2.read())
            ground_truth_list.append(truth)
            print("Trojan ground truth:", truth)
        if abs(truth - result) > 0.5:
            wrong_pred_list.append(model)
    auc_score = roc_auc_score(ground_truth_list, result_list)
    print("AUC:", auc_score)

    # 将概率值处理为二分类的预测标签
    predicted_labels = [1 if prob > 0.5 else 0 for prob in result_list]
    # 使用 accuracy_score 函数计算准确率
    accuracy = accuracy_score(ground_truth_list, predicted_labels)

    print("Acc:", accuracy)
    
    '''
    # 把预测结果差的model拷贝到generated_training_set中，重新训练打分员
    destination_folder = "./generated_training_dataset/models/id_"
    destination_model_id = 0
    for model in wrong_pred_list:
        shutil.copytree(model, destination_folder+str(destination_model_id))
        destination_model_id += 1
    '''
    

if __name__ == "__main__":
    from argparse import ArgumentParser

    temp_parser = ArgumentParser(add_help=False)

    parser = ArgumentParser(
        description="Template Trojan Detector to Demonstrate Test and Evaluation. Should be customized to work with target round in TrojAI."
        "Infrastructure."
    )

    parser.set_defaults(func=lambda args: parser.print_help())

    subparser = parser.add_subparsers(dest='cmd', required=True)

    inf_parser = subparser.add_parser('infer', help='Execute container in inference mode for TrojAI detection.')

    inf_parser.add_argument(
        "--testing_dataset_dirpath",
        type=str,
        help="Dictory path to the pytorch models to be evaluated.",
        required=True
    )
    inf_parser.add_argument(
        "--result_filepath",
        type=str,
        help="File path to the file where output result should be written. After "
        "execution this file should contain a single line with a single floating "
        "point trojan probability.",
        required=True
    )
    inf_parser.add_argument(
        "--scratch_dirpath",
        type=str,
        help="File path to the folder where scratch disk space exists. This folder will "
        "be empty at execution start and will be deleted at completion of "
        "execution.",
        required=True
    )
    inf_parser.add_argument(
        "--metaparameters_filepath",
        help="Path to JSON file containing values of tunable paramaters to be used "
        "when evaluating models.",
        type=str,
        required=True,
    )
    inf_parser.add_argument(
        "--schema_filepath",
        type=str,
        help="Path to a schema file in JSON Schema format against which to validate "
        "the config file.",
        required=True,
    )
    inf_parser.add_argument(
        "--learned_parameters_dirpath",
        type=str,
        help="Path to a directory containing parameter data (model weights, etc.) to "
        "be used when evaluating models.  If --configure_mode is set, these will "
        "instead be overwritten with the newly-configured parameters.",
        required=True,
    )
    inf_parser.set_defaults(func=test_auc)

    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        )

    args, extras = temp_parser.parse_known_args()

    if '--help' in extras or '-h' in extras:
        args = parser.parse_args()
    # Checks if new mode of operation is being used, or is this legacy
    elif len(extras) > 0 and extras[0] in ['infer', 'configure']:
        args = parser.parse_args()
        args.func(args)

    else:
        # Assumes we have inference mode if the subparser is not used
        args = inf_parser.parse_args()
        args.func(args)
