import os
import json
import jsonschema
import torch
from tqdm import tqdm
from trojai_mitigation_round.mitigations.finetuning import FineTuningTrojai
from trojai_mitigation_round.trojai_dataset import Round11SampleDataset
import copy
import numpy as np
from collections import Counter

def demo_training_example_data(base_training_dataset_dirpath):
    training_models_dirpath = os.path.join(base_training_dataset_dirpath, 'models')
    for model_dir in os.listdir(training_models_dirpath):
        training_model_dirpath = os.path.join(training_models_dirpath, model_dir)
        new_clean_data_example_dirpath = os.path.join(training_model_dirpath, 'new-clean-example-data')
        new_poisoned_data_example_dirpath = os.path.join(training_model_dirpath, 'new-poisoned-example-data')

        # Iterate over new poisoned example data, each example contains a filename such as '305.png', with associated ground truth '305.json'
        # Ground truth is stored as a dictionary with 'clean_label' and 'poisoned_label'
        if os.path.exists(new_poisoned_data_example_dirpath):
            for example_file in os.listdir(new_poisoned_data_example_dirpath):
                if example_file.endswith('.png'):
                    example_basename_no_ext = os.path.splitext(example_file)[0]
                    example_image_filepath = os.path.join(new_poisoned_data_example_dirpath, example_file)
                    example_groundtruth_filepath = os.path.join(new_poisoned_data_example_dirpath, '{}.json'.format(example_basename_no_ext))

                    with open(example_groundtruth_filepath, 'r') as fp:
                        example_groundtruth_dict = json.load(example_groundtruth_filepath)
                        clean_label = example_groundtruth_dict['clean_label']
                        poisoned_label = example_groundtruth_filepath['poisoned_label']

        # Iterate over new clean example data, each example contains a filename such as '305.png', with associated ground truth '305.json'
        # Ground truth is stored as a dictionary with 'clean_label
        if os.path.exists(new_clean_data_example_dirpath):
            for example_file in os.listdir(new_clean_data_example_dirpath):
                if example_file.endswith('.png'):
                    example_basename_no_ext = os.path.splitext(example_file)[0]
                    example_image_filepath = os.path.join(new_clean_data_example_dirpath, example_file)
                    example_groundtruth_filepath = os.path.join(new_clean_data_example_dirpath, '{}.json'.format(example_basename_no_ext))

                    with open(example_groundtruth_filepath, 'r') as fp:
                        example_groundtruth_dict = json.load(example_groundtruth_filepath)
                        clean_label = example_groundtruth_dict['clean_label']




def prepare_mitigation(args, config_json):
    """Given the command line args, construct and return a subclass of the TrojaiMitigation class

    :param args: The command line args
    :return: A subclass of TrojaiMitigation that can implement a given mitigtaion technique
    """
    # Get required classes for loss and optimizer dynamically
    loss_class = getattr(torch.nn, config_json['loss_class'])
    optim_class = getattr(torch.optim, config_json['optimizer_class'])

    print(f"Using {loss_class} for ft loss")
    print(f"Using {optim_class} for ft optimizer")

    scratch_dirpath = args.scratch_dirpath
    ckpt_dirpath = os.path.join(scratch_dirpath, config_json['ckpt_dir'])

    if not os.path.exists(ckpt_dirpath):
        os.makedirs(ckpt_dirpath, exist_ok=True)

    # Construct defense with args
    mitigation = FineTuningTrojai(
        loss_cls=loss_class,
        optim_cls=optim_class,
        lr=0.01,
        epochs=10,
        batch_size=10,
        num_workers=1,
        device=args.device,
    )
    return mitigation


def prepare_model(path, device):
    """Prepare and load a model defined at a path

    :param path: the path to a pytorch state dict model that will be loaded
    :param device: Either cpu or cuda to push the device onto
    :return: A pytorch model
    """
    model = torch.load(path)
    model = model.to(device=device)
    return model


def prepare_dataset(dataset_path, split_name, require_label=False):
    dataset = Round11SampleDataset(root=dataset_path, split=split_name, require_label=require_label)
    return dataset


def mitigate_model(model, mitigation, dataset, output_dir, output_name):
    """Given the a torch model and a path to a dataset that may or may not contain clean/poisoned examples, output a mitigated
    model into the output directory.

    :param model: Pytorch model to be mitigated
    :param mitigtaion: The given mitigation technique
    :param dataset: The Pytorch dataset that may/may not contain poisoned examples
    :param output_dir: The directory where the mitigated model's state dict is to be saved to.
    :param output_name: the name of the pytorch model that will be saved
    """
    mitigated_model = mitigation.mitigate_model(model, dataset)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(mitigated_model.model, os.path.join(output_dir, output_name))


def mitigate_model_online(model, mitigation, dataset):
    mitigated_model = mitigation.mitigate_model_online(model, dataset)
    return mitigated_model.model

def test_model(model, mitigation, testset, batch_size, num_workers, device):
    """Tests a given model on a given dataset, using a given mitigation's pre and post processing
    before and after interfence. 

    :param model: Pytorch model to test
    :param mitigation: The mitigation technique we're using
    :param testset_path: The the Pytorch testset that may or may not be poisoned
    :param batch_size: Batch size for the dataloader
    :param num_workers: The number of workers to use for the dataloader
    :param device: cuda or cpu device
    :return: dictionary of the results with the labels and logits
    """
    dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    
    model.eval()
    all_logits = torch.tensor([])
    all_labels = torch.tensor([])
    all_fnames = []
    
    # Label could be None in case the dataset did not require it to load
    for x, y, fname in tqdm(dataloader):
        preprocess_x = x
        output_logits = model(preprocess_x.to(device)).detach().cpu()
        final_logits = output_logits

        all_logits = torch.cat([all_logits, final_logits], axis=0)
        all_labels = torch.cat([all_labels, y], axis=0)
        all_fnames.extend(fname)
    
    fname_to_logits = dict(zip(all_fnames, all_logits.tolist()))

    return fname_to_logits



# def build_label(model, dataset, batch_size, num_workers, device):
#     """
#     Run the model on each sample in 'dataset' (in order) to get predictions,
#     then build a *new dataset* that uses those predictions as labels.
#     """
#     # 1. Prepare a DataLoader that does NOT shuffle
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                             num_workers=num_workers, shuffle=False)

#     all_predictions = []
#     model.eval()

#     with torch.no_grad():
#         for x_batch, y_batch, fname_batch in tqdm(dataloader):
#             x_batch = x_batch.to(device)
#             output_logits = model(x_batch).cpu()
#             # preds = output_logits.argmax(dim=1)

#             top2_values, top2_indices = output_logits.topk(2, dim=1)
#             batch_size = top2_indices.size(0)
#             random_selector = torch.randint(0, 2, (batch_size,))
#             preds = top2_indices[torch.arange(batch_size), random_selector]

#             # Accumulate predictions in a list (maintaining the dataset's ordering)
#             all_predictions.extend(preds.tolist())

#     # 2. Build a new dataset with the updated labels
#     #    We'll define a simple "wrapper" dataset that references the original
#     #    dataset for images/fnames, but overrides the labels with predictions.

#     class LabeledRound11SampleDataset(torch.utils.data.Dataset):
#         def __init__(self, original_dataset, new_labels):
#             assert len(original_dataset) == len(new_labels), \
#                 "Number of new labels must match dataset length"
#             self.original_dataset = original_dataset
#             self.new_labels = new_labels

#         def __len__(self):
#             return len(self.original_dataset)

#         def __getitem__(self, idx):
#             # Get the original image + fname from the old dataset
#             img, old_label, fname = self.original_dataset[idx]
#             # Override old_label with the new predicted label
#             return img, self.new_labels[idx], fname

#     # Instantiate the new dataset with predictions
#     new_dataset = LabeledRound11SampleDataset(dataset, all_predictions)
#     return new_dataset

import torch.nn.functional as F

def build_label(
    model, 
    dataset, 
    batch_size, 
    num_workers, 
    device, 
    keep_ratio=0.1
):

    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio must be in the interval (0, 1].")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=False)
    model.eval()

    all_logits = []
    original_indices = []
    with torch.no_grad():
        current_start = 0
        for x_batch, _, _ in tqdm(dataloader):
            x_batch = x_batch.to(device)
            
            logits = model(x_batch).cpu()
            all_logits.append(logits)

            batch_size_curr = x_batch.size(0)
            batch_indices = range(current_start, current_start + batch_size_curr)
            original_indices.extend(batch_indices)

            current_start += batch_size_curr

    all_logits = torch.cat(all_logits, dim=0)
    N = all_logits.size(0)

    probs = F.softmax(all_logits, dim=1)       
    top3_probs, _ = probs.topk(k=3, dim=1)     
    p1 = top3_probs[:, 0]
    p2 = top3_probs[:, 1]
    p3 = top3_probs[:, 2]
    eps = 1e-12
    score = (p2 / (p1 + eps)) # - (p3 / (p2 + eps))

    final_preds = all_logits.argmax(dim=1)  # shape: (N,)

    from collections import defaultdict
    groups = defaultdict(list)

    for i in range(N):
        c = final_preds[i].item()
        groups[c].append((original_indices[i], score[i].item()))

    class_counts = {c: len(samples) for c, samples in groups.items() if len(samples) > 0}
    if not class_counts:
        return EmptyDataset()

    min_count = min(class_counts.values())
    print(min_count)

    for c in list(groups.keys()):
        if len(groups[c]) == 0:
            del groups[c]
            continue
        
        groups[c].sort(key=lambda x: x[1]) 
        if len(groups[c]) > min_count:
            groups[c] = groups[c][:min_count]

    keep_class_count = int(min_count * keep_ratio)
    keep_class_count = max(keep_class_count, 1) 
    
    for c in list(groups.keys()):
        groups[c] = groups[c][:keep_class_count]

    final_kept_indices = []
    for c, sample_list in groups.items():
        final_kept_indices.extend([t[0] for t in sample_list])

    final_kept_indices_set = set(final_kept_indices)

    idx_to_pred = {}
    idx_to_fname = {}

    for i in final_kept_indices:
        pred_label = final_preds[i].item()
        _, _, fname = dataset[i]
        idx_to_pred[i] = pred_label
        idx_to_fname[i] = fname


    class FilteredLabeledDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, kept_indices, idx_to_pred, idx_to_fname):
            self.original_dataset = original_dataset
            self.kept_indices = sorted(kept_indices) 
            self.idx_to_pred = idx_to_pred
            self.idx_to_fname = idx_to_fname

        def __len__(self):
            return len(self.kept_indices)

        def __getitem__(self, i):
            orig_idx = self.kept_indices[i]
            x, _, _ = self.original_dataset[orig_idx]
            new_label = self.idx_to_pred[orig_idx]
            fname = self.idx_to_fname[orig_idx]
            return x, new_label, fname

    new_dataset = FilteredLabeledDataset(
        original_dataset=dataset,
        kept_indices=final_kept_indices,
        idx_to_pred=idx_to_pred,
        idx_to_fname=idx_to_fname
    )

    return new_dataset


def self_test_model(model, mitigation, clean_set, batch_size, num_workers, device, majority_label):
    dataloader = torch.utils.data.DataLoader(clean_set, 
                                             batch_size=batch_size, 
                                             num_workers=num_workers)

    model.eval()

    correct_total = 0
    num_samples = 0

    with torch.no_grad():
        for x, y, fname in tqdm(dataloader):
            preprocess_x = x.to(device)
            
            output_logits = model(preprocess_x).cpu()  # shape: [B, C]
            
            if y is not None:
                predictions = output_logits.argmax(dim=1)  # shape: [B]
                
                _, top2_indices = output_logits.topk(2, dim=1)
                
                for i in range(len(predictions)):
                    if predictions[i].item() == majority_label:
                        predictions[i] = top2_indices[i, 1]
                
                correct_total += (predictions == y).sum().item()
                num_samples += y.size(0)

    if num_samples > 0:
        accuracy = correct_total / num_samples
        print(f"Accuracy: {accuracy:.4f}")
    else:
        print("No labels available to compute accuracy.")


# Executes in mitigate mode, generating an approach to mitigate the model
def run_mitigate_mode(args):

    # Example for how to access training dataset new example data
    base_training_dataset_dirpath = args.round_training_dataset_dirpath
    # demo_training_example_data(base_training_dataset_dirpath)



    # Validate config file against schema
    with open(args.metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(args.schema_filepath) as schema_file:
        schema_json = json.load(schema_file)

    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)

    model = prepare_model(args.model_filepath, args.device)
    mitigation = prepare_mitigation(args, config_json)
    dataset = prepare_dataset(args.dataset_dirpath, split_name='train', require_label=True)

    mitigate_model(model, mitigation, dataset, args.output_dirpath, args.model_output_name)


def check_label_flip(results, results_r):

    sample_ids = list(results.keys())
    
    preds_original = {}
    preds_new = {}
    for sid in sample_ids:
        logits_orig = results[sid]
        logits_new = results_r[sid]
        
        pred_orig = int(max(range(len(logits_orig)), key=lambda i: logits_orig[i]))
        pred_new  = int(max(range(len(logits_new)),  key=lambda i: logits_new[i]))
        
        preds_original[sid] = pred_orig
        preds_new[sid] = pred_new
    
    label_flips = [sid for sid in sample_ids if preds_original[sid] != preds_new[sid]]
    
    if not label_flips:
        print("No label flips. Return results.")
        return results

    original_classes_when_flipped = [preds_original[sid] for sid in label_flips]
    cnt = Counter(original_classes_when_flipped)
    
    top_two = cnt.most_common(2)
    total_flip_count = len(label_flips)
    
    most_common_class, most_common_count = top_two[0]
    ratio_top1 = most_common_count / total_flip_count

    total_top2_ratio = ratio_top1 + ratio_top2

    all_original_classes = set(preds_original.values())
    num_classes = len(all_original_classes) if len(all_original_classes) > 0 else 1
    uniform_ratio_for_two_classes = 2 / num_classes

    ratio_coeff = total_top2_ratio / uniform_ratio_for_two_classes

    if total_flip_count > 100 and ratio_coeff > 6.6:
        print("backdoor!")
        return 2
    elif total_flip_count > 100 and ratio_coeff < 1.7:
        print("clean!")
        return 0
    else:
        return 1



# Executes in test model, outputting model logits for each example
def run_test_mode(args):
    # Validate config file against schema
    with open(args.metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(args.schema_filepath) as schema_file:
        schema_json = json.load(schema_file)

    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)


    # poi_set = prepare_dataset("/N/slate/zwa2/trojai-datasets/sts-dataset/models/id-00000000/new-poisoned-example-data", split_name='test', require_label=True)

    # clean_set = prepare_dataset("/N/slate/zwa2/trojai-datasets/sts-dataset/models/id-00000000/new-clean-example-data", split_name='test', require_label=True)
    # torch.save(clean_set, "clean_set_0.pth")

    # clean_set = prepare_dataset("/N/slate/zwa2/trojai-datasets/sts-dataset/models/id-00000001/new-clean-example-data", split_name='test', require_label=True)
    # torch.save(clean_set, "clean_set_1.pth")

    # clean_set = prepare_dataset("/N/slate/zwa2/trojai-datasets/sts-dataset/models/id-00000002/new-clean-example-data", split_name='test', require_label=True)
    # torch.save(clean_set, "clean_set_2.pth")

    # torch.save(poi_set, "poi_set.pth")

    # clean_set = torch.load("clean_set.pth")
    # poi_set = torch.load("poi_set.pth")
    # poi_set = None
    # exit(0)


    model = prepare_model(args.model_filepath, args.device)
    mitigation = prepare_mitigation(args, config_json)
    dataset = prepare_dataset(args.dataset_dirpath, split_name='test')

    results = test_model(model, mitigation, dataset, args.batch_size, args.num_workers, args.device)


    pred_labels = []
    for k, logits in results.items():
        pred_labels.append(np.argmax(logits))
    counter = Counter(pred_labels)
    most_common_class, highest_count = counter.most_common(1)[0]
    num_samples = len(results) 
    max_ratio = highest_count / num_samples
    num_classes = len(next(iter(results.values())))
    uniform_ratio = 1.0 / num_classes
    ratio_over_uniform = max_ratio / uniform_ratio
    print("k = ", ratio_over_uniform)

    labeled_dataset = build_label(model, dataset, args.batch_size, args.num_workers, args.device)
    model = mitigate_model_online(model, mitigation, labeled_dataset)

    # labeled_dataset = build_label(model, dataset, args.batch_size, args.num_workers, args.device)
    # model = mitigate_model_online(model, mitigation, labeled_dataset)

    # labeled_dataset = build_label(model, dataset, args.batch_size, args.num_workers, args.device)
    # model = mitigate_model_online(model, mitigation, labeled_dataset)

    results_r = test_model(model, mitigation, dataset, args.batch_size, args.num_workers, args.device)

    is_backdoor = check_label_flip(results, results_r)

    model_name = model.__class__.__name__
    
    # label_flip_count = 0
    # total_samples = len(results)

    # for filename in results:
    #     pred_label = np.argmax(results[filename])
    #     pred_label_r = np.argmax(results_r[filename])
    #     if pred_label != pred_label_r:
    #         label_flip_count += 1

    # label_flip_ratio = label_flip_count / total_samples
    # print("Label flip ratio:", label_flip_ratio)

    # if label_flip_ratio > 0.02:
    #     with open(os.path.join(args.output_dirpath, "results.json"), 'w+') as f:
    #         json.dump(results_r, f)
    # else:
    #     with open(os.path.join(args.output_dirpath, "results.json"), 'w+') as f:
    #         json.dump(results, f)

    with open(os.path.join(args.output_dirpath, "results.json"), 'w+') as f:
        json.dump(results_r, f)

    # clean_set = torch.load("clean_set_1.pth")
    # self_test_model(model, mitigation, clean_set, args.batch_size, args.num_workers, args.device, most_frequent_label)



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Parser for mitigation round, with two modes of operation, mitigate and test')

    parser.set_defaults(func=lambda args: parser.print_help())

    subparser = parser.add_subparsers(dest='cmd', required=True)

    mitigate_parser = subparser.add_parser('mitigate', help='Generates a mitigated model')

    # Mitigation arguments
    mitigate_parser.add_argument("--metaparameters_filepath", type=str, required=True, help="Path JSON file containing values of tunable parameters based on json schema")
    mitigate_parser.add_argument("--schema_filepath", type=str, help="Path to a schema file in JSON Schema format against which to validate the metaparameters file.", required=True)
    mitigate_parser.add_argument('--model_filepath', type=str, default="./model.pt", help="File path to the model that will be mitigated")
    mitigate_parser.add_argument('--dataset_dirpath', type=str, help="A dataset of examples to train the mitigated model with.", required=True)
    mitigate_parser.add_argument('--output_dirpath', type=str, default="./out", help="The directory path to where the output will be dumped")
    mitigate_parser.add_argument('--model_output_name', type=str, default="mitigated.pt", help="Name of the mitigated model that will be written to the output dirpath")
    mitigate_parser.add_argument('--scratch_dirpath', type=str, default="./scratch", help="The directory where a scratch space is located.")
    mitigate_parser.add_argument('--batch_size', type=int, default=32, help="The batch size that the technique would use for data loading")
    mitigate_parser.add_argument('--device', type=str, default='cuda', help="The device to use")
    mitigate_parser.add_argument('--num_workers', type=int, default=1, help="The number of CPU processes to use to load data")
    mitigate_parser.add_argument("--round_training_dataset_dirpath", type=str, help="File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.", default=None)

    # Test arguments
    test_parser = subparser.add_parser('test', help='Tests a mitigated model with example data')
    test_parser.add_argument("--metaparameters_filepath", type=str, required=True, help="Path JSON file containing values of tunable parameters based on json schema")
    test_parser.add_argument("--schema_filepath", type=str, help="Path to a schema file in JSON Schema format against which to validate the metaparameters file.", required=True)
    test_parser.add_argument('--model_filepath', type=str, default="./model.pt", help="File path to the mitigated model that will be tested")
    test_parser.add_argument('--dataset_dirpath', type=str, help="A dataset of examples to test the mitigated model with.", required=True)
    test_parser.add_argument('--scratch_dirpath', type=str, default="./scratch", help="The directory where a scratch space is located.")
    test_parser.add_argument('--output_dirpath', type=str, default="./out", help="The directory path to where the output will be dumped")
    test_parser.add_argument('--batch_size', type=int, default=32, help="The batch size that the technique would use for data loading")
    test_parser.add_argument('--device', type=str, default='cuda', help="The device to use")
    test_parser.add_argument('--num_workers', type=int, default=1, help="The number of CPU processes to use to load data")
    test_parser.add_argument("--round_training_dataset_dirpath", type=str, help="File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.", default=None)

    # Setup default function to call for mitigate/test
    mitigate_parser.set_defaults(func=run_mitigate_mode)
    test_parser.set_defaults(func=run_test_mode)

    args = parser.parse_args()

    # Call appropriate function
    args.func(args)
        
