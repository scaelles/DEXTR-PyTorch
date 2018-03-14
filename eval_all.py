import os.path

from torch.utils.data import DataLoader
from evaluation.eval import eval_one_result
import dataloaders.pascal as pascal

exp_root_dir = './'

method_names = []
method_names.append('run_0')

if __name__ == '__main__':

    # Dataloader
    dataset = pascal.VOCSegmentation(transform=None, retname=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Iterate through all the different methods
    for method in method_names:
        results_folder = os.path.join(exp_root_dir, method, 'Results')

        filename = os.path.join(exp_root_dir, 'eval_results', method.replace('/', '-') + '.txt')
        if not os.path.exists(os.path.join(exp_root_dir, 'eval_results')):
            os.makedirs(os.path.join(exp_root_dir, 'eval_results'))

        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                val = float(f.read())
        else:
            print("Evaluating method: {}".format(method))
            jaccards = eval_one_result(dataloader, results_folder, mask_thres=0.8)
            val = jaccards["all_jaccards"].mean()

        # Show mean and store result
        print("Result for {:<80}: {}".format(method, str.format("{0:.1f}", 100*val)))
        with open(filename, 'w') as f:
            f.write(str(val))
