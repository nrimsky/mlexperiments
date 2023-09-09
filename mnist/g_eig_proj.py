import torch as t
from train_mnist import (
    CNN,
    load_mnist_data,
    load_pure_number_pattern_data,
    CombinedDataLoader,
    test,
)
import argparse
import numpy as np
from utils import (
    orthogonal_complement,
    plot_pertubation_results,
)
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from matplotlib import pyplot as plt
from g_eig import hessian_eig_gauss_newton
from visualize_conv_kernels import save_frames_weights, make_conv_movies


def perturb_in_direction(
    fname,
    patterns_per_num,
    direction,
    n_p=50,
    just_return_proj_v=False,
    make_movie=False,
):
    """
    fname: checkpoint file name
    patterns_per_num: number of patterns per digit
    direction: direction to perturb in ('pattern' or 'number')
    n_p: number of vectors to use for projection
    """
    # Load model
    model = CNN(input_size=28)
    model.load_state_dict(t.load(fname))
    model.to(device="cuda")
    model.eval()

    # Load pure pattern number data
    data_loader_test_number, data_loader_test_pattern = load_pure_number_pattern_data(
        patterns_per_num, is_train=False
    )

    _, _, eigenvectors_number = hessian_eig_gauss_newton(
        model,
        data_loader_test_number,
        num_batches=50,
        device="cuda",
        n_top_vectors=n_p,
    )
    _, _, eigenvectors_pattern = hessian_eig_gauss_newton(
        model,
        data_loader_test_pattern,
        num_batches=50,
        device="cuda",
        n_top_vectors=n_p,
    )

    if direction == "number":
        orth = orthogonal_complement(eigenvectors_number)  # 3340 x 3340
    elif direction == "pattern":
        orth = orthogonal_complement(eigenvectors_pattern)  # 3340 # 3340

    if direction == "number":
        v = eigenvectors_pattern[-1]  # 3340
    elif direction == "pattern":
        v = eigenvectors_number[-1]  # 3340

    proj_v = np.matmul(orth, v)  # 3340

    if just_return_proj_v:
        return proj_v

    # get opacity 0.5 dataloader
    _, data_loader_05_test = load_mnist_data(patterns_per_num, opacity=0.5)

    t_values = np.linspace(0, 5, 50)

    # store results
    acc_results = []
    loss_results = []
    for idx, t_val in enumerate(t_values):
        # load model
        model = CNN(input_size=28)
        model.load_state_dict(t.load(fname))
        model.eval()
        model.to(device="cuda")
        # perturb parameters by t * proj_v
        params_vector = parameters_to_vector(model.parameters()).detach()
        pertubation = t.tensor(t_val * proj_v, dtype=t.float32).to(device="cuda")
        params_vector = params_vector + pertubation
        vector_to_parameters(params_vector, model.parameters())
        if make_movie:
            save_frames_weights(idx, model)
        # evaluating model
        op_05_accuracy, op_05_loss = test(
            model,
            data_loader_05_test,
            do_print=False,
            device="cuda",
            calc_loss=True,
            max_batches=100,
        )
        pure_num_acc, pure_num_loss = test(
            model,
            data_loader_test_number,
            do_print=False,
            device="cuda",
            calc_loss=True,
            max_batches=100,
        )
        pure_pattern_acc, pure_pattern_loss = test(
            model,
            data_loader_test_pattern,
            do_print=False,
            device="cuda",
            calc_loss=True,
            max_batches=100,
        )
        # print results
        print(
            f"t_val: {t_val:.2f}, direction: {direction}, op_05_acc: {op_05_accuracy:.6f}, pure_num_acc: {pure_num_acc:.6f}, pure_pattern_acc: {pure_pattern_acc:.6f}, op_05_loss: {op_05_loss:.6f}, pure_num_loss: {pure_num_loss:.6f}, pure_pattern_loss: {pure_pattern_loss:.6f}"
        )
        # store results
        acc_results.append((t_val, op_05_accuracy, pure_num_acc, pure_pattern_acc))
        loss_results.append((t_val, op_05_loss, pure_num_loss, pure_pattern_loss))

    # write results to textfile
    with open(f"perturbation_results_{direction}_acc.txt", "w") as f:
        for result in acc_results:
            f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")
    with open(f"perturbation_results_{direction}_loss.txt", "w") as f:
        for result in loss_results:
            f.write(f"{result[0]},{result[1]},{result[2]},{result[3]}\n")

    # plot results
    plot_pertubation_results(
        acc_results, f"perturbation_acc_results_{direction}.png", yaxis="Accuracy (%)"
    )
    plot_pertubation_results(
        loss_results, f"perturbation_loss_results_{direction}.png", yaxis="Loss"
    )
    if make_movie:
        make_conv_movies()


def get_hessian_eig_mnist(
    fname, patterns_per_num, opacity=0.5, use_mixed_dataloader=False
):
    """
    Load model from fname checkpoint and calculate eigenvalues
    """
    model = CNN(input_size=28)
    model.load_state_dict(t.load(fname))
    model.to(device="cuda")
    model.eval()
    _, data_loader_test = load_mnist_data(patterns_per_num, opacity)
    if use_mixed_dataloader:
        d1, d2 = load_pure_number_pattern_data(patterns_per_num, is_train=False)
        data_loader_test = CombinedDataLoader(d1, d2)
    hessian_eig_gauss_newton(
        model, data_loader_test, num_batches=30, device="cuda"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to model to load")
    parser.add_argument(
        "--preserve",
        help="preserve number or pattern performance",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--movie",
        help="generate movie visualizing conv kernels",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--opacity",
        help="opacity of patterns in loss data",
        type=float,
        default=0.5,
        required=False,
    )
    parser.add_argument(
        "--patterns_per_num",
        help="number of patterns per digit",
        type=int,
        default=10,
        required=False,
    )
    parser.add_argument(
        "--mixed",
        help="use mixed data loader",
        action="store_true",
        default=False,
        required=False,
    )
    args = parser.parse_args()
    model_path = args.model_path
    opacity = args.opacity
    patterns_per_num = args.patterns_per_num
    use_mixed_dataloader = args.mixed
    movie = args.movie
    if args.preserve is not None:
        perturb_in_direction(
            model_path, patterns_per_num, args.preserve, make_movie=movie
        )
    else:
        get_hessian_eig_mnist(
            model_path,
            patterns_per_num=patterns_per_num,
            opacity=opacity,
            use_mixed_dataloader=use_mixed_dataloader,
        )
