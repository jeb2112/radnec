import argparse
from cProfile import Profile
from pstats import SortKey,Stats

import sam_hf

with Profile() as profile:
# if True:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help=(
            "Path to the directory where masks will be output. Output will be either a folder "
            "of PNGs per image or a single json with COCO-style masks."
        ),
        default=None
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=False,
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
        default='vit_b'
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        required=False,
        help="The path to the SAM pretrained model.",
        default=None
    ),
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="The path to the SAM fine-tuned checkpoint to use for mask generation.",
        default=None
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        help="bbox|point",
        default='bbox'
    )
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
    parser.add_argument("--tag",type=str, default="",help="Tag word for file naming")
    parser.add_argument("--layer",type=str, default="",help="TC|WT annotation of output file")
    parser.add_argument("--orient",type=str, default="ax",help="orientation of 2d SAM slice")
    parser.add_argument('--debug',action = 'store_true',default=False)
    args = parser.parse_args()
    print(args)
# with Profile() as profile:
    sam_hf.main(args)

    if True:
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.TIME)
            .print_stats(15)
        )


