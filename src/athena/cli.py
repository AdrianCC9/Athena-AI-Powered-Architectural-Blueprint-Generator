import argparse
from pathlib import Path

from . import __version__
from .defaults import (
    DEFAULT_ARTIFICIAL_PKL,
    DEFAULT_CHECKPOINT,
    DEFAULT_CLEANED_CSV,
    DEFAULT_GENERATED_IMAGE,
    DEFAULT_IMAGES_DIR,
    DEFAULT_LOSS_LOG,
    DEFAULT_PROCESSED_IMAGES,
    DEFAULT_TOKENIZED_TEXTS,
    OUTPUTS_DIR,
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="athena",
        description="Train and run the Athena floorplan generator.",
    )
    parser.add_argument("--version", action="version", version=f"athena {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("clean", help="Clean Tell2Design annotations into a CSV.")
    clean.add_argument("--input-pkl", type=Path, default=DEFAULT_ARTIFICIAL_PKL)
    clean.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    clean.add_argument("--output-csv", type=Path, default=DEFAULT_CLEANED_CSV)
    clean.set_defaults(func=cmd_clean)

    preprocess = subparsers.add_parser("preprocess-images", help="Convert images into tensors.")
    preprocess.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    preprocess.add_argument("--output", type=Path, default=DEFAULT_PROCESSED_IMAGES)
    preprocess.add_argument("--size", type=int, default=256)
    preprocess.set_defaults(func=cmd_preprocess_images)

    tokenize = subparsers.add_parser("tokenize", help="Tokenize cleaned descriptions.")
    tokenize.add_argument("--input-csv", type=Path, default=DEFAULT_CLEANED_CSV)
    tokenize.add_argument("--output", type=Path, default=DEFAULT_TOKENIZED_TEXTS)
    tokenize.add_argument("--model-name", default="t5-small")
    tokenize.add_argument("--max-length", type=int, default=128)
    tokenize.set_defaults(func=cmd_tokenize)

    prepare = subparsers.add_parser("prepare", help="Run clean, image preprocessing, and tokenization.")
    prepare.add_argument("--input-pkl", type=Path, default=DEFAULT_ARTIFICIAL_PKL)
    prepare.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    prepare.add_argument("--cleaned-csv", type=Path, default=DEFAULT_CLEANED_CSV)
    prepare.add_argument("--processed-images", type=Path, default=DEFAULT_PROCESSED_IMAGES)
    prepare.add_argument("--tokenized-texts", type=Path, default=DEFAULT_TOKENIZED_TEXTS)
    prepare.add_argument("--model-name", default="t5-small")
    prepare.add_argument("--max-length", type=int, default=128)
    prepare.add_argument("--size", type=int, default=256)
    prepare.set_defaults(func=cmd_prepare)

    train = subparsers.add_parser("train", help="Train the Athena model.")
    train.add_argument("--text-tokens", type=Path, default=DEFAULT_TOKENIZED_TEXTS)
    train.add_argument("--image-tensors", type=Path, default=DEFAULT_PROCESSED_IMAGES)
    train.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    train.add_argument("--loss-log", type=Path, default=DEFAULT_LOSS_LOG)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=0.0002)
    train.add_argument("--device", default=None)
    train.add_argument("--limit", type=int, default=None)
    train.add_argument("--augment", action="store_true")
    train.add_argument("--model-name", default="t5-small")
    train.add_argument("--no-pretrained-backbone", action="store_true")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint.")
    evaluate.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    evaluate.add_argument("--text-tokens", type=Path, default=DEFAULT_TOKENIZED_TEXTS)
    evaluate.add_argument("--image-tensors", type=Path, default=DEFAULT_PROCESSED_IMAGES)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--max-samples", type=int, default=5000)
    evaluate.add_argument("--device", default=None)
    evaluate.add_argument("--output-json", type=Path, default=None)
    evaluate.add_argument("--model-name", default="t5-small")
    evaluate.set_defaults(func=cmd_evaluate)

    generate = subparsers.add_parser("generate", help="Generate a blueprint image from a prompt.")
    generate.add_argument("prompt")
    generate.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    generate.add_argument("--output", type=Path, default=DEFAULT_GENERATED_IMAGE)
    generate.add_argument("--reference-image", type=Path, default=None)
    generate.add_argument("--device", default=None)
    generate.add_argument("--size", type=int, default=256)
    generate.add_argument("--max-length", type=int, default=128)
    generate.add_argument("--model-name", default="t5-small")
    generate.set_defaults(func=cmd_generate)

    visualize = subparsers.add_parser("visualize", help="Create or show a small image preview grid.")
    visualize.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    visualize.add_argument("--num-images", type=int, default=6)
    visualize.add_argument("--output", type=Path, default=OUTPUTS_DIR / "image_preview.png")
    visualize.add_argument("--show", action="store_true")
    visualize.set_defaults(func=cmd_visualize)

    demo = subparsers.add_parser("demo", help="Launch a lightweight browser demo.")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8000)
    demo.add_argument(
        "--prompt",
        default="two bedroom apartment with open kitchen, living room, and one bathroom",
    )
    demo.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Write a standalone HTML demo instead of starting the server.",
    )
    demo.add_argument("--no-browser", action="store_true", help="Do not open a browser tab.")
    demo.set_defaults(func=cmd_demo)

    return parser


def cmd_clean(args):
    from .cleaning_data import clean_annotations

    clean_annotations(args.input_pkl, args.images_dir, args.output_csv)


def cmd_preprocess_images(args):
    from .image_processing import preprocess_images

    preprocess_images(args.images_dir, args.output, image_size=(args.size, args.size))


def cmd_tokenize(args):
    from .text_tokenization import tokenize_csv

    tokenize_csv(args.input_csv, args.output, model_name=args.model_name, max_length=args.max_length)


def cmd_prepare(args):
    from .cleaning_data import clean_annotations
    from .image_processing import preprocess_images
    from .text_tokenization import tokenize_csv

    clean_annotations(args.input_pkl, args.images_dir, args.cleaned_csv)
    preprocess_images(args.images_dir, args.processed_images, image_size=(args.size, args.size))
    tokenize_csv(
        args.cleaned_csv,
        args.tokenized_texts,
        model_name=args.model_name,
        max_length=args.max_length,
    )


def cmd_train(args):
    from .training import train

    train(
        args.text_tokens,
        args.image_tensors,
        args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        augment=args.augment,
        limit=args.limit,
        text_model_name=args.model_name,
        pretrained_backbone=not args.no_pretrained_backbone,
        loss_log_path=args.loss_log,
    )


def cmd_evaluate(args):
    from .evaluate_model import evaluate

    evaluate(
        args.checkpoint,
        args.text_tokens,
        args.image_tensors,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
        output_json=args.output_json,
        text_model_name=args.model_name,
    )


def cmd_generate(args):
    from .generation import generate_blueprint

    generate_blueprint(
        args.prompt,
        args.checkpoint,
        args.output,
        reference_image=args.reference_image,
        device=args.device,
        image_size=(args.size, args.size),
        max_length=args.max_length,
        text_model_name=args.model_name,
    )


def cmd_visualize(args):
    from .data_visualization import visualize_images

    output = None if args.show else args.output
    visualize_images(args.images_dir, num_images=args.num_images, output_path=output)


def cmd_demo(args):
    from .demo import export_demo_html, serve_demo

    if args.export:
        output_path = export_demo_html(args.prompt, args.export)
        print(f"[INFO] Demo exported to {output_path}")
        return

    serve_demo(args.host, args.port, prompt=args.prompt, open_browser=not args.no_browser)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ModuleNotFoundError as exc:
        missing = exc.name or "a dependency"
        parser.exit(
            1,
            f"Missing dependency: {missing}. Install ML dependencies with `pip install -e \".[ml]\"` first.\n",
        )


if __name__ == "__main__":
    main()
