import argparse

from jinja2 import Template

from src.helper.filepaths import FilePaths as FP


"""
Simple script to create a colext config for a given experiment. It injects the fl_algorithm,
the data configuration, and any further configuration
"""

def generate():

    parser = argparse.ArgumentParser()
    parser.add_argument("--fl_algorithm", required=True, type=str, help="FL algorithm to be tested")
    parser.add_argument("--data_config", required=True, type=str, help="Data partitioning folder")
    parser.add_argument("--more_config", required=False, default="", type=str,
                        help="Any additional parameters to be passed to the run scripts")
    args = parser.parse_args()
    args_dict = vars(args)

    with open(FP.COLEXT_TEMPLATE, "r") as file:
        template = Template(file.read())

    rendered_content = template.render(args_dict)

    with open(FP.COLEXT_CONFIG, "w") as file:
        file.write(rendered_content)

    print("YAML file generated successfully.")


if __name__ == "__main__":
    generate()
