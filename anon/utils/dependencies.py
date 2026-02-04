from importlib.metadata import version, PackageNotFoundError
from packaging.requirements import Requirement
from packaging.version import parse


def check_dependencies(requirements_file):
    missing_dependencies = []
    nonmatching_versions = []

    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            requirement = Requirement(line)

            try:
                installed_version = version(requirement.name)
                if not parse(installed_version) in requirement.specifier:
                    nonmatching_versions.append((requirement, installed_version))
            except PackageNotFoundError:
                missing_dependencies.append(line)

    error_msg = ''
    if missing_dependencies:
        error_msg += f'Missing dependencies: {", ".join(missing_dependencies)}.\n'
    if nonmatching_versions:
        error_msg += f'The following packages are installed with a version that does not match the requirement:\n'
        for req, installed_version in nonmatching_versions:
            error_msg += f'Package: {req.name}, installed: {installed_version}, required: {str(req.specifier)}\n'

    if len(error_msg) > 0:
        raise ModuleNotFoundError(f'{error_msg}--Make sure to install {requirements_file} to run this code!--')

