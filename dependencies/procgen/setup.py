from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.join(SCRIPT_DIR, "procgen")

def determine_version():
    version = open(os.path.join(PACKAGE_ROOT, "version.txt"), "r").read().strip()
    sha = "unknown"
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR)
            .decode("ascii")
            .strip()
        )
    except Exception:
        pass
    if "GITHUB_REF" in os.environ:
        ref = os.environ["GITHUB_REF"]
        parts = ref.split("/")
        assert parts[0] == "refs"
        if parts[1] == "tags":
            tag = parts[2]
            assert (
                tag == version
            ), "mismatch in tag vs version, expected: %s actual: %s" % (
                tag,
                version,
            )
            return version
    if sha == "unknown":
        return version
    else:
        return version + "+" + sha[:7]

class DummyExtension(Extension):
    def __init__(self):
        Extension.__init__(self, "dummy", sources=[])

class custom_build_ext(build_ext):
    def run(self):
        if self.inplace:
            print("skipping inplace build, extension will be built on demand")
            return
        sys.path.append(PACKAGE_ROOT)
        import builder
        lib_dir = builder.build(package=True)
        
        for filename in ["libenv.so", "libenv.dylib", "env.dll"]:
            src = os.path.join(lib_dir, filename)
            dst = os.path.join(self.build_lib, "procgen", "data", "prebuilt", filename)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.replace(src, dst)

# The rest is handled by pyproject.toml, but we need these custom classes
setup(
    ext_modules=[DummyExtension()],
    cmdclass={"build_ext": custom_build_ext},
    version=determine_version(),
)