from distutils.core import setup

setup(
    name="bootstraptools",
    version="1.0.0",
    description="Small Python utilities to apply the bootstrap",
    long_description=open("README.md").read(),
    author="Edoardo Cignoni",
    author_email="edoardo.cignoni96@gmail.com",
    install_requires=["numpy", "scipy"],
)
