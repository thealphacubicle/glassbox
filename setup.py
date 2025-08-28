from setuptools import setup, find_packages

setup(
    name="glassbox",
    version="0.1.0",
    description="Real-time observability for model training and hyperparameter tuning.",
    author="Srihari Raman",
    packages=find_packages(),
    install_requires=[
        # Add your runtime dependencies here, e.g.:
        # "numpy", "scikit-learn"
    ],
    python_requires=">=3.7",
)

