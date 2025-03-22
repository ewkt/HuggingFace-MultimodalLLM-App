from setuptools import setup, find_packages

setup(
    name="multimodal",
    version="0.5.0",
    description="School project, building a streamlit app to interact with various HF llms",
    author="ewkt",
    packages=find_packages(),
    python_requires="~=3.10",
    install_requires=[
        "pip==25.0",
        "python-dotenv",
        "ruff==0.11.0",
        "tqdm",
        "typer==0.15.1",
        "transformers==4.48.3",
        "sentence-transformers==3.4.0",
        "faiss-cpu==1.9.0",
        "PyPDF2==2.10.3",
        "pillow==11.1.0",
        "streamlit==1.43.2",
        "torch==2.6.0",
    ],
    extras_require={
        "test": [
            "pytest>=8.3.3",
            "pytest-cov>=6.0.0",
            "pytest-asyncio>=0.23.5",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)