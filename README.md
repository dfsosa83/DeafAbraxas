# DeafAbraxas: Forex Prediction Project

DeafAbraxas is a forex prediction project that focuses on analyzing and predicting different forex pairs, such as JPY/USD and CHF/USD, across various time frame windows. The project aims to develop machine learning and deep learning models to capture patterns and trends in the forex market and generate accurate predictions.

## Project Overview

The main objectives of the DeafAbraxas project are:
- Collect and preprocess historical forex data for different currency pairs
- Engineer relevant features from the raw data to capture market dynamics
- Develop and train machine learning and deep learning models for forex prediction
- Evaluate and optimize the models' performance using appropriate metrics
- Deploy the trained models for real-time prediction and monitoring

## Forex Pairs and Time Frame Windows

The project will focus on the following forex pairs:
- JPY/USD (Japanese Yen / US Dollar)
- CHF/USD (Swiss Franc / US Dollar)
- [Add more forex pairs as needed]

The analysis and predictions will be performed across different time frame windows, such as:
- 30-minute
- 1-hour
- 4-hour
- Daily

By considering multiple time frame windows, the project aims to capture both short-term and long-term trends and patterns in the forex market.

## Repository Structure

The repository is structured as follows:
- `data/`: Contains the historical forex data and preprocessed datasets
- `notebooks/`: Jupyter notebooks for data exploration, analysis, and model development
- `src/`: Python scripts for data preprocessing, feature engineering, and model training
- `models/`: Trained machine learning and deep learning models
- `docs/`: Documentation and project-related resources

## Getting Started

To get started with the DeafAbraxas project, follow these steps:
1. Clone the repository: `git clone https://github.com/dfsosa83/DeafAbraxas.git`
2. Create and activate the virtual environment: `conda env create -f environment.yml` and `conda activate deafabrax`
3. Explore the Jupyter notebooks in the `notebooks/` directory to understand the data and models
4. Run the Python scripts in the `src/` directory to preprocess data, engineer features, and train models
5. Evaluate and optimize the trained models using the provided evaluation scripts

- Update the Conda Environment:

Open your terminal and navigate to the directory containing your environment.yml file. Then run the following command to update your environment:

- conda env update --name lolara --file environment.yml --prune

- Remove the Environment:
- conda env remove --name lolara
- conda env list

## Contributing

Contributions to the DeafAbraxas project are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

# upload to git repo: https://github.com/dfsosa83/DeafAbraxas.git

GitHub repository:

Open your terminal or command prompt.

Navigate to the directory where your project files are located using the cd command. For example:

cd C:\Users\david\OneDrive\Documents\DeafAbraxas
Initialize a new Git repository in your project directory (if you haven't already done so):

- git init
Stage all the changes in your project directory:

- git add .
This command adds all the modified and new files to the staging area.

Commit the changes with a meaningful commit message:

- git commit -m "Your commit message"
Replace "Your commit message" with a brief description of the changes you made.

Add the remote repository URL (in this case, your GitHub repository):

- git remote add origin https://github.com/dfsosa83/DeafAbraxas.git
This command adds the GitHub repository as a remote named "origin".

Push the changes to the GitHub repository:

- git push -u origin master
This command pushes the committed changes to the "master" branch of your GitHub repository. If prompted, enter your GitHub username and password.

Note: If you encounter an error saying that the remote repository is not empty, you can force push your changes using:

- git push -u origin master --force
However, be cautious when using the --force option as it overwrites the remote repository with your local changes.

After the push is successful, you can visit your GitHub repository page (https://github.com/dfsosa83/DeafAbraxas) to verify that your changes have been uploaded.

That's it! Your work should now be saved in your GitHub repository.

A few additional tips:

- It's a good practice to commit your changes frequently and push them to the remote repository regularly to keep your work backed up and accessible from anywhere.
- If you make further changes to your files, repeat steps 4-7 to stage, commit, and push the new changes.
- If you want to collaborate with others or work on your project from multiple machines, you can clone the repository using git clone - -https://github.com/dfsosa83/DeafAbraxas.git on the other machine and follow the same steps to push your changes.