# TeleXGI
TeleXGI: Explainable Gastrointestinal Image Classification for TeleSurgery Systems

# Downloading Kaggle Datasets

To download Kaggle datasets using notebook, follow these steps:

## 1. Install Kaggle API

Make sure you have the Kaggle API installed. If not, you can install it via pip:

```bash
!pip install kaggle
```

## 2. Obtain Kaggle API Credentials

You need to have Kaggle API credentials. Here's how to get them:

- Go to the [Kaggle website](https://www.kaggle.com) and sign in to your account.
- Click on your profile picture at the top right and select "Account".
- Scroll down to the "API" section and click on "Create New API Token".
- This will download a file named `kaggle.json`.

## 3. Configure Kaggle API

Move the downloaded `kaggle.json` file to the `~/.kaggle/` directory:

```bash
!mkdir ~/.kaggle
!mv /path/to/downloaded/kaggle.json ~/.kaggle/
```
Ensure that permissions are set correctly for the kaggle.json file:

```bash
!chmod 600 ~/.kaggle/kaggle.json
```

## 4. Download Required Kaggle Datasets

```bash
!kaggle datasets download -d meetpatel0963/kvasir-v2
!kaggle datasets download -d meet0963/kvasir-v2-folds
!kaggle datasets download -d mp10201/kvasir-v2-resnet50-epochs-100
```