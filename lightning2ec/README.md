# lightning2ec

`lightning2ec` is a Python pipeline to collocate **EarthCARE MSI/CPR observations** with **satellite lightning detections** from **MTG-LI** and **GOES-GLM**.

The pipeline processes a user-defined date range and automatically:
- queries **EarthCARE MSI and CPR data** from the ESA MAAP STAC catalogue
- (down)loads **lightning observations**
- performs spatial–temporal collocation
- outputs matched datasets for further analysis

## Getting Started

First clone the repository and navigate to the project directory:

```bash
git clone https://github.com/piskalab/Lightning2EarthCARE.git
cd Lightning2EarthCARE
```
You can install the required dependencies either using **conda** or **pip**.

**Option 1: Conda environment**<br>
Create the environment:
```bash
conda env create -f environment.yaml
```
Activate the environment:
```bash
conda activate Lightning2ec
```

**Option 2: pip installation**<br>
Create a Python environment:
```bash
python -m venv lightning2ec-env
source lightning2ec-env/bin/activate
```
Install the dependencies:
```bash
pip install -r requirements.txt
```

## Credentials

The pipeline requires credentials for:
- EUMETSAT Data Store (lightning data)
- ESA MAAP API (EarthCARE catalogue access)
All credentials are stored in a `credentials.txt` file located **one directory above** the `lightning2ec` package.

### Creating `credentials.txt`

Create a plain text file named `credentials.txt` with the following format:
```
EUMETSAT_KEY=your_eumetsat_key_here
EUMETSAT_SECRET=your_eumetsat_secret_here
CLIENT_ID=offline-token
CLIENT_SECRET=p1eL7uonXs6MDxtGbgKdPVRAmnGxHpVE
OFFLINE_TOKEN=your_earthcare_longlasting_token_here
```

**Notes**
- Do **not** use quotes around the values.  
- Ensure there are no trailing spaces or extra characters.

### EarthCARE access token

The pipeline uses a **long-lived offline token** to obtain temporary access tokens for the MAAP API.<br>
Generate your token here: https://portal.maap.eo.esa.int/ini/services/auth/token/90dToken.php<br>
Place the generated value in the OFFLINE_TOKEN field.

## Usage

Run the pipeline via the CLI module for a defined date range:
```bash
python -m lightning2ec.cli \
    --lightning-dir PATH_TO_LIGHTNING_DATA \
    --start-date 2025-08-01 \
    --end-date 2025-09-30
```
By default the pipeline processes **all supported lightning platforms**:
- MTG-I1
- GOES-16/19
- GOES-18

Example restricting to **MTG-LI only**:
```bash
python -m lightning2ec.cli \
    --lightning-dir "PATH_TO_LIGHTNING_DATA" \
    --start-date 2025-08-01 \
    --end-date 2025-09-30 \
    --lightning-platform MTG-I1
```
