import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import os

app = FastAPI(title="MDR Stratify API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define encoding mappings
PATHOGEN_MAPPING = {
    "Staphylococcus aureus": 0
}

PHENOTYPE_MAPPING = {
    "MSSA": 1,
    "MRSA": 0
}

COUNTRY_MAPPING = {
    "Europe": 0,
    "Asia": 1,
    "North America": 2,
    "South America": 3,
    "Middle East": 4,
    "Oceania": 5,
    "Africa": 6
}

SEX_MAPPING = {
    "Female": 1,
    "Male": 0
}

AGE_GROUP_MAPPING = {
    "65-84 Years": 1,
    "85+ Years": 2,
    "13-18 Years": 5,
    "19-64 Years": 0,
    "3-12 Years": 3,
    "0-2 Years": 4,
    "Unknown": 6
}

WARD_MAPPING = {
    "Medicine General": 0,
    "Surgery General": 1,
    "ICU": 2,
    "Emergency Room": 3,
    "Pediatric General": 4,
    "Clinic / Office": 5,
    "Pediatric ICU": 6,
    "Nursing Home / Rehab": 7,
    "Unknown": 8
}

SPECIMEN_TYPE_MAPPING = {
    "Wound": 0, "Blood": 1, "Sputum": 2, "Abscess": 3, "Endotracheal aspirate": 4,
    "Gastric Abscess": 5, "Skin: Other": 6, "Ulcer": 7, "Urine": 8, "Bronchus": 9,
    "Bronchoalveolar lavage": 10, "Skin": 11, "Trachea": 12, "Cellulitis": 13,
    "Peritoneal Fluid": 14, "Respiratory: Other": 15, "Decubitus": 16, "Burn": 17,
    "Nose": 18, "Furuncle": 19, "Catheters": 20, "Exudate": 21, "Impetiginous lesions": 22,
    "Tissue Fluid": 23, "Thoracentesis Fluid": 24, "Abdominal Fluid": 25, "Ear": 26,
    "Intestinal: Other": 27, "Eye": 28, "Bone": 29, "Synovial Fluid": 30, "Lungs": 31,
    "Throat": 32, "None Given": 33, "Bodily Fluids": 34, "Carbuncle": 35, "Aspirate": 36,
    "HEENT: Other": 37, "Pleural Fluid": 38, "Respiratory: Sinuses": 39, "Muscle": 40,
    "Bladder": 41, "Genitourinary: Other": 42, "Gall Bladder": 43, "Vagina": 44,
    "Stomach": 45, "Drains": 46, "Urethra": 47, "CSF": 48, "Instruments: Other": 49,
    "Circulatory: Other": 50, "Kidney": 51, "Colon": 52, "Skeletal: Other": 53,
    "Integumentary (Skin Nail Hair)": 54, "Appendix": 55, "Liver": 56, "Pancreas": 57,
    "Mouth": 58, "Spinal Cord": 59, "Penis": 60, "Head": 61, "CNS: Other": 62,
    "Prostate": 63, "Rectum": 64, "Bile": 65, "Ureter": 66, "Heart": 67,
    "Lymph Nodes": 68, "Feces/Stool": 69, "Uterus": 70, "Peripheral Nerves": 71,
    "Blood Vessels": 72, "Diverticulum": 73, "Nails": 74, "Bone Marrow": 75,
    "Placenta": 76, "Testis": 77, "Brain": 78, "Fallopian Tubes": 79, "Hair": 80,
    "Cervix": 81, "Ovary": 82, "Nasopharyngeal Aspirate": 83, "Nasotracheal Aspirate": 84,
    "Lymphatic Fluid": 85, "Vas Deferens": 86, "Transtracheal Aspirate": 87,
    "Esophagus": 88, "Bronchiole": 89
}

IN_OUT_PATIENT_MAPPING = {
    "Inpatient": 0,
    "None Given": 1,
    "Outpatient": 2,
    "Other": 3
}

# Antibiotic names for result interpretation
ANTIBIOTIC_NAMES = [
    "Clindamycin", "Erythromycin", "Levofloxacin", "Linezolid", "Minocycline",
    "Tigecycline", "Vancomycin", "Ceftaroline", "Daptomycin", "Gentamicin",
    "Moxifloxacin", "Oxacillin", "Teicoplanin", "Trimethoprim sulfa"
]


class MDRPredictionInput(BaseModel):
    pathogen: str
    phenotype: str
    country: str
    sex: str
    age_group: str
    ward: str
    specimen_type: str
    in_out_patient: str
    year: int


class AntibioticResult(BaseModel):
    name: str
    resistance: bool
    confidence: float


class MDRPredictionOutput(BaseModel):
    overall_mdr_risk: bool
    mdr_confidence: float
    risk_level: str
    resistant_antibiotics: List[AntibioticResult]
    susceptible_antibiotics: List[AntibioticResult]
    total_resistant_count: int
    resistance_percentage: float


# Load the pickle model
MODEL_PATH = "models/mdr_model.pkl"


def load_model():
    """Load the trained MDR prediction model"""
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        else:
            return None
    except Exception as e:
        # print(f"Error loading model: {e}")
        return None


model = load_model()


@app.get("/")
async def root():
    return {"message": "MDR Stratify API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/mappings")
async def get_mappings():
    """Get all the mapping dictionaries for the frontend"""
    return {
        "pathogens": list(PATHOGEN_MAPPING.keys()),
        "phenotypes": list(PHENOTYPE_MAPPING.keys()),
        "countries": list(COUNTRY_MAPPING.keys()),
        "sexes": list(SEX_MAPPING.keys()),
        "age_groups": list(AGE_GROUP_MAPPING.keys()),
        "wards": list(WARD_MAPPING.keys()),
        "specimen_types": list(SPECIMEN_TYPE_MAPPING.keys()),
        "in_out_patient": list(IN_OUT_PATIENT_MAPPING.keys()),
        "antibiotics": ANTIBIOTIC_NAMES
    }


def encode_input(input_data: MDRPredictionInput) -> np.ndarray:
    """Convert text input to numeric encoding expected by the model"""
    try:
        encoded_features = [
            PATHOGEN_MAPPING.get(input_data.pathogen, 0),
            PHENOTYPE_MAPPING.get(input_data.phenotype, 0),
            COUNTRY_MAPPING.get(input_data.country, 0),
            SEX_MAPPING.get(input_data.sex, 0),
            AGE_GROUP_MAPPING.get(input_data.age_group, 0),
            WARD_MAPPING.get(input_data.ward, 0),
            SPECIMEN_TYPE_MAPPING.get(input_data.specimen_type, 0),
            IN_OUT_PATIENT_MAPPING.get(input_data.in_out_patient, 0),
            input_data.year
        ]

        return np.array(encoded_features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error encoding input: {str(e)}")


def interpret_predictions(predictions: np.ndarray) -> MDRPredictionOutput:
    """Interpret the model's antibiotic resistance predictions"""
    print(predictions)
    num_antibiotics = len(ANTIBIOTIC_NAMES)

    if len(predictions) != num_antibiotics:
        raise ValueError(f"Expected {num_antibiotics} predictions, but got {len(predictions)}")



    resistant_antibiotics = []
    susceptible_antibiotics = []

    for i in range(len(predictions)):
        pred = predictions[i]
        prob = 0.5

        antibiotic_result = AntibioticResult(
            name=ANTIBIOTIC_NAMES[i],
            resistance=bool(pred),
            confidence=float(prob)
        )

        if pred:
            resistant_antibiotics.append(antibiotic_result)
        else:
            susceptible_antibiotics.append(antibiotic_result)

    total_resistant = len(resistant_antibiotics)
    resistance_percentage = (total_resistant / num_antibiotics) * 100

    # Determine MDR status
    overall_mdr_risk = total_resistant >= 3  # if one is resistant to more than 1 drug
    mdr_confidence = min(0.95, max(0.5, resistance_percentage / 100))

    if resistance_percentage >= 50:
        risk_level = "High"
    elif resistance_percentage >= 25:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return MDRPredictionOutput(
        overall_mdr_risk=overall_mdr_risk,
        mdr_confidence=mdr_confidence,
        risk_level=risk_level,
        resistant_antibiotics=resistant_antibiotics,
        susceptible_antibiotics=susceptible_antibiotics,
        total_resistant_count=total_resistant,
        resistance_percentage=resistance_percentage
    )



def mock_prediction(input_data: MDRPredictionInput) -> np.ndarray:
    """Generate mock predictions for development"""
    # Create realistic mock predictions based on known resistance patterns
    base_resistance = np.random.choice(
        [0, 1], size=len(ANTIBIOTIC_NAMES), p=[0.7, 0.3])

    if input_data.phenotype == "MRSA":
        # MRSA typically shows more resistance
        base_resistance[11] = 1  # Oxacillin resistance
        base_resistance[1] = 1   # Erythromycin resistance
        base_resistance[2] = 1   # Levofloxacin resistance

    # Adjust based on ward (ICU typically has more resistance)
    if input_data.ward == "ICU":
        # Increase resistance probability
        for i in range(len(base_resistance)):
            if np.random.random() < 0.4:  # 40% chance to add resistance
                base_resistance[i] = 1

    return base_resistance


@app.post("/predict", response_model=MDRPredictionOutput)
async def predict_mdr(input_data: MDRPredictionInput):
    """
    Predict antibiotic resistance patterns for Staphylococcus aureus
    """
    try:
        # Encode input for model
        encoded_input = encode_input(input_data)

        if model is None:
            # Use mock prediction for development only
            predictions = mock_prediction(input_data)
        else:
            # Use actual model
            predictions = model.predict(encoded_input)[0]

           
        result = interpret_predictions(predictions)

        return result

    except Exception as e:
        # print(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
