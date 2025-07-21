import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MDR Stratify API with USSD - Multi-Pathogen", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Common mappings across all models
COUNTRY_MAPPING = {
    "Argentina": 0, "Australia": 1, "Austria": 2, "Belgium": 3, "Brazil": 4, 
    "Bulgaria": 5, "Cameroon": 6, "Canada": 7, "Chile": 8, "China": 9, 
    "Colombia": 10, "Costa Rica": 11, "Croatia": 12, "Czech Republic": 13, 
    "Denmark": 14, "Dominican Republic": 15, "Egypt": 16, "El Salvador": 17, 
    "Estonia": 18, "Finland": 19, "France": 20, "Germany": 21, "Ghana": 22, 
    "Greece": 23, "Guatemala": 24, "Honduras": 25, "Hong Kong": 26, "Hungary": 27, 
    "India": 28, "Indonesia": 29, "Ireland": 30, "Israel": 31, "Italy": 32, 
    "Ivory Coast": 33, "Jamaica": 34, "Japan": 35, "Jordan": 36, "Kenya": 37, 
    "Korea, South": 38, "Kuwait": 39, "Latvia": 40, "Lebanon": 41, "Lithuania": 42, 
    "Malawi": 43, "Malaysia": 44, "Mauritius": 45, "Mexico": 46, "Morocco": 47, 
    "Namibia": 48, "Netherlands": 49, "New Zealand": 50, "Nicaragua": 51, 
    "Nigeria": 52, "Norway": 53, "Oman": 54, "Pakistan": 55, "Panama": 56, 
    "Philippines": 57, "Poland": 58, "Portugal": 59, "Puerto Rico": 60, "Qatar": 61, 
    "Romania": 62, "Russia": 63, "Saudi Arabia": 64, "Serbia": 65, "Singapore": 66, 
    "Slovak Republic": 67, "Slovenia": 68, "South Africa": 69, "Spain": 70, 
    "Sweden": 71, "Switzerland": 72, "Taiwan": 73, "Thailand": 74, "Tunisia": 75, 
    "Turkey": 76, "Uganda": 77, "Ukraine": 78, "United Kingdom": 79, 
    "United States": 80, "Venezuela": 81, "Vietnam": 82
}

WARD_MAPPING = {
    "Medical ward": 0,
    "Surgical ward": 1, 
    "ICU": 2,
    "Emergency": 3,
    "Pediatric ward": 4,
    "Clinic": 6,
    "NICU": 7,
    "Nursing home": 8
}

SPECIMEN_TYPE_MAPPING = {
    "Skin & Soft Tissue": 1,
    "Respiratory": 2,
    "Blood & Circulatory": 3,
    "Gastrointestinal": 4,
    "Urinary & Genital": 5,
    "Ascitic Fluid": 6,
    "ENT & CNS": 7,
    "Musculoskeletal & Bone": 8,
    "Other": 9
}

AGE_GROUP_MAPPING = {
    "85 and Over": 1,
    "65 to 84 Years": 2,
    "19 to 64 Years": 3,
    "13 to 18 Years": 4,
    "3 to 12 Years": 5,
    "0 to 2 Years": 6
}

IN_OUT_PATIENT_MAPPING = {
    "Inpatient": 0,
    "Outpatient": 1
}

SEX_MAPPING = {
    "Male": 0,
    "Female": 1
}

# Pathogen-specific mappings
PATHOGEN_MODELS = {
    "Staphylococcus aureus": {
        "model_path": "models/staphylococcus_aureus_model.pkl",
        "phenotype_mapping": {
            "MSSA": 1,
            "MRSA": 0
        },
        "antibiotics": [
            "Clindamycin", "Erythromycin", "Levofloxacin", "Linezolid", "Minocycline",
            "Tigecycline", "Vancomycin", "Ceftaroline", "Daptomycin", "Gentamicin",
            "Moxifloxacin", "Oxacillin", "Teicoplanin", "Trimethoprim sulfa"
        ]
    },
    "Escherichia coli": {
        "model_path": "models/escherichia_coli_model.pkl",
        "phenotype_mapping": {
            "SPM-Neg": 0, "GIM-Neg": 1, "ESBL": 2, "NDM-Neg": 3, "CTX-M-15": 4,
            "CTX-M-14": 5, "CTX-M-27": 6, "CMY-2": 7, "TEM-OSBL": 8, "CTX-M-32": 9, "TEM-52": 10
        },
        "antibiotics": [
            "Amikacin", "Amoxycillin clavulanate", "Ampicillin", "Cefepime", "Ceftazidime",
            "Levofloxacin", "Meropenem", "Piperacillin tazobactam", "Tigecycline"
        ]
    },
    "Klebsiella pneumoniae": {
        "model_path": "models/klebsiella_pneumoniae_model.pkl",
        "phenotype_mapping": {
            "GIM-Neg": 0, "ESBL": 1, "SPM-Neg": 2, "NDM-Neg": 3, "KPC": 4, "Unknown": 5
        },
        "antibiotics": [
            "Amikacin", "Amoxycillin clavulanate", "Ampicillin", "Cefepime",
            "Levofloxacin", "Meropenem", "Piperacillin tazobactam", "Tigecycline"
        ]
    },
    "Acinetobacter baumannii": {
        "model_path": "models/acinetobacter_baumannii_model.pkl",
        "phenotype_mapping": {
            "OXA-23": 0, "OXA-24": 1, "OXA-58": 2, "SPM-Neg": 3, "GIM-Neg": 4,
            "OXA-239": 5, "OXA-72": 6, "OXA-40": 7, "OXA-366": 8, "OXA-435": 9,
            "OXA-398": 10, "OXA-TYPE": 11, "OXA-437": 12, "OXA-420": 13,
            "OXA-440": 14, "OXA-397": 15, "Unknown": 16
        },
        "antibiotics": [
            "Amikacin", "Cefepime", "Ceftazidime", "Levofloxacin", "Meropenem", "Piperacillin tazobactam"
        ]
    },
    "Pseudomonas aeruginosa": {
        "model_path": "models/pseudomonas_aeruginosa_model.pkl",
        "phenotype_mapping": {
            "GIM-Neg": 0, "SPM-Neg": 1, "Unknown": 2
        },
        "antibiotics": [
            "Amikacin", "Cefepime", "Ceftazidime", "Levofloxacin", "Meropenem", "Piperacillin tazobactam"
        ]
    }
}

# USSD Session Management
class USSDSession:
    """Manages USSD session state"""
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {
            'step': 'start',
            'data': {},
            'menu_stack': []
        })
    
    def update_session(self, session_id: str, data: Dict[str, Any]):
        self.sessions[session_id] = data
        logger.info(f"Session {session_id} updated: step={data.get('step')}")
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session {session_id} cleared")

# Updated USSD Menu Definitions
MAIN_MENU = {
    'text': 'MDR Stratify - Antibiotic Resistance Prediction\n1. New Prediction\n2. Help\n3. About',
    'options': {'1': 'new_prediction', '2': 'help', '3': 'about'}
}

PATHOGEN_MENU = {
    'text': 'Select Pathogen:\n1. Staphylococcus aureus\n2. Escherichia coli\n3. Klebsiella pneumoniae\n4. Acinetobacter baumannii\n5. Pseudomonas aeruginosa',
    'options': {
        '1': 'Staphylococcus aureus',
        '2': 'Escherichia coli', 
        '3': 'Klebsiella pneumoniae',
        '4': 'Acinetobacter baumannii',
        '5': 'Pseudomonas aeruginosa'
    }
}

# Dynamic phenotype menus will be generated based on selected pathogen

COUNTRY_MENU = {
    'text': 'Select Country (1-20):\n1. Argentina\n2. Australia\n3. Austria\n4. Belgium\n5. Brazil\n6. Bulgaria\n7. Cameroon\n8. Canada\n9. Chile\n10. China\n11. Colombia\n12. Costa Rica\n13. Croatia\n14. Czech Republic\n15. Denmark\n16. Dominican Republic\n17. Egypt\n18. El Salvador\n19. Estonia\n20. More...',
    'options': {str(i+1): list(COUNTRY_MAPPING.keys())[i] for i in range(min(20, len(COUNTRY_MAPPING)))}
}

WARD_MENU = {
    'text': 'Ward/Department:\n1. Medical ward\n2. Surgical ward\n3. ICU\n4. Emergency\n5. Pediatric ward\n6. Clinic\n7. NICU\n8. Nursing home',
    'options': {
        '1': 'Medical ward', '2': 'Surgical ward', '3': 'ICU',
        '4': 'Emergency', '5': 'Pediatric ward', '6': 'Clinic',
        '7': 'NICU', '8': 'Nursing home'
    }
}

SPECIMEN_MENU = {
    'text': 'Specimen Type:\n1. Skin & Soft Tissue\n2. Respiratory\n3. Blood & Circulatory\n4. Gastrointestinal\n5. Urinary & Genital\n6. Ascitic Fluid\n7. ENT & CNS\n8. Musculoskeletal & Bone\n9. Other',
    'options': {
        '1': 'Skin & Soft Tissue', '2': 'Respiratory', '3': 'Blood & Circulatory',
        '4': 'Gastrointestinal', '5': 'Urinary & Genital', '6': 'Ascitic Fluid',
        '7': 'ENT & CNS', '8': 'Musculoskeletal & Bone', '9': 'Other'
    }
}

AGE_GROUP_MENU = {
    'text': 'Patient Age Group:\n1. 0-2 Years\n2. 3-12 Years\n3. 13-18 Years\n4. 19-64 Years\n5. 65-84 Years\n6. 85+ Years',
    'options': {
        '1': '0 to 2 Years', '2': '3 to 12 Years', '3': '13 to 18 Years',
        '4': '19 to 64 Years', '5': '65 to 84 Years', '6': '85 and Over'
    }
}

SEX_MENU = {
    'text': 'Patient Gender:\n1. Male\n2. Female',
    'options': {'1': 'Male', '2': 'Female'}
}

PATIENT_TYPE_MENU = {
    'text': 'Patient Type:\n1. Inpatient\n2. Outpatient',
    'options': {'1': 'Inpatient', '2': 'Outpatient'}
}

# Initialize session manager
session_manager = USSDSession()

# Updated models
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
    pathogen: str
    phenotype: str
    overall_mdr_risk: bool
    mdr_confidence: float
    risk_level: str
    resistant_antibiotics: List[AntibioticResult]
    susceptible_antibiotics: List[AntibioticResult]
    total_resistant_count: int
    resistance_percentage: float

# Load models
def load_models():
    """Load all pathogen-specific models"""
    models = {}
    for pathogen, config in PATHOGEN_MODELS.items():
        try:
            model_path = config["model_path"]
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[pathogen] = pickle.load(f)
                    logger.info(f"Loaded model for {pathogen}")
            else:
                logger.warning(f"Model file not found for {pathogen}: {model_path}")
                models[pathogen] = None
        except Exception as e:
            logger.error(f"Error loading model for {pathogen}: {e}")
            models[pathogen] = None
    return models

models = load_models()

def encode_input(input_data: MDRPredictionInput) -> np.ndarray:
    """Convert text input to numeric encoding expected by the model"""
    try:
        pathogen_config = PATHOGEN_MODELS[input_data.pathogen]
        phenotype_mapping = pathogen_config["phenotype_mapping"]
        
        encoded_features = [
            phenotype_mapping.get(input_data.phenotype, 0),
            COUNTRY_MAPPING.get(input_data.country, 0),
            SEX_MAPPING.get(input_data.sex, 0),
            AGE_GROUP_MAPPING.get(input_data.age_group, 0),
            WARD_MAPPING.get(input_data.ward, 0),
            SPECIMEN_TYPE_MAPPING.get(input_data.specimen_type, 0),
            IN_OUT_PATIENT_MAPPING.get(input_data.in_out_patient, 0),
            input_data.year,
        ]
        print(encoded_features)


        return np.array(encoded_features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error encoding input: {str(e)}")

def interpret_predictions(predictions: np.ndarray, pathogen: str, phenotype: str) -> MDRPredictionOutput:
    """Interpret the model's antibiotic resistance predictions"""
    pathogen_config = PATHOGEN_MODELS[pathogen]
    antibiotic_names = pathogen_config["antibiotics"]
    num_antibiotics = len(antibiotic_names)

    if len(predictions) != num_antibiotics:
        raise ValueError(f"Expected {num_antibiotics} predictions for {pathogen}, but got {len(predictions)}")

    resistant_antibiotics = []
    susceptible_antibiotics = []

    for i, pred in enumerate(predictions):
        # Assuming predictions are probabilities, convert to binary
        is_resistant = pred > 0.5
        confidence = float(abs(pred - 0.5) + 0.5)  # Convert to confidence score

        antibiotic_result = AntibioticResult(
            name=antibiotic_names[i],
            resistance=bool(is_resistant),
            confidence=confidence
        )

        if is_resistant:
            resistant_antibiotics.append(antibiotic_result)
        else:
            susceptible_antibiotics.append(antibiotic_result)

    total_resistant = len(resistant_antibiotics)
    resistance_percentage = (total_resistant / num_antibiotics) * 100

    # Determine MDR status (3 or more resistant antibiotics)
    overall_mdr_risk = total_resistant >= 3
    mdr_confidence = min(0.95, max(0.5, resistance_percentage / 100))

    if resistance_percentage >= 50:
        risk_level = "High"
    elif resistance_percentage >= 25:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return MDRPredictionOutput(
        pathogen=pathogen,
        phenotype=phenotype,
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
    pathogen_config = PATHOGEN_MODELS[input_data.pathogen]
    num_antibiotics = len(pathogen_config["antibiotics"])
    
    # Generate mock probabilities instead of binary values
    base_resistance_probs = np.random.uniform(0.1, 0.9, size=num_antibiotics)
    
    # Adjust based on phenotype and pathogen characteristics
    if input_data.pathogen == "Staphylococcus aureus" and input_data.phenotype == "MRSA":
        # MRSA typically shows more resistance
        base_resistance_probs += 0.2
    elif input_data.pathogen == "Escherichia coli" and "ESBL" in input_data.phenotype:
        # ESBL strains show beta-lactam resistance
        base_resistance_probs[1:4] += 0.3  # Amoxicillin, Ampicillin, Cefepime
    
    # Adjust based on ward (ICU typically has more resistance)
    if input_data.ward == "ICU":
        base_resistance_probs += 0.1
    
    # Ensure probabilities are within [0, 1]
    base_resistance_probs = np.clip(base_resistance_probs, 0.0, 1.0)
    
    return base_resistance_probs

# Helper function to generate phenotype menu based on selected pathogen
def generate_phenotype_menu(pathogen: str) -> Dict:
    """Generate phenotype menu based on selected pathogen"""
    if pathogen not in PATHOGEN_MODELS:
        return {'text': 'Invalid pathogen selected', 'options': {}}
    
    phenotypes = list(PATHOGEN_MODELS[pathogen]["phenotype_mapping"].keys())
    menu_text = f'Select Phenotype for {pathogen}:\n'
    options = {}
    
    for i, phenotype in enumerate(phenotypes, 1):
        menu_text += f'{i}. {phenotype}\n'
        options[str(i)] = phenotype
    
    return {'text': menu_text.strip(), 'options': options}


@app.get("/variables")
def get_model_input_variables(pathogen_name: str):
    pathogen_model_map = {
        "Staphylococcus aureus": "models/staphylococcus_aureus_model.pkl",
        "Escherichia coli": "models/escherichia_coli_model.pkl",
        "Klebsiella pneumoniae": "models/klebsiella_pneumoniae_model.pkl",
        "Acinetobacter baumannii": "models/acinetobacter_baumannii_model.pkl",
        "Pseudomonas aeruginosa": "models/pseudomonas_aeruginosa_model.pkl",
    }
    if pathogen_name not in pathogen_model_map:
        raise ValueError(f"No model found for pathogen '{pathogen_name}'")

    model_path = pathogen_model_map[pathogen_name]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Try to extract feature names
    feature_names = []

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    elif hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                feature_names = list(step.feature_names_in_)
                break
    elif hasattr(model, "get_booster"):  # For XGBoost
        booster = model.get_booster()
        feature_names = booster.feature_names
    elif hasattr(model, "coef_"):
        feature_names = [f"feature_{i}" for i in range(len(model.coef_))]
    else:
        raise RuntimeError("Could not extract feature names from model")

    return feature_names



@app.post("/predict", response_model=MDRPredictionOutput)
async def predict_mdr(input_data: MDRPredictionInput):
    """Predict antibiotic resistance for a given pathogen"""
    try:
        # Validate pathogen
        if input_data.pathogen not in PATHOGEN_MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported pathogen: {input_data.pathogen}. Supported: {list(PATHOGEN_MODELS.keys())}"
            )
        
        # Validate phenotype for the selected pathogen
        pathogen_config = PATHOGEN_MODELS[input_data.pathogen]
        valid_phenotypes = list(pathogen_config["phenotype_mapping"].keys())
        if input_data.phenotype not in valid_phenotypes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid phenotype '{input_data.phenotype}' for {input_data.pathogen}. Valid options: {valid_phenotypes}"
            )

        # Encode input
        encoded_input = encode_input(input_data)
        
        # Get model for the specific pathogen
        model = models.get(input_data.pathogen)
        
        if model is None:
            logger.warning(f"Using mock prediction for {input_data.pathogen}")
            predictions = mock_prediction(input_data)
        else:
            # Make prediction using the actual model
            predictions = model.predict(encoded_input)[0]  # Assuming it returns probabilities
            print(predictions)
        
        # Interpret predictions
        result = interpret_predictions(predictions, input_data.pathogen, input_data.phenotype)
        
        logger.info(f"Prediction completed for {input_data.pathogen}: {result.total_resistant_count}/{len(pathogen_config['antibiotics'])} resistant")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/pathogens")
async def get_supported_pathogens():
    """Get list of supported pathogens and their phenotypes"""
    result = {}
    for pathogen, config in PATHOGEN_MODELS.items():
        result[pathogen] = {
            "phenotypes": list(config["phenotype_mapping"].keys()),
            "antibiotics": config["antibiotics"],
            "model_loaded": models.get(pathogen) is not None
        }
    return result

@app.get("/countries")
async def get_supported_countries():
    """Get list of supported countries"""
    return {"countries": list(COUNTRY_MAPPING.keys())}



# USSD Endpoints 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)