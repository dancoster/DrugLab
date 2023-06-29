NAME_ID_COL = "ITEMID"
ID_COL = "OldITEMID"

# MIMIC Dataset constants
LAB_MAPPING = {
    'Alanine aminotransferase': [50861, 769, 220644],
    'Fraction inspired oxygen': [189],
    'Asparate aminotransferase': [50878, 770, 220587],
    'Hemoglobin': [814, 220228, 51222, 50811],
    'Prothrombin time INR': [51237, 815, 1530, 227467],
    'Bilirubin': [51465, 50883, 803, 225651, 50885, 1538, 848, 225690, 50884],
    'Calcium': [786, 1522, 3746, 51029, 50893, 225625],
    'Calcium ionized': [50808, 816, 225667, 3766],
    'Creatinine': [791, 1525, 220615, 50912],
    'Glucose': [50931, 807, 811, 1529, 50809, 3745, 225664, 220621, 226537],
    'Lactic acid': [818, 225668, 1531],
    'Magnesium': [821, 1532, 220635, 50960],
    'Platelets': [51265, 828, 227457],
    'Potassium': [829, 1535, 227464, 50971, 50822],
    'Sodium': [837, 1536, 220645, 226534, 50983, 50824],
    'Uric Acid': [51007],
    'PTT': [825, 1533, 227466, 51275],
    'Hematocrit': [813, 220545, 51221, 50810],
    'Red blood cell': [51279, 833],
    'Albumin': [50862, 772, 1521, 227456],
    'Bicarbonate':[50882, 227443, 50803],
    'Blood urea nitrogen':[51006, 781, 1162, 225624],
    'Chloride':[50806, 220602, 226536, 1523, 788, 50902],
    'Lactate dehydrogenase': [50954],
    'Lymphocytes':[51244],
    'Mean corpuscular volume':[51250],
    'Neutrophils':[51256],
    'White blood cell count':[861, 1542, 220546, 51301, 51300, 1127],

    # Vital signs
    'Heart Rate': [211, 220045],
    'Diastolic blood pressure': [8368, 220051, 225310, 8555, 8441, 220180, 8502, 8440, 8503,
                                 8504,8507, 8506, 224643],
    'Systolic blood pressure': [51, 220050, 225309, 6701, 455, 220179, 3313, 3315, 442, 3317,
                                3323, 3321, 224167, 227243],
    'Temperature (C)': [3655, 677, 676, 223762],
    'Temperature (F)': [678, 223761, 679],
    'Respiratory rate': [618, 220210, 3603, 224689, 614, 651, 224422, 615, 224690, 619, 224688],
    'Oxygen saturation': [834, 50817, 8498, 220227, 646]
}

OLD_LAB_MAPPING = {
    'ALT': {'Alanine aminotransferase': [50861, 769, 220644]},
    'ANA': {'Fraction inspired oxygen': [189]},
    'AST': {'Asparate aminotransferase': [50878, 770, 220587]},
    'Hemoglobin': {'Hemoglobin percent': [50852],
                   'Hemoglobin C': [51224],
                   'Hemoglobin F': [51225],
                   'Hemoglobin A2': [51223],
                   'Hemoglobin': [814, 220228, 51222, 50811]},
    'INR': {'Prothrombin time INR': [51237, 815, 1530, 227467]},
    'bilirubin': {'Bilirubin': [51465,
                                50883,
                                803,
                                225651,
                                50885,
                                1538,
                                848,
                                225690,
                                50884],
                  #   'Bilirubin, Total, Pleural': [51049],
                  #   'Bilirubin, Total, Body Fluid': [51028],
                  #   'Bilirubin, Total, Ascites': [50838]
                  },
    'calcium': {'Calcium': [786, 1522, 3746, 51029, 50893, 225625],
                'Calcium ionized': [50808, 816, 225667, 3766],
                #   'Calcium urine': [51066, 51077]
                },
    'creatinine': {'Creatinine': [791, 1525, 220615, 50912],
                   'Creatinine ascites': [50841],
                   #   'Creatinine body fluid': [51032],
                   #   'Creatinine pleural': [51052],
                   #   'Creatinine urine': [51082]
                   },
    'glucose': {'Glucose': [50931,
                            807,
                            811,
                            1529,
                            50809,
                            3745,
                            225664,
                            220621,
                            226537],
                #   'Glucose urine': [51478],
                #   'Glucose, CSF': [51014],
                #   'Estimated Actual Glucose': [51529],
                #   'Glucose, Urine': [51084],
                #   'Glucose, Pleural': [51053],
                #   'Glucose, Joint Fluid': [51022],
                #   'Glucose, Ascites': [50842],
                #   'Glucose, Body Fluid': [51034]
                },
    'lactic acid': {'Lactic acid': [818, 225668, 1531]},
    'magnesium': {'Magnesium': [50960], 'Magnesium, Urine': [51088]},
    'platelets': {'Platelets': [51265, 828, 227457], 'Large Platelets': [51240]},
    'potassium': {'Potassium': [829, 1535, 227464, 50971, 50822],
                  #   'Potassium serum': [227442],
                  #   'Potassium, Body Fluid': [51041],
                  #   'Potassium, Pleural': [51057],
                  #   'Potassium, Stool': [51064],
                  #   'Potassium, Urine': [51097],
                  #   'Potassium, Ascites': [50847]
                  },
    'sodium': {'Sodium': [837, 1536, 220645, 226534, 50983, 50824],
               #   'Sodium, Ascites': [50848],
               #   'Sodium, Body Fluid': [51042],
               #   'Sodium, Pleural': [51058],
               #   'Sodium, Stool': [51065],
               #   'Sodium, Urine': [51100]
               },
    'Uric acid': {'Uric Acid': [51007], 'Uric Acid, Urine': [51105]},
    'B12': {'Vitamin B12': [51010]},
    'prolactin': {'Prolactin': [50973]},
    'Amylase': {'Amylase': [50867],
                #  'Amylase, Ascites': [50836], 'Amylase, Body Fluid': [51026],'Amylase, Joint Fluid': [51020],'Amylase, Pleural': [51047], 'Amylase, Urine': [51072]
                },
    'Lipase': {'Lipase': [50956],
               #  'Lipase, Ascites': [50844], 'Lipase, Body Fluid': [51036]
               },
    'Aptt': {'PTT': [825, 1533, 227466, 51275]},
    'Hematocrit': {'Hematocrit': [813, 220545, 51221, 50810]},
    'Red blood cell': {'Red blood cell': [51279, 833]},
    'Albumin': {'Albumin': [50862, 772, 1521, 227456]},
    'Magnesium': {'Magnesium': [821, 1532, 220635, 50960]},
    'CPK': {}
}
CHARTEVENT_PARTS = 52

LAB_VECT_COLS = ['ADMISSION_LOCATION',
                 'ADMISSION_TYPE',
                 'ADMITTIME',
                 'AGE',
                 'CHARTTIME',
                 'DEATHTIME',
                 'DIAGNOSIS',
                 'DISCHARGE_LOCATION',
                 'DISCHTIME',
                 'DOB',
                 'DOD',
                 'DOD_HOSP',
                 'DOD_SSN',
                 'EDOUTTIME',
                 'EDREGTIME',
                 'ETHNICITY',
                 'EXPIRE_FLAG',
                 'GENDER',
                 'HADM_ID',
                 'HAS_CHARTEVENTS_DATA',
                 'HOSPITAL_EXPIRE_FLAG',
                 'INSURANCE',
                 'ITEMID',
                 'LANGUAGE',
                 'MARITAL_STATUS',
                 'MIMICExtractName',
                 'RELIGION',
                 'SUBJECT_ID',
                 'VALUE',
                 'VALUENUM',
                 'VALUEUOM']

MIMIC_MED_MAPPING = {
    'Furosemide': [221794, 228340, 3439],
    'Spironolactone': [6296, 5962],
    'Omeprazole': [7745, 7252, 227694],
    'Pantoprazole': [8074,
                     44820,
                     46522,
                     46014,
                     1097,
                     1226,
                     7372,
                     46548,
                     6741,
                     6616,
                     2778,
                     6756,
                     46565,
                     40550,
                     1898,
                     41583,
                     225910,
                     46203,
                     40700],
    'Acetylsalycilic acid (aspirin)': [7325, 7325],
    'Paracetamole (acetaminophen)': [50856],
    'Warfarin': [225913],
    'Haloperidol': [221824],
    'Ceftriaxone': [90018, 225855],
    'Procainamide ': [50962, 45853],
}

MIMIC_DICT_MED_MAPPING = {id: k for k, v in MIMIC_MED_MAPPING.items() for id in v}
MIMIC_III_PREPROCESSED_PATH = "mimiciii/1.4/preprocessed"
MIMIC_III_RAW_PATH = "mimiciii/1.4/raw"
MIMIC_III_MED_PREPROCESSED_FILE_PATH = "mimiciii_med_preprocessed.csv"
MIMIC_III_MED_LAB_PAIRS = [('Acetylsalycilic acid (aspirin)', 'Hemoglobin'), ('Acetylsalycilic acid (aspirin)', 'platelets'), ('Amoxicilin-clavulanate', 'ALT'), ('Amoxicilin-clavulanate', 'AST'), ('Ceftriaxone', 'bilirubin'), ('Citalopram', 'sodium'), ('Clozapine', 'platelets'), ('Dabigatran', 'Aptt'), ('Esmoprazole', 'B12'), ('Fluoxetine', 'sodium'), ('Furosemide', 'magnesium'), ('Glibenclamide', 'glucose'), ('Glimepiride', 'glucose'), ('Haloperidol', 'prolactin'), ('Hydrochlorothiazide', 'Uric acid'), ('Hydrochlorothiazide', 'calcium'), ('Metformin', 'B12'), ('Metformin', 'lactic acid'), ('Omeprazole', 'B12'), ('Pantoprazole', 'B12'), ('Paracetamole (acetaminophen)', 'ALT'), ('Paroxetine', 'sodium'), ('Procainamide', 'ANA'), ('Quetiapine', 'prolactin'), ('Ramipril', 'potassium'), ('Rivaroxaban', 'INR'), ('Simvastatin', 'CPK'), ('Spironolactone', 'potassium'), ('Trimetoprim-sulphamethoxazole', 'creatinine'), ('Trimetoprim-sulphamethoxazole', 'potassium'), ('Valproic acid', 'Amylase'), ('Valproic acid', 'Lipase'), ('Valsartan', 'potassium'), ('Warfarin', 'INR'), ('Insulin - Regular', 'glucose'), ('Packed Red Blood Cells', 'Hemoglobin'), ('Calcium Gluconate (CRRT)', 'calcium'), ('Packed Red Blood Cells', 'Red blood cell'), ('Packed Red Blood Cells', 'Hematocrit'), ('Albumin', 'Albumin'), ('Albumin', 'Hematocrit'), ('Albumin 5%', 'Albumin'), ('Albumin 5%', 'Hematocrit'), ('Albumin 25%', 'Albumin'), ('Albumin 25%', 'Hematocrit'), ('Magnesium Sulfate', 'Magnesium')]
MIMIC_III_LABEVENT_PREPROCESSED = "lab_patient_data_with_mimic_extract_names_1.csv"
MIMIC_III_PREPROCESSED_LABDATA = "lab_patient_data_mimic_extract_2.csv"

#  HIRID Dataset constants
HIRID_MAPPING = {
    "Hemoglobin": [24000526, 24000548, 24000549, 20000900, 24000836],
    "ALT": [20002600],
    "AST": [24000330],
    "INR": [24000567],
    "bilirubin": [20004300, 24000560],
    "calcium": [24000522, 20005100],
    "creatinine": [20000600, 24000572, 24000573],
    "glucose": [20005110, 24000523, 24000585, 24000400],
    "magnesium": [24000230],
    "platelets": [20000110],
    "potassium": [20000500, 24000520, 24000833, 24000867],
    "sodium": [20000400, 24000519, 24000658, 24000835, 24000866],
    "Amylase": [24000427, 24000587],
    # "Lipase":[24000555],
    # "Aptt":[20004410],
    "albumin": [24000605],
    "magnesium": [24000230],
    'Respiratory rate': [5685, 310],
    'Heart rate':        [200],
    'Peripheral oxygen saturation': [4000],
    'Core body temperature': [410],
    'Invasive systolic arterial pressure': [100],
    'Invasive diastolic arterial pressure': [120],
    'Non-invasive systolic arterial pressure': [600],
    'Non-invasive diastolic arterial pressure': [620],
    'Pulmonary artery systolic pressure': [150],
    'Pulmonary artery diastolic pressure': [170],
    'Chloride': [24000439, 24000521],
    'Bicarbonate':[20004200],
    'Lactate Cerebral spinal': [24000564],
    'Lactate Venous blood': [24000732, 24000485],
    'Lactate Arterial blood': [24000524],
    'Lymphocytes':[24000480],
    'Neutrophils':[24000550]
}

vital_signs = ['Heart Rate', 'Respiratory rate','Oxygen saturation', 'Systolic blood pressure', 'Diastolic blood pressure', 'Temperature']
labs_bmp = ['Glucose','Potassium','Sodium','Chloride', 'Creatinine', 'Blood urea nitrogen', 'Bicarbonate', 'Calcium',
'Albumin', 'Lactate dehydrogenase','Magnesium','Lactic acid']
labs_cbc = ['Hematocrit','Hemoglobin', 'Platelets', 'White blood cell count', 'Red blood cell count',
'Mean corpuscular volume', 'Lymphocytes', 'Neutrophils']
labs_cauglation = ['Prothrombin time INR']

# labs_bmp = ['Blood urea nitrogen', 'Lactate dehydrogenase', 'Lactic acid']
# labs_cbc = ['Hematocrit', 'White blood cell count', 'Red blood cell count', 'Mean corpuscular volume']

HIRID_LAB_IDS = [l for k in HIRID_MAPPING.values() for l in k]

HIRID_MED_MAPPING = {
    "Insulin" : [1000963, 1000379, 1000381, 1000724, 15],
    "Vancomycin": [189, 331],
    "KCL": [1000398, 1001063, 1000080],
    "Intravenous blood transfusion of packed cells": [1000100, 1000743],
    "Glucose": [1000022, 1000690, 1000689, 1000544, 1000835, 1000746, 1000060, 1000567],
    "Magnesium Sulfate": [1000421],
    "Magnesiocard 5 mmol sachet": [1000420],
}

HIRID_TO_MIMIC_CONV = {
    'Bicarbonate':("","", 1),
    'Chloride':("","", 1),
    'Diastolic blood pressure':("","", 1),
    'Heart Rate':("","", 1),
    'Prothrombin time INR':("","", 1),
    'Lactic acid':("","", 1),
    'Oxygen saturation':("","", 1),
    'Platelets':("","", 1),
    'Potassium':("","", 1),
    'Respiratory rate':("","", 1),
    'Sodium':("","", 1),
    'Systolic blood pressure':("","", 1),
    'Temperature':("","", 1),
    # unit conversion
    'Neutrophils': ("","%",18),
    "Hemoglobin":("g/l", "g/dl", 0.1),
    "Magnesium": ("mmol/l", "mg/dL", 0.411),
    "Albumin": ("g/L", "gr/dL", 0.1),
    "Calcium": ("mmol/l", "mg/dL", 4.008),
    "Creatinine": ("umol/l","mg/dL", 0.01131),
    "Glucose": ("mmol/l", "mg/dL", 18),
    'Lymphocytes':("","",1),    
}


HIRID_MIMIC_FEATURE_MAPPING = {
    'Albumin [Mass/volume] in Serum or Plasma' : 'Albumin',
    'Bicarbonate [Moles/volume] in Arterial blood':'Bicarbonate',
    'Calcium [Moles/volume] in Blood': 'Calcium', 
    'Hemoglobin [Mass/volume] in blood': 'Hemoglobin',
    'Chloride [Moles/volume] in Blood': 'Chloride', 
    'Core body temperature': 'Temperature',
    'Creatinine [Moles/volume] in Blood': 'Creatinine',
    'Diastolic arterial pressure': 'Diastolic blood pressure',
    'Glucose [Moles/volume] in Serum or Plasma': 'Glucose',
    'Heart rate': 'Heart Rate',
    'INR in Blood by Coagulation assay': 'Prothrombin time INR', 
    'Lactate [Mass/volume] in blood': 'Lactic acid',
    'Lymphocytes [#/volume] in Blood': 'Lymphocytes', 
    'Magnesium [Moles/volume] in Blood': 'Magnesium',
    'Neutrophils/100 leukocytes in Blood': 'Neutrophils', 
    'Peripheral oxygen saturation': 'Oxygen saturation',  
    'Platelets [#/volume] in Blood': 'Platelets', 
    'Potassium [Moles/volume] in Blood': 'Potassium',
    'Respiratory rate': 'Respiratory rate',
    'Sodium [Moles/volume] in Blood': 'Sodium', 
    'Systolic arterial pressure': 'Systolic blood pressure'
}
